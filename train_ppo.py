"""
PPO を使用してカスタム Mario ゲーム環境で学習を行う

使用方法:
    python train_ppo.py --level Level1-1 --total-steps 2000000 --num-envs 8

重要な変更点 (2026-08):
    * エピソード上限は「ゲーム内フレーム数」。SkipFrame(4) の内側で数えるので
      --max-episode-steps 8000 は意思決定 2000 回に相当する。
      以前の 1000 ではステージの 1/4 で打ち切られ、ゴールに到達できなかった。
    * 観測は 4 フレームスタック (12, 84, 84)。速度・落下方向が判別できるようになる。
    * VecMonitor を有効化し、クリア率 / 到達距離 / 終了理由を毎ロールアウトで記録する。
    * --random-level を付けない限り、--level で指定した 1 ステージだけを学習する。
    * 継続学習では学習率を既定で定数にする（線形減衰を毎回リセットすると
      再開のたびに lr が枯れて実質学習が止まるため）。
"""

import os
import argparse
from collections import defaultdict, deque
from datetime import datetime
import re

import numpy as np

# BLAS / OpenBLAS のメモリ使用量を抑える
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# 描画ウィンドウなしで pygame を動かす
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from classes.wrappers import make_mario_env


# ----------------------------------------------------------------------------
# 学習率スケジュール
# ----------------------------------------------------------------------------
def linear_schedule(initial_value: float):
    """進行度 (1.0 -> 0.0) に合わせて学習率を線形減衰させる。"""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


# ----------------------------------------------------------------------------
# 環境生成
# ----------------------------------------------------------------------------
def make_env(level, rank, seed=0, max_episode_steps=8000, random_level=False, levels=None):
    """並列化用の環境生成関数。

    以前は make_env の中で level=None / random_level=True がハードコードされており、
    --level 引数が完全に無視され、毎エピソード levels/ 以下からランダムに
    ステージが選ばれていた。
    """
    def _init():
        env = make_mario_env(
            level=None if random_level else level,
            max_episode_steps=max_episode_steps,
            render_mode=None,
            random_level=random_level,
            levels=levels,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


# ----------------------------------------------------------------------------
# 学習の進捗を可視化するコールバック
# ----------------------------------------------------------------------------
class MarioStatsCallback(BaseCallback):
    """クリア率・到達距離・終了理由の内訳を TensorBoard とコンソールに記録する。

    これがないと「どこまで進めているのか」「死んでいるのか時間切れなのか」が
    まったく分からず、学習が進んでいるかどうかの判断ができない。
    """

    def __init__(self, window=100, verbose=0):
        super().__init__(verbose)
        self.progress = deque(maxlen=window)
        self.clears = deque(maxlen=window)
        self.recent_reasons = deque(maxlen=window)
        self.total_reasons = defaultdict(int)
        self.best_progress = 0.0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            # VecMonitor はエピソード終了時にだけ "episode" キーを挿入する
            if "episode" not in info:
                continue
            reason = info.get("reason", "unknown")
            progress = float(info.get("progress", 0.0))

            self.recent_reasons.append(reason)
            self.total_reasons[reason] += 1
            self.progress.append(progress)
            self.clears.append(1.0 if reason == "level_complete" else 0.0)
            self.best_progress = max(self.best_progress, progress)
        return True

    def _on_rollout_end(self) -> None:
        if not self.progress:
            return
        n = len(self.recent_reasons)
        self.logger.record("mario/progress_mean", float(np.mean(self.progress)))
        self.logger.record("mario/progress_max", float(np.max(self.progress)))
        self.logger.record("mario/progress_best_ever", self.best_progress)
        self.logger.record("mario/clear_rate", float(np.mean(self.clears)))
        for key in ("level_complete", "game_over", "timeout"):
            frac = sum(1 for r in self.recent_reasons if r == key) / n
            self.logger.record(f"mario/frac_{key}", frac)
        self.logger.record("mario/episodes_total", sum(self.total_reasons.values()))


# ----------------------------------------------------------------------------
# 学習本体
# ----------------------------------------------------------------------------
def train_ppo(
    level="Level1-1",
    total_steps=100000,
    num_envs=4,
    max_episode_steps=8000,
    random_level=False,
    levels=None,
    log_dir="./logs",
    model_dir="./models",
    device="auto",
    load_model_path=None,
    learning_rate=2.5e-4,
    lr_schedule="constant",
    n_steps=512,
    batch_size=256,
    n_epochs=4,
    ent_coef=0.01,
    seed=0,
):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    level_name = level.replace("-", "").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if load_model_path and os.path.exists(load_model_path):
        base_name = os.path.basename(load_model_path).replace(".zip", "")
        match = re.search(r'_(\d+)k_', base_name)
        if match:
            total_steps_k = int(match.group(1)) + total_steps // 1000
            model_name = f"mario_ppo_{level_name}_{total_steps_k}k_{timestamp}"
        else:
            model_name = f"{base_name}_plus_{total_steps // 1000}k_{timestamp}"
    else:
        model_name = f"mario_ppo_{level_name}_{total_steps // 1000}k_{timestamp}"

    log_path = os.path.join(log_dir, model_name)
    model_path = os.path.join(model_dir, model_name)

    agent_steps_per_ep = max_episode_steps // 4
    print("=" * 60)
    print("🎮 Mario PPO 学習を開始します")
    print("=" * 60)
    print(f"   - レベル          : {'ランダム' if random_level else level}")
    if random_level and levels:
        print(f"   - 候補レベル      : {levels}")
    print(f"   - 総ステップ数    : {total_steps:,}")
    print(f"   - 並列環境数      : {num_envs}")
    print(f"   - エピソード上限  : {max_episode_steps:,} ゲームフレーム "
          f"(= 意思決定 {agent_steps_per_ep:,} 回)")
    print(f"   - 学習率          : {learning_rate} ({lr_schedule})")
    print(f"   - ログ            : {log_path}")
    print(f"   - モデル保存先    : {model_path}.zip")
    print()

    print("🔧 環境を初期化中...")
    env_fns = [
        make_env(level, i, seed=seed, max_episode_steps=max_episode_steps,
                 random_level=random_level, levels=levels)
        for i in range(num_envs)
    ]
    if num_envs > 1:
        envs = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        envs = DummyVecEnv(env_fns)
    # ★ VecMonitor がないと ep_rew_mean / ep_len_mean がログに出ず、
    #    エピソード終了時の info も拾えない
    envs = VecMonitor(envs)

    eval_env = Monitor(make_mario_env(
        level=(levels[0] if (random_level and levels) else level),
        max_episode_steps=max_episode_steps,
        render_mode=None,
        random_level=False,
    ))

    lr = linear_schedule(learning_rate) if lr_schedule == "linear" else learning_rate

    if load_model_path and os.path.exists(load_model_path):
        print(f"🔄 既存のモデルを読み込んで学習を再開します: {load_model_path}")
        try:
            model = PPO.load(
                load_model_path,
                env=envs,
                device=device,
                custom_objects={"learning_rate": lr},
            )
        except (ValueError, KeyError, AssertionError) as e:
            print("❌ 既存モデルを読み込めませんでした:")
            print(f"   {e}")
            print("   観測空間(フレームスタック)とアクション空間(走りジャンプ追加)を")
            print("   変更したため、旧モデルとは互換性がありません。")
            print("   --load-model を外して新規に学習してください。")
            envs.close()
            eval_env.close()
            return
        model.tensorboard_log = log_path
    else:
        print("🤖 新しい PPO エージェントを初期化中...")
        model = PPO(
            policy="CnnPolicy",
            env=envs,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            # 以前は target_kl=0.02 で毎回 early stopping していたため解除
            target_kl=None,
            verbose=1,
            device=device,
            seed=seed,
            tensorboard_log=log_path,
        )

    print(f"   観測空間: {envs.observation_space}")
    print(f"   行動空間: {envs.action_space}")
    print("✅ 環境とモデルの初期化完了\n")
    print("🚀 学習を開始します...")
    print("-" * 60)

    try:
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, (total_steps // 10) // num_envs),
            save_path=model_dir,
            name_prefix=f"mario_ppo_{level_name}_checkpoint",
            save_replay_buffer=False,
        )
        stats_callback = MarioStatsCallback()

        model.learn(
            total_timesteps=total_steps,
            callback=[checkpoint_callback, stats_callback],
            progress_bar=True,
            reset_num_timesteps=False if load_model_path else True,
        )
        print("-" * 60)
        print("✅ 学習完了！\n")

        print(f"💾 モデルを保存中: {model_path}.zip")
        model.save(model_path)
        print("✅ モデル保存完了\n")

        print("📊 学習済みモデルで評価中...")
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=5, deterministic=True
        )
        print(f"   平均報酬: {mean_reward:.2f} (+/- {std_reward:.2f})")
        print()

        print("=" * 60)
        print("🎉 学習完了サマリー")
        print("=" * 60)
        print(f"✅ モデル保存先 : {model_path}.zip")
        print(f"✅ ログ         : {log_path}")
        print(f"✅ 平均報酬     : {mean_reward:.2f}")
        print(f"✅ 最高到達率   : {stats_callback.best_progress * 100:.1f}%")
        print(f"✅ 終了理由内訳 : {dict(stats_callback.total_reasons)}")
        print()
        print("次のステップ:")
        print(f"  tensorboard --logdir {log_path}")
        print(f"  python infer_ppo.py --model {model_path}.zip --level {level}")
        print()

    except KeyboardInterrupt:
        print("\n⚠️  学習を中断しました")
        model.save(f"{model_path}_interrupted")
        print(f"💾 保存しました: {model_path}_interrupted.zip")

    finally:
        envs.close()
        eval_env.close()


def main():
    parser = argparse.ArgumentParser(
        description="PPO を使用してカスタム Mario ゲームで学習を行う",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 1ステージ固定で 200万ステップ学習（推奨）
  python train_ppo.py --level Level1-1 --total-steps 2000000 --num-envs 8

  # 短い動作確認
  python train_ppo.py --total-steps 8192 --num-envs 2
        """
    )
    parser.add_argument("--level", type=str, default="Level1-1")
    parser.add_argument("--total-steps", type=int, default=2000000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=8000,
                        help="エピソード上限（ゲームフレーム数）。"
                             "人間が 4000 フレームでクリアするなら 8000 程度")
    parser.add_argument("--levels", type=str, default=None,
                         help="--random-level の抽選候補をカンマ区切りで限定する。"
                              "例: Level_hard_01,Level_hard_02,Level_hard_03")
    parser.add_argument("--random-level", action="store_true",
                        help="毎エピソード levels/ からランダムに選ぶ（既定は --level 固定）")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--model-dir", type=str, default="./models")
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--lr-schedule", type=str, default="constant",
                        choices=["constant", "linear"],
                        help="継続学習では constant 推奨（linear は再開のたびにリセットされる）")
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    train_ppo(
        level=args.level,
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        max_episode_steps=args.max_episode_steps,
        random_level=args.random_level,
        levels=[l.strip() for l in args.levels.split(",")] if args.levels else None,
        device=args.device,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        load_model_path=args.load_model,
        learning_rate=args.lr,
        lr_schedule=args.lr_schedule,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        ent_coef=args.ent_coef,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
