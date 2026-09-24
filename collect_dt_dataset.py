"""
学習済み PPO モデルでプレイし、Decision Transformer 学習用のトラジェクトリ
(state画像・action・reward・returns_to_go) を収集する。

DT は「様々な収益(return)に条件付けて行動を予測する」学習をするため、
最高性能のモデルだけでなく複数スキルレベルのモデル・確率的行動を混ぜて
収益に幅を持たせたデータセットにするのが望ましい。

使用方法:
    python collect_dt_dataset.py \
        --models models/mario_ppo_level11_300k_20260804_105943.zip,models/mario_ppo_level11_900k_20260806_131602.zip \
        --episodes-per-model 40 \
        --output-dir dt_dataset/frames \
        --output-meta dt_dataset/metadata.pkl

出力:
    - <output-dir>/ep{idx:05d}_t{t:05d}.png : 各ステップの状態画像 (3,84,84 相当を保存)
    - <output-meta> : pickle化された episode のリスト
        [{'image_paths': [...], 'actions': np.ndarray, 'rewards': np.ndarray,
          'returns_to_go': np.ndarray, 'level': str}, ...]
"""

import os
import argparse
import pickle
import random

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
from PIL import Image
from stable_baselines3 import PPO

# ★ 学習時 (train_ppo.py) とまったく同じラッパーを使う。
#   ここで定義を重複させると学習時と観測がずれ、性能が出ない。
from classes.wrappers import make_mario_env, latest_frame, StallGuard


def get_levels(levels_dir="./levels", exclude=None):
    # 壊れたレベルと、汎化検証用のホールドアウトレベルは既定で除く
    from classes.MarioGymEnv import MarioEnv
    if exclude is None:
        exclude = MarioEnv._BROKEN_LEVELS
    levels = []
    for f in sorted(os.listdir(levels_dir)):
        if f.endswith(".json"):
            name = f.replace(".json", "")
            if name not in exclude and not MarioEnv.is_heldout(name):
                levels.append(name)
    return levels


def collect_episode(model, level, max_frames, deterministic, frame_dir, ep_idx, epsilon=0.0):
    """1エピソード分のトラジェクトリを収集する。

    max_frames は「ゲームフレーム数」。SkipFrame(4) の内側で数えるので
    実際の意思決定回数は max_frames / 4 になる。
    以前は 1000 が既定で、人間が 4000 フレーム必要なステージでは
    全エピソードが 1/4 地点で打ち切られたデータしか集まらなかった。
    """
    env = make_mario_env(
        level=level,
        max_episode_steps=max_frames,
        render_mode=None,
        random_level=False,
    )

    obs, info = env.reset()
    image_paths = []
    actions = []
    rewards = []

    t = 0
    done = False
    mario_x = 0
    # 障害物前でargmax/決定論的挙動が完全に固着するのを防ぐ安全弁。
    # DTの学習データが「毎回同じ場所で永遠に停滞するタイムアウト」ばかりに
    # 汚染されるのを避け、ちゃんと先まで進んだ高収益トラジェクトリを増やす。
    guard = StallGuard()
    while not done:
        # obs はスタック済み (12,84,84)。DT 用には最新1フレーム (3,84,84) だけ保存する
        # os.path.join は Windows で "\\" を使い、Linux に持っていくと読めなくなるので "/" で固定
        img_path = f"{frame_dir.rstrip('/')}/ep{ep_idx:05d}_t{t:05d}.png"
        frame = latest_frame(obs, channels=3)
        Image.fromarray(np.transpose(frame, (1, 2, 0))).save(img_path)

        action, _ = model.predict(obs, deterministic=deterministic)
        # epsilon > 0 のとき、確率 epsilon で一様ランダム行動に置き換える。
        # 同じ方策のスタイルを保ったまま収益だけを連続的に下げ、
        # DT の returns-to-go 条件付けに必要な「低〜中収益の手本」を作るのが目的。
        # 別系統のモデルを混ぜて収益差を作ると模倣対象が二峰化して崩れる（v5/v7 の教訓）。
        if epsilon > 0 and random.random() < epsilon:
            action = random.randrange(env.action_space.n)
        action = guard.choose(int(action), mario_x)
        obs, reward, terminated, truncated, info = env.step(action)
        mario_x = info.get('mario_x', mario_x)

        image_paths.append(img_path)
        actions.append(action)
        rewards.append(float(reward))

        done = terminated or truncated
        t += 1

    env.close()

    rewards = np.array(rewards, dtype=np.float32)
    returns_to_go = np.cumsum(rewards[::-1])[::-1].astype(np.float32)

    return {
        "image_paths": image_paths,
        "actions": np.array(actions, dtype=np.int64),
        "rewards": rewards,
        "returns_to_go": returns_to_go,
        "level": level,
        "final_reason": info.get("reason", "timeout"),
        "progress": float(info.get("progress", 0.0)),
        "epsilon": float(epsilon),
    }


def main():
    parser = argparse.ArgumentParser(description="PPOエージェントのプレイからDT学習用データセットを収集する")
    parser.add_argument("--models", type=str, required=True,
                         help="カンマ区切りのPPOモデルパス（複数指定でスキルレベルを混在させる）")
    parser.add_argument("--levels", type=str, default=None,
                         help="カンマ区切りのレベル名。省略時は levels/ 以下を自動検出（Level1-2除く）")
    parser.add_argument("--episodes-per-model", type=int, default=40,
                         help="モデル1つあたりの収集エピソード数（レベルに均等配分）")
    parser.add_argument("--episodes-per-level", type=int, default=None,
                         help="指定時はレベル単位で収集し、レベルごとにこの本数を全モデルに均等配分する"
                              "（各レベルで十分な高収益サンプルを確保したい場合に使う）")
    parser.add_argument("--max-frames", type=int, default=8000,
                         help="1エピソードの最大ゲームフレーム数 (SkipFrame の内側)")
    parser.add_argument("--stochastic-ratio", type=float, default=0.7,
                         help="確率的行動(deterministic=False)で集めるエピソードの割合。0〜1")
    parser.add_argument("--output-dir", type=str, default="./dt_dataset/frames")
    parser.add_argument("--output-meta", type=str, default="./dt_dataset/metadata.pkl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epsilons", type=str, default="0",
                         help="カンマ区切りのランダム行動確率。エピソードごとに順番に割り当てる "
                              "(例: 0.2,0.4,0.6)。0 なら劣化なし")
    parser.add_argument("--resume", action="store_true",
                         help="--output-meta が既にあれば読み込み、続きから収集する。"
                              "中断・再起動しても収集済み分を捨てずに済む")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                         help="何本ごとに途中保存するか")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_meta), exist_ok=True)

    model_paths = [p.strip() for p in args.models.split(",") if p.strip()]
    levels = [l.strip() for l in args.levels.split(",")] if args.levels else get_levels()
    print(f"📦 モデル: {model_paths}")
    print(f"🗺️  レベル: {levels}")

    print("🤖 全モデルを読み込み中...")
    models = {p: PPO.load(p, device=args.device) for p in model_paths}

    # 収集する (レベル, モデル) の並びを先に確定させる。
    # こうしておくと「既に N 本収集済みなら先頭 N 件を飛ばす」だけで再開でき、
    # 中断のたびに最初からやり直す必要がなくなる。
    epsilons = [float(x) for x in args.epsilons.split(",")]
    plan = []
    if args.episodes_per_level is not None:
        for level in levels:
            for i in range(args.episodes_per_level):
                plan.append((model_paths[i % len(model_paths)], level, epsilons[i % len(epsilons)]))
    else:
        for model_path in model_paths:
            for i in range(args.episodes_per_model):
                plan.append((model_path, levels[i % len(levels)], epsilons[i % len(epsilons)]))

    episodes = []
    if args.resume and os.path.exists(args.output_meta):
        with open(args.output_meta, "rb") as f:
            episodes = pickle.load(f)
        print(f"🔄 既存の {args.output_meta} から {len(episodes)} エピソードを読み込み、続きから収集します")
        if len(episodes) >= len(plan):
            print("✅ 計画済みのエピソードは既にすべて収集済みです")
            return

    ep_idx = len(episodes)
    # 長時間収集の途中でプロセスが落ちても(セッション切断・再起動・スリープなど)、
    # 直近 checkpoint_every 本分しか失わないよう、定期的に部分保存する。
    checkpoint_every = args.checkpoint_every

    def save_checkpoint():
        # 書き込み中に落ちても既存の保存分を壊さないよう、一時ファイル経由で置き換える
        tmp = args.output_meta + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(episodes, f)
        os.replace(tmp, args.output_meta)

    def run_one(model_path, level, ep_idx, epsilon=0.0):
        model = models[model_path]
        deterministic = random.random() >= args.stochastic_ratio
        ep_data = collect_episode(model, level, args.max_frames, deterministic, args.output_dir, ep_idx,
                                  epsilon=epsilon)
        total_reward = ep_data["rewards"].sum()
        print(f"  ep{ep_idx:05d} model={os.path.basename(model_path)} level={level} "
              f"steps={len(ep_data['actions'])} return={total_reward:.1f} "
              f"progress={ep_data['progress'] * 100:.0f}% "
              f"det={deterministic} eps={epsilon:.2f} reason={ep_data['final_reason']}")
        episodes.append(ep_data)
        if len(episodes) % checkpoint_every == 0:
            save_checkpoint()
            print(f"  💾 途中経過を保存 ({len(episodes)} エピソード -> {args.output_meta})")

    print(f"\n📋 収集計画: 全 {len(plan)} 本 / 残り {len(plan) - ep_idx} 本")
    prev_level = None
    for model_path, level, epsilon in plan[ep_idx:]:
        if level != prev_level:
            print(f"\n🗺️  レベル: {level}")
            prev_level = level
        run_one(model_path, level, ep_idx, epsilon)
        ep_idx += 1

    save_checkpoint()

    returns = [ep["rewards"].sum() for ep in episodes]
    print("\n" + "=" * 60)
    print(f"✅ 収集完了: {len(episodes)} エピソード -> {args.output_meta}")
    print(f"   総フレーム数: {sum(len(ep['actions']) for ep in episodes)}")
    print(f"   return 範囲: {min(returns):.1f} 〜 {max(returns):.1f} (平均 {np.mean(returns):.1f})")


if __name__ == "__main__":
    main()
