"""
PPO を使用してカスタム Mario ゲーム環境で 100,000 ステップ学習を行う

使用方法:
    python train_ppo.py --level Level1-1 --total-steps 100000 --log-dir ./logs

出力:
    - models/mario_ppo_level1-1_100k.zip : 学習済みモデル
    - logs/mario_ppo_level1-1_100k/ : TensorBoard ログ
"""

import os
import argparse
import gymnasium as gym
import numpy as np
from datetime import datetime
from pathlib import Path
import cv2

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback
from stable_baselines3.common.evaluation import evaluate_policy

# カスタム環境をインポート
from classes.MarioGymEnv import MarioEnv


class DictToImageWrapper(gym.ObservationWrapper):
    """Dict 観測から画像のみを抽出し、リサイズするラッパー"""
    def __init__(self, env):
        super().__init__(env)
        
        # リサイズ後のサイズ（標準的な 84x84 にする）
        self.new_size = (84, 84)
        
        # 観測空間を uint8（0-255の整数）に変更
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, self.new_size[1], self.new_size[0]),
            dtype=np.uint8
        )
    
    def observation(self, obs):
        """Dict 観測から画像を抽出し、リサイズ"""
        if isinstance(obs, dict):
            image = obs['image']
        else:
            image = obs
            
        # 画像の形を (C, H, W) から (H, W, C) に変換してリサイズ
        image = np.transpose(image, (1, 2, 0))
        image = cv2.resize(image, self.new_size, interpolation=cv2.INTER_AREA)
        # 再び (C, H, W) に戻す
        image = np.transpose(image, (2, 0, 1))
        
        # float32での正規化（/255.0）はここでは行わず、uint8のまま返す
        return image


def make_env(level: str, rank: int, seed: int = 0):
    """並列化用の環境生成関数"""
    def _init():
        env = MarioEnv(
            level=level,
            render_mode=None,  # headless mode
            max_episode_steps=1000,
            random_level=False
        )
        # Dict 観測から画像を抽出
        env = DictToImageWrapper(env)
        # 各プロセスにシードをずらして設定
        env.reset(seed=seed + rank)
        return env
    return _init


def train_ppo(
    level: str = "Level1-1",
    total_steps: int = 100000,
    num_envs: int = 4,
    log_dir: str = "./logs",
    model_dir: str = "./models",
    device: str = "auto"
):
    """
    PPO で Mario ゲームを学習
    
    Parameters
    ----------
    level : str
        学習するレベル (例: "Level1-1", "Level1-2")
    total_steps : int
        総学習ステップ数 (デフォルト: 100,000)
    num_envs : int
        並列環境数 (デフォルト: 4)
    log_dir : str
        TensorBoard ログディレクトリ
    model_dir : str
        学習済みモデル保存ディレクトリ
    device : str
        GPU使用の有無 ("cuda", "cpu", "auto")
    """
    
    # ディレクトリ作成
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # ログファイル名生成
    level_name = level.replace("-", "").lower()
    steps_k = total_steps // 1000
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_name = f"mario_ppo_{level_name}_{steps_k}k"
    log_path = os.path.join(log_dir, f"{model_name}_{timestamp}")
    model_path = os.path.join(model_dir, model_name)
    
    print("=" * 60)
    print("🎮 Mario PPO 学習を開始します")
    print("=" * 60)
    print(f"📊 設定:")
    print(f"   - レベル: {level}")
    print(f"   - 総ステップ数: {total_steps:,}")
    print(f"   - 並列環境数: {num_envs}")
    print(f"   - ログディレクトリ: {log_path}")
    print(f"   - モデル保存先: {model_path}.zip")
    print()
    
    # 並列化された環境を作成
    print("🔧 環境を初期化中...")
    envs = SubprocVecEnv(
        [make_env(level, i) for i in range(num_envs)],
        start_method="spawn"  # Windows では spawn を使用
    )
    
    # 評価用環境（シングル）
    eval_env = MarioEnv(
        level=level,
        render_mode=None,
        max_episode_steps=1000
    )
    eval_env = DictToImageWrapper(eval_env)
    
    # PPO エージェントを作成
    print("🤖 PPO エージェントを初期化中...")
    
    # PPO ハイパーパラメータ
    model = PPO(
        policy="CnnPolicy",  # CNN ポリシー（画像入力用）
        env=envs,
        learning_rate=3e-4,
        n_steps=2048,  # バッチサイズ = n_steps * num_envs
        batch_size=64,
        n_epochs=10,
        gamma=0.99,  # 割引率
        gae_lambda=0.95,  # GAE パラメータ
        clip_range=0.2,  # PPO クリップ範囲
        ent_coef=0.01,  # エントロピー係数
        vf_coef=0.5,  # 価値関数損失係数
        max_grad_norm=0.5,
        verbose=1,
        device=device,
        tensorboard_log=log_path,
    #    policy_kwargs=dict(normalize_images=False)  # unit8なら正規化不要
    )
    
    print("✅ 環境とモデルの初期化完了")
    print()
    print("🚀 学習を開始します...")
    print("-" * 60)
    
    try:
        # チェックポイントコールバック（10万ステップごとにモデルを保存）
        checkpoint_callback = CheckpointCallback(
            save_freq=max(10000, total_steps // 10),
            save_path=model_dir,
            name_prefix=f"mario_ppo_{level_name}_checkpoint",
            save_replay_buffer=False,
        )
        
        # プログレスバーコールバック
        progress_callback = ProgressBarCallback()
        
        # モデルを学習
        model.learn(
            total_timesteps=total_steps,
            callback=[checkpoint_callback, progress_callback],
            progress_bar=True
        )
        
        print("-" * 60)
        print("✅ 学習完了！")
        print()
        
        # モデルを保存
        print(f"💾 モデルを保存中: {model_path}.zip")
        model.save(model_path)
        print("✅ モデル保存完了")
        print()
        
        # 学習済みモデルで評価
        print("📊 学習済みモデルで評価中...")
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=5,
            deterministic=True
        )
        print(f"   平均報酬: {mean_reward:.2f} (+/- {std_reward:.2f})")
        print()
        
        # サマリー出力
        print("=" * 60)
        print("🎉 学習完了サマリー")
        print("=" * 60)
        print(f"✅ モデル保存先: {model_path}.zip")
        print(f"✅ ログディレクトリ: {log_path}")
        print(f"✅ 総学習ステップ: {total_steps:,}")
        print(f"✅ 平均報酬: {mean_reward:.2f}")
        print()
        print("次のステップ:")
        print(f"  1. TensorBoard で学習過程を確認:")
        print(f"     tensorboard --logdir {log_path}")
        print(f"  2. 推論スクリプトで動作確認:")
        print(f"     python infer_ppo.py --model {model_path}.zip")
        print()
        
    except KeyboardInterrupt:
        print("\n⚠️  学習を中断しました")
        print(f"💾 中途モデルを保存中: {model_path}_interrupted.zip")
        model.save(f"{model_path}_interrupted")
        print("✅ 保存完了")
    
    finally:
        # 環境をクローズ
        envs.close()
        eval_env.close()


def main():
    parser = argparse.ArgumentParser(
        description="PPO を使用してカスタム Mario ゲームで学習を行う",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # デフォルト設定で 100,000 ステップ学習
  python train_ppo.py

  # Level1-2 で 200,000 ステップ学習
  python train_ppo.py --level Level1-2 --total-steps 200000

  # GPU を使用して 8 並列で学習
  python train_ppo.py --device cuda --num-envs 8
        """
    )
    
    parser.add_argument(
        "--level",
        type=str,
        default="Level1-1",
        help="学習するレベル (デフォルト: Level1-1)"
    )
    
    parser.add_argument(
        "--total-steps",
        type=int,
        default=100000,
        help="総学習ステップ数 (デフォルト: 100000)"
    )
    
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="並列環境数 (デフォルト: 4)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="デバイス (デフォルト: auto)"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="TensorBoard ログディレクトリ (デフォルト: ./logs)"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models",
        help="モデル保存ディレクトリ (デフォルト: ./models)"
    )
    
    args = parser.parse_args()
    
    # 学習を実行
    train_ppo(
        level=args.level,
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        device=args.device,
        log_dir=args.log_dir,
        model_dir=args.model_dir
    )


if __name__ == "__main__":
    main()
