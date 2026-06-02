"""
学習済み PPO モデルで Mario ゲームの推論を行う

使用方法:
    python infer_ppo.py --model models/mario_ppo_level1-1_100k.zip --level Level1-1 --render
"""

import os
import argparse
import numpy as np
import gymnasium as gym
from pathlib import Path

from stable_baselines3 import PPO

from classes.MarioGymEnv import MarioEnv


class DictToImageWrapper(gym.ObservationWrapper):
    """Dict 観測から画像のみを抽出するラッパー"""
    def __init__(self, env):
        super().__init__(env)
        original_obs_space = env.observation_space
        
        # Dict から 'image' キーを取得
        if isinstance(original_obs_space, gym.spaces.Dict):
            image_space = original_obs_space['image']
        else:
            raise ValueError("Environment must have Dict observation space")
        
        # 観測空間を画像のみに変更（正規化済みなので 0-1 に）
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(image_space.shape[0], image_space.shape[1], image_space.shape[2]),
            dtype=np.float32
        )
    
    def observation(self, obs):
        """Dict 観測から画像を抽出し、正規化"""
        # obs は Dict{'image': ..., 'state': ...}
        if isinstance(obs, dict):
            image = obs['image'].astype(np.float32)
        else:
            image = obs.astype(np.float32)
        
        # 画像を 0-1 に正規化
        image = image / 255.0
        return image


def run_inference(
    model_path: str,
    level: str = "Level1-1",
    num_episodes: int = 5,
    max_steps: int = 1000,
    render: bool = False,
    deterministic: bool = True
):
    """
    学習済みモデルで推論を実行
    
    Parameters
    ----------
    model_path : str
        モデルファイルパス (.zip)
    level : str
        プレイするレベル
    num_episodes : int
        実行するエピソード数
    max_steps : int
        エピソードあたりの最大ステップ
    render : bool
        画面に表示するか
    deterministic : bool
        決定論的な行動を取るか (最大確率の行動)
    """
    
    # モデルの存在確認
    if not os.path.exists(model_path):
        print(f"❌ エラー: モデルが見つかりません: {model_path}")
        print("train_ppo.py で学習を実行してください")
        return
    
    print("=" * 60)
    print("🎮 Mario PPO 推論を開始します")
    print("=" * 60)
    print(f"📊 設定:")
    print(f"   - モデル: {model_path}")
    print(f"   - レベル: {level}")
    print(f"   - エピソード数: {num_episodes}")
    print(f"   - 最大ステップ: {max_steps}")
    print(f"   - 表示: {'はい' if render else 'いいえ'}")
    print(f"   - 決定論的: {'はい' if deterministic else 'いいえ'}")
    print()
    
    # 環境を作成
    print("🔧 環境を初期化中...")
    env = MarioEnv(
        level=level,
        render_mode="human" if render else None,
        max_episode_steps=max_steps
    )
    # Dict 観測から画像を抽出
    env = DictToImageWrapper(env)
    print("✅ 環境初期化完了")
    print()
    
    # モデルを読み込む
    print("🤖 モデルを読み込み中...")
    model = PPO.load(model_path, env=env)
    print("✅ モデル読み込み完了")
    print()
    
    # 推論ループ
    episode_rewards = []
    episode_lengths = []
    
    print("🚀 推論を開始します...")
    print("-" * 60)
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        print(f"エピソード {ep + 1}/{num_episodes}: ", end="", flush=True)
        
        while not done and episode_length < max_steps:
            # obs は既に DictToImageWrapper で画像に変換されている
            # モデルで行動を予測
            action, _ = model.predict(
                obs,
                deterministic=deterministic
            )
            
            # 環境で行動を実行
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        mario_x = info.get('mario_x', 0)
        coins = info.get('coins', 0)
        status = "✅ クリア" if terminated and mario_x > 100 else "🔄 タイムアップ"
        
        print(f"報酬={episode_reward:7.2f}, ステップ={episode_length}, X={mario_x:3d}, コイン={coins}, {status}")
    
    print("-" * 60)
    print()
    
    # 結果をサマリー
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print("=" * 60)
    print("📊 推論結果")
    print("=" * 60)
    print(f"✅ 平均報酬: {mean_reward:.2f} (+/- {std_reward:.2f})")
    print(f"✅ 平均ステップ: {mean_length:.1f}")
    print(f"✅ 最大報酬: {max(episode_rewards):.2f}")
    print(f"✅ 最小報酬: {min(episode_rewards):.2f}")
    print()
    
    # 環境をクローズ
    env.close()
    print("✅ 推論完了")


def main():
    parser = argparse.ArgumentParser(
        description="学習済み PPO モデルで推論を実行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 学習済みモデルを実行
  python infer_ppo.py --model models/mario_ppo_level1-1_100k.zip

  # 画面に表示して実行
  python infer_ppo.py --model models/mario_ppo_level1-1_100k.zip --render

  # 異なるレベルでテスト
  python infer_ppo.py --model models/mario_ppo_level1-1_100k.zip --level Level1-2

  # ランダムな行動を取る
  python infer_ppo.py --model models/mario_ppo_level1-1_100k.zip --no-deterministic

  # 10 エピソード実行
  python infer_ppo.py --model models/mario_ppo_level1-1_100k.zip --episodes 10
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="モデルファイルパス (必須)"
    )
    
    parser.add_argument(
        "--level",
        type=str,
        default="Level1-1",
        help="プレイするレベル (デフォルト: Level1-1)"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="実行するエピソード数 (デフォルト: 5)"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="エピソードあたりの最大ステップ (デフォルト: 1000)"
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="画面に表示する"
    )
    
    parser.add_argument(
        "--no-deterministic",
        action="store_false",
        dest="deterministic",
        help="確率的な行動を取る"
    )
    
    args = parser.parse_args()
    
    # 推論を実行
    run_inference(
        model_path=args.model,
        level=args.level,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        deterministic=args.deterministic
    )


if __name__ == "__main__":
    main()
