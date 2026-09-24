"""
学習済み PPO モデルで Mario ゲームの推論を行う

使用方法:
    python infer_ppo.py --model models/mario_ppo_level1-1_100k.zip --level Level1-1 --render
"""

import os
import argparse
import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from stable_baselines3 import PPO

# ★ 学習時とまったく同じラッパーを使う（定義を重複させないこと）
from classes.wrappers import make_mario_env, StallGuard


def run_inference(
    model_path: str,
    level: str = "Level1-1",
    num_episodes: int = 5,
    max_frames: int = 8000,
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
    max_frames : int
        エピソードあたりの最大ゲームフレーム数 (SkipFrame の内側で数える)
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
    print(f"   - 最大フレーム: {max_frames} (= 意思決定 {max_frames // 4} 回)")
    print(f"   - 表示: {'はい' if render else 'いいえ'}")
    print(f"   - 決定論的: {'はい' if deterministic else 'いいえ'}")
    print()
    
    # 環境を作成
    print("🔧 環境を初期化中...")
    env = make_mario_env(
        level=level,
        max_episode_steps=max_frames,
        render_mode="human" if render else None,
        random_level=False,
    )
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
        mario_x = 0
        guard = StallGuard()

        print(f"エピソード {ep + 1}/{num_episodes}: ", end="", flush=True)

        while not done:
            # obs は既に DictToImageWrapper で画像に変換されている
            # モデルで行動を予測
            action, _ = model.predict(
                obs,
                deterministic=deterministic
            )
            # 障害物前での argmax 固着（同一観測の自己ループ）を防ぐ安全弁
            action = guard.choose(int(action), mario_x)

            # 環境で行動を実行
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            mario_x = info.get('mario_x', mario_x)
            done = terminated or truncated

            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        mario_x = info.get('mario_x', 0)
        coins = info.get('coins', 0)
        reason = info.get('reason', 'unknown')
        if terminated and reason == 'level_complete':
            status = '✅ クリア'
        elif terminated:
            status = f'💀 {reason}'
        else:
            status = '🔄 タイムアップ'
        
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
        "--max-frames",
        type=int,
        default=8000,
        help="エピソードあたりの最大ゲームフレーム数 (デフォルト: 8000)"
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
        max_frames=args.max_frames,
        render=args.render,
        deterministic=args.deterministic
    )


if __name__ == "__main__":
    main()
