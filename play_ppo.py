import os
import argparse
import numpy as np
import cv2
from stable_baselines3 import PPO
import gymnasium as gym

from classes.MarioGymEnv import MarioEnv


class SkipFrame(gym.Wrapper):
    """
    同じアクションを複数フレーム繰り返すラッパー
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class DictToImageWrapper(gym.ObservationWrapper):
    """Dict 観測から画像を抽出し、84x84 にリサイズするラッパー"""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, 84, 84),
            dtype=np.uint8,
        )

    def observation(self, obs):
        if isinstance(obs, dict):
            img = obs['image']
        else:
            img = obs

        img_hwc = np.transpose(img, (1, 2, 0))
        img_resized = cv2.resize(img_hwc, (84, 84), interpolation=cv2.INTER_AREA)
        img_chw = np.transpose(img_resized, (2, 0, 1))
        return img_chw.astype(np.uint8)


def play_trained_agent(
    model_path: str,
    level: str = " Level_custom",
    num_episodes: int = 1,
    max_steps: int = 3000,
    render: bool = True,
    deterministic: bool = True,
    device: str = "auto",
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"モデルが見つかりません: {model_path}")

    print("=" * 60)
    print("🎮 学習済み PPO エージェントの再生を開始します")
    print("=" * 60)
    print(f"📦 モデル: {model_path}")
    print(f"🗺️  レベル: {level}")
    print(f"🔁 エピソード数: {num_episodes}")
    print(f"🖥️  表示: {'あり' if render else 'なし'}")
    print()

    env = MarioEnv(
        level=level,
        render_mode="human" if render else None,
        max_episode_steps=max_steps,
    )
    env = SkipFrame(env, skip=4)
    env = DictToImageWrapper(env)

    print("🤖 モデルを読み込み中...")
    model = PPO.load(
        model_path,
        env=env,
        device=device,
        custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
        },
    )
    print("✅ モデル読み込み完了")
    print()

    episode_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        print(f"エピソード {ep + 1}/{num_episodes} を開始します...")
        while not done and episode_length < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        mario_x = info.get("mario_x", 0)
        status = "✅ クリア" if terminated and mario_x > 100 else "🔄 タイムアップ"
        print(
            f"  報酬={episode_reward:7.2f}, "
            f"ステップ={episode_length}, "
            f"X={mario_x:3d}, "
            f"{status}"
        )

    env.close()

    if episode_rewards:
        print("-" * 60)
        print("📊 再生結果")
        print(f"  平均報酬: {np.mean(episode_rewards):.2f}")
        print(f"  平均ステップ: {np.mean(episode_lengths):.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="学習済み PPO モデルを再生する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="再生する学習済みモデルのパス"
    )
    parser.add_argument(
        "--level",
        type=str,
        default="Level_custom",
        help="プレイするレベル"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="再生するエピソード数"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3000,
        help="1エピソードの最大ステップ数"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="画面表示を有効にする"
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_false",
        dest="deterministic",
        help="確率的な行動を使用する"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="モデル読み込み用デバイス"
    )

    args = parser.parse_args()

    play_trained_agent(
        model_path=args.model,
        level=args.level,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        deterministic=args.deterministic,
        device=args.device,
    )


if __name__ == "__main__":
    main()
