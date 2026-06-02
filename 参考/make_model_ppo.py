import argparse
import os
from typing import Optional

import cv2
import numpy as np
import gymnasium as gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecFrameStack


os.makedirs("datasets", exist_ok=True)
os.makedirs("models", exist_ok=True)


class GymnasiumBridge(gym.Wrapper):
    """
    nes_py の古い Gym API を Gymnasium API に変換するラッパー。
    ・step() の戻り値を4つ(旧)から5つ(新)に変換
    ・reset() 時の seed 引数エラーを吸収
    """
    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            return obs, reward, done, False, info # truncated = False を追加
        return step_result

    def reset(self, seed=None, options=None):
        if seed is not None and hasattr(self.env, 'seed'):
            self.env.seed(seed)
        
        reset_result = self.env.reset()
        if not isinstance(reset_result, tuple):
            return reset_result, {}
        return reset_result


class SkipFrame(gym.Wrapper):
    """指定したフレーム数だけ同じ行動を繰り返す"""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        info = {}
        for _ in range(self._skip):
            obs, reward, done_flag, truncated_flag, info = self.env.step(action)
            total_reward += reward
            done = done or done_flag
            truncated = truncated or truncated_flag
            if done or truncated:
                break
        return obs, total_reward, done, truncated, info


class OpenCVWarpFrame(gym.ObservationWrapper):
    """cv2を用いた高速なグレースケール化とリサイズ"""
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        # PyTorch向けに (H, W, C) で定義 (SB3が自動的にチャネルファーストに変換します)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )

    def observation(self, frame):
        # RGB -> グレースケール
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # 84x84 にリサイズ
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # 次元を追加 (84, 84) -> (84, 84, 1)
        return np.expand_dims(resized, axis=-1)


def make_env(env_id: str, rank: int, seed: int = 0):
    def _init():
        # 環境の生成
        env = gym_super_mario_bros.make(env_id)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        
        # 古いAPIと新しいAPIの橋渡し
        env = GymnasiumBridge(env)
        
        # 独自ラッパーによる画像の前処理
        env = SkipFrame(env, skip=4)
        env = OpenCVWarpFrame(env, width=84, height=84)
        
        # シードの設定（プロセスごとにずらす）
        env.reset(seed=seed + rank)
        return env
    return _init


def create_or_load_model(
    env_id: str = 'SuperMarioBros-1-1-v3', 
    model_path: str = "models/ppo_mario", 
    train_steps: Optional[int] = None, 
    force_train: bool = False,
    num_envs: int = 8
) -> PPO:
    
    env_fns = [make_env(env_id, i) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)
    
    # 複数フレームのスタックは Stable Baselines3 のネイティブ機能を使うのが最も確実
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = VecMonitor(vec_env) 

    if os.path.exists(model_path + ".zip") and (not force_train) and (train_steps is None):
        print("🤖 学習済みモデルを読み込みます...")
        model = PPO.load(model_path, env=vec_env)
        return model

    print("🤖 新しいモデルを作成します...")
    model = PPO(
        'CnnPolicy', 
        vec_env, 
        verbose=1,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )

    if train_steps is not None and train_steps > 0:
        print(f"🧠 モデルを {train_steps} ステップ学習します（並列処理中）...")
        model.learn(total_timesteps=int(train_steps))
        model.save(model_path)
        print(f"💾 モデルを保存しました: {model_path}.zip")
    else:
        model.save(model_path)
        print(f"💾 未学習モデルを保存しました: {model_path}.zip")

    vec_env.close()
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Create or load PPO model for Mario")
    p.add_argument('--train-steps', type=int, default=None, help='学習させるステップ数（指定しないと学習しません）')
    p.add_argument('--force-train', action='store_true', help='既存モデルがあっても強制的に新規学習します')
    p.add_argument('--model-path', type=str, default='models/ppo_mario', help='モデル保存パス（拡張子なし）')
    p.add_argument('--num-envs', type=int, default=8, help='並列実行する環境の数（CPUコア数目安）')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    import multiprocessing
    # Windows/Linux 間のマルチプロセス互換性のための記述
    if multiprocessing.get_start_method() == 'fork':
        pass
        
    create_or_load_model(
        model_path=args.model_path, 
        train_steps=args.train_steps, 
        force_train=args.force_train,
        num_envs=args.num_envs
    )