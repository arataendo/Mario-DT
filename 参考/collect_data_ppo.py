import os
import time
import pickle
import cv2
import numpy as np
import gymnasium as gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# --- 学習時と同じラッパー群を定義 ---
class GymnasiumBridge(gym.Wrapper):
    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            return obs, reward, done, False, info
        return step_result

    def reset(self, seed=None, options=None):
        if seed is not None and hasattr(self.env, 'seed'):
            self.env.seed(seed)
        reset_result = self.env.reset()
        if not isinstance(reset_result, tuple):
            return reset_result, {}
        return reset_result

class SkipFrame(gym.Wrapper):
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
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )

    def observation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=-1)

# --- 環境構築関数 ---
def make_env():
    def _init():
        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v3')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = GymnasiumBridge(env)
        env = SkipFrame(env, skip=4)
        env = OpenCVWarpFrame(env, width=84, height=84)
        return env
    return _init

os.makedirs("datasets", exist_ok=True)
os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/ppo_mario"

# データ収集用なので並列化はせず、単一環境用の DummyVecEnv を使用
vec_env = DummyVecEnv([make_env()])
# 学習時と同じく4フレームスタックを適用
vec_env = VecFrameStack(vec_env, n_stack=4)

if os.path.exists(MODEL_PATH + ".zip"):
    print("🤖 学習済みモデルを読み込みます...")
    model = PPO.load(MODEL_PATH, env=vec_env)
else:
    print("🤖 モデルが見つかりません。先に make_model_ppo.py で学習を行ってください。")
    exit()

print("🎮 エージェントによるプレイとデータ収集を開始します...")
episodes_data = []
NUM_EPISODES = 100

for ep in range(NUM_EPISODES):
    current_episode = {"observations": [], "actions": [], "rewards": [], "dones": []}
    
    # VecEnv の reset は info を返さず、直接 obs (shape: (1, 4, 84, 84)) を返す
    obs = vec_env.reset()
    done = False
    total_reward = 0

    while not done:
        # PPOモデルで行動予測
        action, _states = model.predict(obs, deterministic=False)

        # VecEnv の step はリスト（バッチ）で結果を返す
        next_obs, rewards, dones, infos = vec_env.step(action)
        
        # バッチサイズ1なので、[0] で中身のスカラ値を取り出す
        current_done = dones[0]
        current_reward = rewards[0]
        current_action = action[0]
        
        # 保存用にはバッチ次元 (1, 4, 84, 84) を外した (4, 84, 84) を保存
        current_obs = obs[0] 

        current_episode["observations"].append(current_obs)
        current_episode["actions"].append(current_action)
        current_episode["rewards"].append(current_reward)
        current_episode["dones"].append(current_done)

        obs = next_obs
        total_reward += current_reward
        done = current_done

    episodes_data.append(current_episode)
    print(f"エピソード {ep+1}/{NUM_EPISODES} 終了. 獲得報酬: {total_reward} (ステップ: {len(current_episode['actions'])})")

filename = f"datasets/mario_ppo_log_{int(time.time())}.pkl"
with open(filename, 'wb') as f:
    pickle.dump(episodes_data, f)

print(f"\n全エピソード終了. {NUM_EPISODES} 件のプレイデータを保存しました: {filename}")
vec_env.close()