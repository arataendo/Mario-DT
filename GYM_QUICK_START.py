"""
Mario Game を Gymnasium 環境として使用するクイックスタートガイド
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# ======================== 使用例 ========================

# 1. 基本的な環境初期化（pygame 環境が必要）
# from classes.MarioGymEnv import MarioEnv
# env = MarioEnv(level='Level1-1', render_mode=None)

# 2. リセットとステップ実行
# obs, info = env.reset()
# action = env.action_space.sample()
# obs, reward, terminated, truncated, info = env.step(action)

# ======================== アクション定義 ========================

ACTION_MAPPING = {
    0: "NOP (何もしない)",
    1: "Left (左移動)",
    2: "Right (右移動)",
    3: "Jump (ジャンプ)",
    4: "Left + Jump",
    5: "Right + Jump",
    6: "Dash (ダッシュ)",
    7: "Right + Dash",
}

# ======================== 観測空間 ========================

OBSERVATION_KEYS = {
    'image': {
        'shape': (3, 480, 640),
        'dtype': 'uint8',
        'description': 'RGB 画像（CHW format）'
    },
    'state': {
        'shape': (9,),
        'dtype': 'float32',
        'features': [
            'mario_x',           # マリオの X 座標 (正規化)
            'mario_y',           # マリオの Y 座標
            'velocity_x',        # 速度 X
            'velocity_y',        # 速度 Y
            'camera_x',          # カメラ位置 X
            'progress',          # レベル進行度 (0-1)
            'coins_collected',   # 集めたコイン数
            'entities_count',    # レベル内のエンティティ数
            'powerup_state',     # パワーアップ状態 (0=小, 1=大)
        ]
    }
}

# ======================== 報酬関数 ========================

REWARD_COMPONENTS = {
    'progress': '進行度 * 0.1（右への移動距離に基づく）',
    'time_penalty': '-0.01 per step（時間ペナルティ）',
    'coin_bonus': '+1.0（コイン収集時）',
    'enemy_points': '+point_delta/100（敵撃破ポイント）',
    'death_penalty': '-10.0（ゲームオーバー時）',
    'level_complete': '+100.0（ステージクリア時）',
}

# ======================== 使用例コード ========================

example_code_headless = '''
# headless モード（画面表示なし、高速訓練用）
from classes.MarioGymEnv import MarioEnv

env = MarioEnv(
    level='Level1-1',
    render_mode=None,  # headless
    max_episode_steps=10800
)

obs, info = env.reset()
total_reward = 0

for step in range(1000):
    # ランダムアクション（実際はRL エージェント）
    action = env.action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if terminated or truncated:
        print(f"Episode finished after {step} steps")
        print(f"Total reward: {total_reward}")
        obs, info = env.reset()
        total_reward = 0

env.close()
'''

example_code_render = '''
# render モード（画面表示）
from classes.MarioGymEnv import MarioEnv

env = MarioEnv(
    level='Level1-1',
    render_mode='human',  # 画面表示
)

obs, info = env.reset()

for step in range(10000):
    action = env.action_space.sample()  # ランダムエージェント
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
'''

example_code_rl = '''
# Stable Baselines3 を使用した強化学習
from classes.MarioGymEnv import MarioEnv
from stable_baselines3 import PPO

env = MarioEnv(
    random_level=True,
    render_mode=None,
    max_episode_steps=10800
)

# PPO エージェントを作成
model = PPO("MultiInputPolicy", env, verbose=1)

# 訓練
model.learn(total_timesteps=100000)

# 訓練済みモデルで実行
obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
'''

if __name__ == "__main__":
    print("Mario Game Gymnasium Environment - Quick Start Guide")
    print("=" * 60)
    print()
    
    print("アクション定義:")
    for action_id, action_name in ACTION_MAPPING.items():
        print(f"  {action_id}: {action_name}")
    
    print("\n観測空間:")
    print(f"  - image: {OBSERVATION_KEYS['image']['shape']} RGB 画像")
    print(f"  - state: {OBSERVATION_KEYS['state']['shape']} 状態ベクトル")
    print("\n    状態ベクトルの要素:")
    for i, feature in enumerate(OBSERVATION_KEYS['state']['features']):
        print(f"      {i}: {feature}")
    
    print("\n報酬関数のコンポーネント:")
    for component, description in REWARD_COMPONENTS.items():
        print(f"  - {component}: {description}")
    
    print("\n使用例 (1) - Headless 訓練:")
    print(example_code_headless)
    
    print("\n使用例 (2) - 画面表示:")
    print(example_code_render)
    
    print("\n使用例 (3) - 強化学習 (Stable Baselines3):")
    print(example_code_rl)
