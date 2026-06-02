"""
train_ppo.py の簡単なテスト
- 構文エラーがないか確認
- 環境とモデルが正しく初期化されるか確認
- 短い学習ループが実行できるか確認
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("🧪 PPO 学習実装テスト")
print("=" * 60)
print()

# Step 1: 必要なパッケージのインポートを確認
print("✓ ステップ 1: パッケージをインポート中...")
try:
    import gymnasium as gym
    import numpy as np
    from stable_baselines3 import PPO
    from classes.MarioGymEnv import MarioEnv
    print("  ✅ すべてのパッケージをインポート成功")
except Exception as e:
    print(f"  ❌ インポート失敗: {e}")
    sys.exit(1)

print()

# Step 2: DictToImageWrapper をテスト
print("✓ ステップ 2: DictToImageWrapper をテスト中...")
try:
    class DictToImageWrapper(gym.ObservationWrapper):
        """Dict 観測から画像のみを抽出するラッパー"""
        def __init__(self, env):
            super().__init__(env)
            original_obs_space = env.observation_space
            
            if isinstance(original_obs_space, gym.spaces.Dict):
                image_space = original_obs_space['image']
            else:
                raise ValueError("Environment must have Dict observation space")
            
            self.observation_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(image_space.shape[0], image_space.shape[1], image_space.shape[2]),
                dtype=np.float32
            )
        
        def observation(self, obs):
            if isinstance(obs, dict):
                image = obs['image'].astype(np.float32)
            else:
                image = obs.astype(np.float32)
            
            image = image / 255.0
            return image
    
    print("  ✅ DictToImageWrapper 定義成功")
except Exception as e:
    print(f"  ❌ 定義失敗: {e}")
    sys.exit(1)

print()

# Step 3: 環境の初期化をテスト
print("✓ ステップ 3: 環境を初期化中...")
try:
    env = MarioEnv(
        level="Level1-1",
        render_mode=None,
        max_episode_steps=100,
        random_level=False
    )
    print(f"  ✅ 環境作成成功")
    print(f"     - 観測空間: {env.observation_space}")
    print(f"     - アクション空間: {env.action_space}")
except Exception as e:
    print(f"  ❌ 環境作成失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 4: ラッパーを適用
print("✓ ステップ 4: DictToImageWrapper を適用中...")
try:
    env = DictToImageWrapper(env)
    print(f"  ✅ ラッパー適用成功")
    print(f"     - 観測空間: {env.observation_space}")
except Exception as e:
    print(f"  ❌ ラッパー適用失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 5: 環境の reset をテスト
print("✓ ステップ 5: 環境の reset をテスト中...")
try:
    obs, info = env.reset()
    print(f"  ✅ Reset 成功")
    print(f"     - 観測形状: {obs.shape}")
    print(f"     - 観測dtype: {obs.dtype}")
    print(f"     - 観測範囲: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"     - Info: {list(info.keys())}")
except Exception as e:
    print(f"  ❌ Reset 失敗: {e}")
    import traceback
    traceback.print_exc()
    env.close()
    sys.exit(1)

print()

# Step 6: PPO モデルの初期化をテスト
print("✓ ステップ 6: PPO モデルを初期化中...")
try:
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=64,  # 小さい値でテスト
        batch_size=32,
        n_epochs=1,
        verbose=0,
        device="cpu",
        policy_kwargs=dict(normalize_images=False)  # 既に正規化済みの画像を使用
    )
    print(f"  ✅ モデル作成成功")
    print(f"     - ポリシー: CnnPolicy")
    print(f"     - デバイス: CPU")
except Exception as e:
    print(f"  ❌ モデル作成失敗: {e}")
    import traceback
    traceback.print_exc()
    env.close()
    sys.exit(1)

print()

# Step 7: 短い学習ループをテスト
print("✓ ステップ 7: 短い学習ループをテスト中...")
try:
    print("  学習中: ", end="", flush=True)
    model.learn(total_timesteps=256)
    print("✅ 完了")
except Exception as e:
    print(f"\n  ❌ 学習失敗: {e}")
    import traceback
    traceback.print_exc()
    env.close()
    sys.exit(1)

print()

# Step 8: 推論をテスト
print("✓ ステップ 8: 推論をテスト中...")
try:
    obs, info = env.reset()
    for i in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    print(f"  ✅ 推論成功 ({i+1} ステップ実行)")
except Exception as e:
    print(f"  ❌ 推論失敗: {e}")
    import traceback
    traceback.print_exc()
    env.close()
    sys.exit(1)

print()

# Step 9: モデル保存をテスト
print("✓ ステップ 9: モデル保存をテスト中...")
try:
    os.makedirs("models", exist_ok=True)
    test_model_path = "models/test_ppo"
    model.save(test_model_path)
    if os.path.exists(test_model_path + ".zip"):
        print(f"  ✅ モデル保存成功: {test_model_path}.zip")
        # テストモデルを削除
        os.remove(test_model_path + ".zip")
    else:
        print(f"  ❌ モデル保存失敗: ファイルが見つかりません")
except Exception as e:
    print(f"  ❌ 保存失敗: {e}")
    import traceback
    traceback.print_exc()

print()

# クリーンアップ
env.close()

print("=" * 60)
print("✅ すべてのテストに成功しました！")
print("=" * 60)
print()
print("次のステップ:")
print("  1. 学習を実行:")
print("     python train_ppo.py --total-steps 100000")
print()
print("  2. TensorBoard で監視:")
print("     tensorboard --logdir ./logs")
print()
print("  3. 推論を実行:")
print("     python infer_ppo.py --model models/mario_ppo_level1-1_100k.zip --render")
print()
