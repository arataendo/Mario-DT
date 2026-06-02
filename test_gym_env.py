"""
Mario Gym Environment のテストスクリプト

このスクリプトでは以下をテストします:
1. 環境の基本初期化
2. リセット機能
3. ステップ実行
4. ランダムエージェントのサンプル実行
5. 異なる render_mode の動作確認
"""

import numpy as np
import sys
import os

# パスの設定
sys.path.insert(0, os.path.dirname(__file__))

from classes.MarioGymEnv import MarioEnv


def test_basic_init():
    """基本的な初期化テスト"""
    print("=" * 60)
    print("Test 1: Basic Environment Initialization")
    print("=" * 60)
    
    try:
        env = MarioEnv(render_mode=None)
        print("✓ Environment created successfully (headless mode)")
        
        obs, info = env.reset()
        print("✓ Environment reset successfully")
        print(f"  - Observation type: Dict with keys {obs.keys()}")
        print(f"  - Image shape: {obs['image'].shape}")
        print(f"  - State shape: {obs['state'].shape}")
        print(f"  - Action space: {env.action_space}")
        print(f"  - Observation space: {env.observation_space}")
        
        env.close()
        print("✓ Environment closed successfully")
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step_loop():
    """ステップ実行ループのテスト"""
    print("\n" + "=" * 60)
    print("Test 2: Step Loop (100 steps with random actions)")
    print("=" * 60)
    
    try:
        env = MarioEnv(render_mode=None, level='Level1-1')
        obs, info = env.reset()
        
        total_reward = 0.0
        step_count = 0
        
        for step in range(100):
            action = env.action_space.sample()  # ランダムアクション
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            if step % 20 == 0:
                mario_x = info['mario_x']
                coins = info['coins']
                print(f"  Step {step:3d}: Action={action}, Reward={reward:7.3f}, "
                      f"Cumulative={total_reward:7.3f}, Mario_X={mario_x:4.0f}, Coins={coins}")
            
            if terminated or truncated:
                print(f"  Episode ended at step {step}: {info.get('reason', 'unknown')}")
                break
        
        print(f"✓ Step loop completed: {step_count} steps, Total reward: {total_reward:.3f}")
        env.close()
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_random_agent():
    """ランダムエージェントのテスト"""
    print("\n" + "=" * 60)
    print("Test 3: Random Agent Episode")
    print("=" * 60)
    
    try:
        env = MarioEnv(
            random_level=True,
            render_mode=None,
            max_episode_steps=1000
        )
        
        obs, info = env.reset()
        print(f"✓ Episode started: Level={info['level']}")
        
        episode_rewards = []
        done_reason = None
        
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_rewards.append(reward)
            
            if terminated or truncated:
                done_reason = info.get('reason', 'unknown')
                break
        
        print(f"✓ Episode finished: {done_reason}")
        print(f"  - Steps: {len(episode_rewards)}")
        print(f"  - Total reward: {sum(episode_rewards):.3f}")
        print(f"  - Avg reward/step: {np.mean(episode_rewards):.6f}")
        print(f"  - Max/Min reward: {max(episode_rewards):.3f} / {min(episode_rewards):.3f}")
        print(f"  - Mario position: X={info['mario_x']:.0f}, Y={info['mario_y']:.0f}")
        print(f"  - Coins collected: {info['coins']}")
        
        env.close()
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_render_modes():
    """異なる render_mode のテスト"""
    print("\n" + "=" * 60)
    print("Test 4: Different Render Modes")
    print("=" * 60)
    
    modes_to_test = [None, 'rgb_array']  # 'human' は対話的なため省略
    
    for mode in modes_to_test:
        print(f"\n  Testing render_mode='{mode}'")
        try:
            env = MarioEnv(render_mode=mode, level='Level1-1')
            obs, info = env.reset()
            
            for _ in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
            
            env.close()
            print(f"  ✓ render_mode='{mode}' works correctly")
            return True
        
        except Exception as e:
            print(f"  ✗ Error with render_mode='{mode}': {e}")
            return False


def test_observation_space():
    """観測空間の妥当性チェック"""
    print("\n" + "=" * 60)
    print("Test 5: Observation Space Validation")
    print("=" * 60)
    
    try:
        env = MarioEnv(render_mode=None)
        obs, info = env.reset()
        
        # Image チェック
        image = obs['image']
        assert isinstance(image, np.ndarray), "Image should be numpy array"
        assert image.dtype == np.uint8, f"Image dtype should be uint8, got {image.dtype}"
        assert image.shape == (3, 480, 640), f"Image shape should be (3, 480, 640), got {image.shape}"
        print("✓ Image observation: shape=(3, 480, 640), dtype=uint8")
        
        # State チェック
        state = obs['state']
        assert isinstance(state, np.ndarray), "State should be numpy array"
        assert state.dtype == np.float32, f"State dtype should be float32, got {state.dtype}"
        assert state.shape == (9,), f"State shape should be (9,), got {state.shape}"
        print("✓ State observation: shape=(9,), dtype=float32")
        print(f"  State values: {state}")
        
        env.close()
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_episodes():
    """複数エピソード実行テスト"""
    print("\n" + "=" * 60)
    print("Test 6: Multiple Episodes")
    print("=" * 60)
    
    try:
        env = MarioEnv(random_level=True, render_mode=None, max_episode_steps=500)
        
        rewards_per_episode = []
        
        for episode in range(3):
            obs, info = env.reset()
            episode_reward = 0.0
            step_count = 0
            
            while True:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                step_count += 1
                
                if terminated or truncated:
                    break
            
            rewards_per_episode.append(episode_reward)
            print(f"  Episode {episode + 1}: Level={info['level']}, "
                  f"Steps={step_count}, Reward={episode_reward:.3f}")
        
        print(f"✓ Multiple episodes completed")
        print(f"  - Mean reward: {np.mean(rewards_per_episode):.3f}")
        print(f"  - Max/Min: {max(rewards_per_episode):.3f} / {min(rewards_per_episode):.3f}")
        
        env.close()
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """全テスト実行"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║  " + "Mario Gym Environment Test Suite".center(54) + "  ║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    results = {}
    
    tests = [
        ("Basic Initialization", test_basic_init),
        ("Step Loop", test_step_loop),
        ("Random Agent", test_random_agent),
        ("Render Modes", test_render_modes),
        ("Observation Space", test_observation_space),
        ("Multiple Episodes", test_multiple_episodes),
    ]
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ Unexpected error in {test_name}: {e}")
            results[test_name] = False
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
