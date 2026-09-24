"""パッチ後の動作確認スクリプト（1〜2分で終わる想定）

    python test_smoke_ppo.py

学習を長時間回す前に、これが全部 ✅ になることを確認してください。
"""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

results = []


def check(name, cond, extra=""):
    results.append((name, cond))
    print(("✅ " if cond else "❌ ") + name + (f"   {extra}" if extra else ""))


print("=" * 60)
print("1) 環境とラッパーの確認")
print("=" * 60)

from classes.wrappers import make_mario_env, N_STACK, SKIP

LEVEL = sys.argv[1] if len(sys.argv) > 1 else "Level1-1"
MAX_FRAMES = 8000

env = make_mario_env(level=LEVEL, max_episode_steps=MAX_FRAMES, render_mode=None)
obs, info = env.reset()

check("観測形状 (12, 84, 84)", obs.shape == (3 * N_STACK, 84, 84), str(obs.shape))
check("観測 dtype uint8", obs.dtype == np.uint8)
check("行動空間 Discrete(10)", env.action_space.n == 10, str(env.action_space))

# 1 agent step = SKIP ゲームフレーム
obs, r, te, tr, info = env.step(2)
check(f"1 agent step = {SKIP} ゲームフレーム",
      info["episode_step"] == SKIP, f"episode_step={info['episode_step']}")

print()
print("=" * 60)
print("2) 右に走り続けたときにどこまで進めるか")
print("=" * 60)

obs, info = env.reset()
done = False
steps = 0
total_r = 0.0
while not done:
    # 8 = Right + Dash + Jump（走りジャンプ）を混ぜて進む
    action = 8 if steps % 3 == 0 else 7
    obs, r, te, tr, info = env.step(action)
    total_r += r
    done = te or tr
    steps += 1

print(f"   意思決定 {steps} 回 / {info['episode_step']} ゲームフレーム")
print(f"   到達 X = {info['max_x']}  進捗 = {info['progress'] * 100:.1f}%")
print(f"   終了理由 = {info.get('reason')}  合計報酬 = {total_r:.1f}")

check("エピソードが 250 意思決定で打ち切られていない（＝旧バグの解消）",
      steps > 250 or info.get("reason") != "timeout",
      f"steps={steps}, reason={info.get('reason')}")
check("走りジャンプが実際に効いている（少しは前進する）", info["max_x"] > 100)

env.close()

print()
print("=" * 60)
print("3) PPO が数ステップ回るか")
print("=" * 60)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from train_ppo import make_env, MarioStatsCallback

venv = VecMonitor(DummyVecEnv([make_env(LEVEL, 0, max_episode_steps=MAX_FRAMES)]))
model = PPO("CnnPolicy", venv, n_steps=64, batch_size=32, n_epochs=1,
            verbose=0, device="cpu")
model.learn(total_timesteps=128, callback=[MarioStatsCallback()], progress_bar=False)
check("PPO の学習ループが例外なく回る", True)
venv.close()

print()
print("=" * 60)
n_ok = sum(c for _, c in results)
print(f"{n_ok}/{len(results)} PASS")
if n_ok == len(results):
    print("\n次のコマンドで本学習を開始してください:")
    print(f"  python train_ppo.py --level {LEVEL} --total-steps 2000000 "
          f"--num-envs 8 --max-episode-steps 8000")
sys.exit(0 if n_ok == len(results) else 1)
