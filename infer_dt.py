"""
学習済み Decision Transformer にマリオをプレイさせる。

DT は「目標収益 (target return)」を条件に自己回帰的に行動を予測する。
過去 context_len ステップの (RTG, State, Action) 系列をスライディングウィンドウで
保持しながら、毎ステップ次の行動を予測する標準的な DT 推論ループ。

使用方法:
    python infer_dt.py --model models/mario_dt_20260808_130035_epoch15.pth \
        --level Level1-1 --target-return 300 --render
"""

import os
import argparse
from collections import deque

import numpy as np
import torch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from classes.MarioGymEnv import MarioEnv
# ★ DT は PPO と違い単一フレーム(3,84,84)を使う（時間情報は context_len 側の
#   系列で扱うため）。SkipFrame/MarioImageWrapper だけ共有し FrameStack は使わない。
from classes.wrappers import SkipFrame, MarioImageWrapper, StallGuard
from train_dt import DecisionTransformer, ACTION_VOCAB_SIZE


def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    frame_stack = ckpt.get("frame_stack", 1)
    model = DecisionTransformer(
        action_vocab_size=ckpt.get("action_vocab_size", ACTION_VOCAB_SIZE),
        hidden_size=ckpt["hidden_size"],
        context_len=ckpt["context_len"],
        frame_stack=frame_stack,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.frame_stack = frame_stack
    return model, ckpt["rtg_min"], ckpt["rtg_max"], ckpt["context_len"]


def run_episode(model, rtg_min, rtg_max, context_len, level, target_return,
                 max_steps, device, render, sample=False, temperature=1.0):
    # MarioEnv の max_episode_steps は生フレーム単位。SkipFrame(skip=4) を通すと
    # 1マクロステップ=4フレームになるため、意図したマクロステップ数になるよう skip 倍する。
    skip = 4
    env = MarioEnv(level=level, render_mode="human" if render else None, max_episode_steps=max_steps * skip)
    env = SkipFrame(env, skip=skip)
    env = MarioImageWrapper(env)

    obs, info = env.reset()
    guard = StallGuard()
    mario_x = 0

    k = getattr(model, "frame_stack", 1)
    recent = deque(maxlen=k)   # 直近フレーム。状態はこれを重ねて作る
    states = []      # list of (3*k,84,84) float32 in [0,1]
    actions = []      # list of int
    rtgs = []         # list of float (normalized 0-1)
    rtg_span = max(rtg_max - rtg_min, 1e-5)

    cur_rtg_raw = target_return
    total_reward = 0.0
    t = 0
    reason = "running"

    for t in range(max_steps):
        frame = obs.astype(np.float32) / 255.0
        if not recent:
            # エピソード先頭は過去フレームが無いので先頭フレームで埋める（学習時と同じ扱い）
            for _ in range(k):
                recent.append(frame)
        else:
            recent.append(frame)
        states.append(np.concatenate(list(recent), axis=0))
        norm_rtg = (cur_rtg_raw - rtg_min) / rtg_span
        rtgs.append(norm_rtg)
        # 直近の行動が無ければダミー(0)で埋める。予測後に実際の行動で置き換える
        actions.append(0)

        # 直近 context_len 分を取り出す
        window = min(len(states), context_len)
        s = np.array(states[-window:], dtype=np.float32)
        a = np.array(actions[-window:], dtype=np.int64)
        r = np.array(rtgs[-window:], dtype=np.float32)
        ts = np.arange(max(0, t - window + 1), t + 1, dtype=np.int64)

        pad = context_len - window
        if pad > 0:
            s = np.concatenate([np.zeros((pad, 3 * k, 84, 84), dtype=np.float32), s], axis=0)
            a = np.concatenate([np.zeros(pad, dtype=np.int64), a], axis=0)
            r = np.concatenate([np.zeros(pad, dtype=np.float32), r], axis=0)
            ts = np.concatenate([np.zeros(pad, dtype=np.int64), ts], axis=0)
        mask = np.concatenate([np.zeros(pad, dtype=np.float32), np.ones(window, dtype=np.float32)]) if pad > 0 \
            else np.ones(context_len, dtype=np.float32)

        with torch.no_grad():
            s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            a_t = torch.tensor(a, dtype=torch.long, device=device).unsqueeze(0)
            r_t = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
            ts_t = torch.tensor(ts, dtype=torch.long, device=device).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)

            logits = model(s_t, a_t, r_t, ts_t, attention_mask=mask_t)
            if sample:
                # argmax は分布がどれだけ平らでも最頻値を返すため、低 RTG で学習した
                # 「乱れた行動分布」が挙動に反映されない。サンプリングなら分布の形が
                # そのまま行動に出るので、target_return による制御が効きやすくなる。
                probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
                action = int(torch.multinomial(probs, 1).item())
            else:
                action = int(torch.argmax(logits[0, -1]).item())

        # 障害物前でargmaxが固着する自己ループを防ぐ安全弁（infer_ppo.py と同じ仕組み）
        action = guard.choose(action, mario_x)
        actions[-1] = action  # 予測した行動で置き換え

        obs, reward, terminated, truncated, info = env.step(action)
        mario_x = info.get("mario_x", mario_x)
        total_reward += reward
        cur_rtg_raw -= reward
        # 学習データに存在する RTG 範囲へクリップする。
        # これが無いと、例えば target=0 で走らせたとき RTG が報酬の分だけ
        # 際限なく負へ流れ（-300 など、学習時の最小値 -32 を大きく下回る）、
        # 条件付け入力が範囲外に飽和して事実上無視されてしまう。
        # diagnose_rtg_drift.py では target=0 のとき全ステップの 77〜86% が
        # 学習範囲外になっていた。
        cur_rtg_raw = float(np.clip(cur_rtg_raw, rtg_min, rtg_max))

        if render:
            env.render()

        if terminated or truncated:
            reason = info.get("reason", "timeout" if truncated else "unknown")
            break

    env.close()
    mario_x = info.get("mario_x", 0)
    return total_reward, t + 1, mario_x, reason


def main():
    parser = argparse.ArgumentParser(description="学習済み Decision Transformer でマリオをプレイする")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--level", type=str, default="Level1-1")
    parser.add_argument("--target-return", type=float, default=300.0,
                         help="条件付ける目標収益")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sample", action="store_true", help="argmax ではなく分布からサンプリングする")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, rtg_min, rtg_max, context_len = load_model(args.model, device)
    print(f"モデル読み込み完了 (context_len={context_len}, rtg範囲=[{rtg_min:.1f}, {rtg_max:.1f}])")

    for ep in range(args.episodes):
        total_reward, steps, mario_x, reason = run_episode(
            model, rtg_min, rtg_max, context_len,
            args.level, args.target_return, args.max_steps, device, args.render,
            sample=args.sample, temperature=args.temperature,
        )
        print(f"ep{ep+1}: target={args.target_return:.0f} 実績reward={total_reward:.1f} "
              f"steps={steps} X={mario_x} reason={reason}")


if __name__ == "__main__":
    main()
