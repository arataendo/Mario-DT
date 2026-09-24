"""
diagnose_rtg.py で「RTG は行動分布に効いている」ことが分かったため、
非単調性の原因を別方向から調べる。

仮説: 推論時の (timestep, RTG) の組み合わせが学習データに存在しない領域へ逸脱している。

  学習データでは RTG は「そのエピソードの残り収益」なので、
  時間が進むほど必ず 0 に向かって単調減少する。
  つまり「t が大きいのに RTG が高い」というサンプルは存在しない。

  一方 infer_dt.py は cur_rtg_raw = target から実報酬を引いていくだけなので、
  エージェントが期待通り稼げないと RTG が高いまま t だけが進む。
  これは学習データに無い (t, RTG) 領域 = 外挿になり、行動が壊れる。

この仮説が正しければ、target を上げるほど逸脱が早く始まり、
高い target ほど性能が落ちるという観測と一致する。

使用方法:
    python diagnose_rtg_drift.py --model models/mario_dt_20260822_233744_epoch30.pth \
        --data dt_dataset_v3/metadata.pkl --level Level_test_medium
"""

import os
import argparse
import pickle

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

from classes.MarioGymEnv import MarioEnv
from classes.wrappers import SkipFrame, MarioImageWrapper, StallGuard
from infer_dt import load_model


def dataset_rtg_envelope(episodes, n_buckets, bucket_size):
    """学習データ中の timestep バケットごとの RTG 分布（最小/最大/分位点）"""
    buckets = [[] for _ in range(n_buckets)]
    for ep in episodes:
        rtg = ep["returns_to_go"]
        for t, v in enumerate(rtg):
            b = t // bucket_size
            if b < n_buckets:
                buckets[b].append(float(v))
    return buckets


def run_and_log_rtg(model, rtg_min, rtg_max, context_len, level, target,
                    max_steps, device, args_no_clip=False):
    """1 エピソード走らせ、各ステップの (t, cur_rtg_raw) を記録する"""
    skip = 4
    env = MarioEnv(level=level, render_mode=None, max_episode_steps=max_steps * skip)
    env = SkipFrame(env, skip=skip)
    env = MarioImageWrapper(env)

    obs, info = env.reset()
    guard = StallGuard()
    mario_x = 0

    states, actions, rtgs = [], [], []
    rtg_span = max(rtg_max - rtg_min, 1e-5)
    cur_rtg_raw = target
    trace = []
    reason = "running"

    for t in range(max_steps):
        trace.append((t, cur_rtg_raw))

        img = obs.astype(np.float32) / 255.0
        states.append(img)
        rtgs.append((cur_rtg_raw - rtg_min) / rtg_span)
        actions.append(0)

        window = min(len(states), context_len)
        s = np.array(states[-window:], dtype=np.float32)
        a = np.array(actions[-window:], dtype=np.int64)
        r = np.array(rtgs[-window:], dtype=np.float32)
        ts = np.arange(max(0, t - window + 1), t + 1, dtype=np.int64)

        pad = context_len - window
        if pad > 0:
            s = np.concatenate([np.zeros((pad, 3, 84, 84), dtype=np.float32), s], axis=0)
            a = np.concatenate([np.zeros(pad, dtype=np.int64), a], axis=0)
            r = np.concatenate([np.zeros(pad, dtype=np.float32), r], axis=0)
            ts = np.concatenate([np.zeros(pad, dtype=np.int64), ts], axis=0)
            mask = np.concatenate([np.zeros(pad, dtype=np.float32), np.ones(window, dtype=np.float32)])
        else:
            mask = np.ones(context_len, dtype=np.float32)

        with torch.no_grad():
            logits = model(
                torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0),
                torch.tensor(a, dtype=torch.long, device=device).unsqueeze(0),
                torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1),
                torch.tensor(ts, dtype=torch.long, device=device).unsqueeze(0),
                attention_mask=torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0),
            )
            action = int(torch.argmax(logits[0, -1]).item())

        action = guard.choose(action, mario_x)
        actions[-1] = action

        obs, reward, terminated, truncated, info = env.step(action)
        mario_x = info.get("mario_x", mario_x)
        cur_rtg_raw -= reward
        if not args_no_clip:
            cur_rtg_raw = float(np.clip(cur_rtg_raw, rtg_min, rtg_max))

        if terminated or truncated:
            reason = info.get("reason", "timeout" if truncated else "unknown")
            break

    env.close()
    return trace, reason


def main():
    parser = argparse.ArgumentParser(description="推論時 RTG が学習データ分布から逸脱するかを診断する")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, default="dt_dataset_v3/metadata.pkl")
    parser.add_argument("--level", type=str, default="Level_test_medium")
    parser.add_argument("--targets", type=str, default="0,150,300")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--bucket-size", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-clip", action="store_true",
                        help="推論時の RTG クリップを無効化する（修正前の挙動を再現する）")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, rtg_min, rtg_max, context_len = load_model(args.model, device)

    with open(args.data, "rb") as f:
        episodes = [ep for ep in pickle.load(f) if len(ep["actions"]) > 0]

    n_buckets = args.max_steps // args.bucket_size + 1
    buckets = dataset_rtg_envelope(episodes, n_buckets, args.bucket_size)

    # t=0 の RTG = そのエピソードの総収益。推論時に target を与えるのはこの位置なので、
    # 「target がデータに支えられているか」はこの分布で決まる。
    start_rtg = np.array([float(ep["returns_to_go"][0]) for ep in episodes])
    print("=" * 78)
    print("[0] t=0 における RTG の分布（= エピソード総収益 = target を与える位置）")
    print("=" * 78)
    for target in [float(t) for t in args.targets.split(",")]:
        near = int((np.abs(start_rtg - target) <= 25).sum())
        print(f"  target={target:6.0f} の ±25 以内にある学習エピソード: {near:4d} / {len(start_rtg)} "
              f"({near / len(start_rtg) * 100:5.1f}%)")
    print()

    print("=" * 78)
    print("[A] 学習データ中の timestep 別 RTG 分布")
    print("=" * 78)
    print(f"{'timestep':>12s} {'件数':>8s} {'最小':>8s} {'5%':>8s} {'中央':>8s} {'95%':>8s} {'最大':>8s}")
    env_lo, env_hi = {}, {}        # min/max（緩い基準）
    p_lo, p_hi = {}, {}            # 5-95 パーセンタイル（実質的な密度域）
    for b, vals in enumerate(buckets):
        lo_t, hi_t = b * args.bucket_size, (b + 1) * args.bucket_size
        if not vals:
            print(f"{lo_t:5d}-{hi_t:5d} {0:8d}  (データ無し)")
            continue
        v = np.array(vals)
        env_lo[b], env_hi[b] = float(v.min()), float(v.max())
        p_lo[b], p_hi[b] = float(np.percentile(v, 5)), float(np.percentile(v, 95))
        print(f"{lo_t:5d}-{hi_t:5d} {len(v):8d} {v.min():8.1f} {np.percentile(v, 5):8.1f} "
              f"{np.median(v):8.1f} {np.percentile(v, 95):8.1f} {v.max():8.1f}")

    print()
    print("=" * 78)
    print(f"[B] 推論時の RTG 軌跡と、学習データ範囲からの逸脱  (level={args.level})")
    print("=" * 78)

    targets = [float(t) for t in args.targets.split(",")]
    for target in targets:
        print(f"\n  --- target_return = {target:.0f} ---")
        for ep in range(args.episodes):
            trace, reason = run_and_log_rtg(
                model, rtg_min, rtg_max, context_len, args.level, target,
                args.max_steps, device, args_no_clip=args.no_clip,
            )
            # 各ステップで、その timestep バケットの学習データ RTG 範囲を外れているか判定
            out, out_p = 0, 0
            first_out = None
            for t, rtg in trace:
                b = t // args.bucket_size
                if b in env_hi:
                    if rtg > env_hi[b] or rtg < env_lo[b]:
                        out += 1
                        if first_out is None:
                            first_out = t
                    if rtg > p_hi[b] or rtg < p_lo[b]:
                        out_p += 1
            n = max(len(trace), 1)
            fo = f"t={first_out}" if first_out is not None else "なし"
            print(f"    ep{ep + 1}: steps={len(trace):3d} 終了={reason:14s} "
                  f"範囲外(min/max)={out * 100 / n:5.1f}% 疎な領域(5-95%外)={out_p * 100 / n:5.1f}% "
                  f"逸脱開始={fo}")


if __name__ == "__main__":
    main()
