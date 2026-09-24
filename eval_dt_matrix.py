"""
Decision Transformer が「目標収益(target_return)に応じて難易度の異なる
レベルをクリアし分けられているか」を検証する評価スクリプト。

複数レベル × 複数 target_return の組み合わせで複数エピソードずつ実行し、
クリア率・平均報酬・平均到達距離を一覧表示する。

使用方法:
    python eval_dt_matrix.py --model models/mario_dt_20260820_133757_epoch30.pth
"""

import argparse
import json
import numpy as np
import torch

from infer_dt import load_model, run_episode


def main():
    parser = argparse.ArgumentParser(description="DTの target_return × レベル 評価マトリクス")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--levels", type=str,
                         default="Level_easy_01,Level_easy_02,Level_medium_01,Level_medium_02,"
                                 "Level_hard_01,Level_hard_02")
    parser.add_argument("--targets", type=str, default="0,150,300")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", action="store_true", help="argmax ではなく分布からサンプリングする")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="dt_eval_matrix_result.json",
                        help="結果 JSON の保存先。複数プロセスで並列評価するときは別々にすること")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, rtg_min, rtg_max, context_len = load_model(args.model, device)
    print(f"モデル読み込み完了 (context_len={context_len}, rtg範囲=[{rtg_min:.1f}, {rtg_max:.1f}])\n")

    levels = [l.strip() for l in args.levels.split(",")]
    targets = [float(t.strip()) for t in args.targets.split(",")]

    results = {}
    header = f"{'level':16s} " + " ".join(f"target={t:<6.0f}" for t in targets)
    print(header)
    print("-" * len(header))

    for level in levels:
        row = []
        for target in targets:
            rewards, xs, clears = [], [], []
            for ep in range(args.episodes):
                total_reward, steps, mario_x, reason = run_episode(
                    model, rtg_min, rtg_max, context_len,
                    level, target, args.max_steps, device, render=False,
                    sample=args.sample, temperature=args.temperature,
                )
                rewards.append(total_reward)
                xs.append(mario_x)
                clears.append(1 if reason == "level_complete" else 0)
            clear_rate = np.mean(clears)
            mean_reward = np.mean(rewards)
            mean_x = np.mean(xs)
            results[(level, target)] = dict(clear_rate=clear_rate, mean_reward=mean_reward, mean_x=mean_x)
            row.append(f"clr={clear_rate*100:3.0f}% r={mean_reward:6.1f}")
        print(f"{level:16s} " + "  ".join(row))

    print()
    print("=" * 60)
    print("詳細 (mean_x = 平均到達X座標)")
    print("=" * 60)
    for level in levels:
        for target in targets:
            r = results[(level, target)]
            print(f"{level:16s} target={target:6.0f}  clear_rate={r['clear_rate']*100:5.1f}%  "
                  f"mean_reward={r['mean_reward']:7.1f}  mean_x={r['mean_x']:6.1f}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({f"{lvl}|{t}": v for (lvl, t), v in results.items()}, f, ensure_ascii=False, indent=2)
    print(f"\n💾 結果を {args.output} に保存しました")


if __name__ == "__main__":
    main()
