"""
ルールベースのエージェント群（rule_agent.py）でステージの難易度を測る。

PPO パネル (panel_difficulty.py) と同じ形式の JSON を出力するので、
analyze_validity.py にそのまま渡せる。

環境シードの規則も DT・PPO パネルと同じ（エピソード e は env_seed=seed+e）で、
敵の初期の向きがステージ・エージェント間で共通になる。

使用例:
    python rule_panel.py --levels-from corpus/v1/manifest.json --episodes 8 --workers 8 \
        --output validity_out/v1/rule_panel.json
    python rule_panel.py --calibrate          # 学習用ステージで各プリセットの腕前を確かめる
"""

import argparse
import json
import multiprocessing as mp
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from eval_common import read_levels, summarize_panel
from rule_agent import PRESETS, run_episode

# 腕前の確認用。評価に使うコーパスとは別の、学習用ステージで調整する（評価データで調整しない）
CALIB_LEVELS = ["Level_easy_01", "Level_easy_02", "Level_medium_01", "Level_medium_02",
                "Level_hard_01", "Level_hard_02", "Level_hard_03"]


def evaluate(levels, agent_names, episodes=8, seed=0, workers=0, max_steps=500):
    jobs = [dict(level=lv, agent=a, agent_name=name, episode=e, env_seed=seed + e,
                 sample_seed=(seed + e) * 1000 + 700 + a, max_steps=max_steps)
            for lv in levels for a, name in enumerate(agent_names) for e in range(episodes)]
    if workers > 0:
        ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
        with ctx.Pool(workers) as pool:
            results = pool.map(run_episode, jobs, chunksize=4)
    else:
        results = [run_episode(j) for j in jobs]
    return summarize_panel(results, agent_names)


def main():
    ap = argparse.ArgumentParser(description="ルールベースのエージェント群でステージの難易度を測る")
    ap.add_argument("--levels", default="")
    ap.add_argument("--levels-from", default=None, help="make_corpus.py の manifest.json")
    ap.add_argument("--agents", default=",".join(PRESETS), help="使うプリセット名（カンマ区切り）")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true",
                    help="学習用ステージで各プリセットの平均クリア率を確かめる（評価コーパスは使わない）")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    names = [s.strip() for s in args.agents.split(",") if s.strip()]
    unknown = [n for n in names if n not in PRESETS]
    if unknown:
        raise SystemExit(f"未知のプリセット: {unknown}（選べるもの: {list(PRESETS)}）")
    levels = CALIB_LEVELS if args.calibrate else read_levels(args.levels, args.levels_from)

    t0 = time.time()
    res = evaluate(levels, names, args.episodes, args.seed, args.workers, args.max_steps)
    el = time.time() - t0
    print(f"{len(levels) * len(names) * args.episodes} エピソードを {el:.0f} 秒で評価\n")

    print(f"{'stage':24s} " + " ".join(f"{n:>8s}" for n in names))
    for lv in levels:
        print(f"{os.path.basename(lv):24s} "
              + " ".join(f"{a['clear_rate'] * 100:7.0f}%" for a in res[lv]["agents"]))
    print(f"{'平均クリア率':22s} " + " ".join(
        f"{np.mean([res[lv]['agents'][i]['clear_rate'] for lv in levels]) * 100:7.0f}%"
        for i in range(len(names))))
    print(f"{'平均到達率':23s} " + " ".join(
        f"{np.mean([res[lv]['agents'][i]['progress'] for lv in levels]) * 100:7.0f}%"
        for i in range(len(names))))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\n💾 {args.output}")


if __name__ == "__main__":
    main()
