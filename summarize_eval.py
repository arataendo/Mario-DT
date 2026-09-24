"""
run_eval.sh が出力したステージ別の評価結果（<ディレクトリ>/<ステージ>.json）を集計する。

  - ステージ × target のクリア率表と、ステージごとの Spearman 順位相関
  - target 別の平均クリア率と全体の単調性 (Spearman rho, p 値)
  - 未知ステージ（Level_test_*）の難易度別クリア率

使い方:
    python summarize_eval.py eval_out/v10s
    python summarize_eval.py eval_out/v10s eval_out/v11   # 複数を並べて比較
"""

import glob
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

ORDER = ["Level_test_easy", "Level_test_easy_02", "Level_test_easy_03",
         "Level_test_medium", "Level_test_medium_02", "Level_test_medium_03",
         "Level_test_hard", "Level_test_hard_02", "Level_test_hard_03",
         "Level_easy_01", "Level_medium_01", "Level_hard_01", "Level_hard_02"]


def load(d):
    res = {}
    for p in glob.glob(os.path.join(d, "*.json")):
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        for key, v in raw.items():
            # eval_dt_matrix.py の出力は "ステージ|target" キー。それ以外の JSON は無視する
            if "|" not in key or not isinstance(v, dict) or "clear_rate" not in v:
                continue
            level, t = key.split("|")
            res.setdefault(level, {})[float(t)] = v["clear_rate"] * 100
    targets = sorted({t for v in res.values() for t in v})
    levels = [l for l in ORDER if l in res] + sorted(l for l in res if l not in ORDER)
    table = {l: [res[l].get(t, np.nan) for t in targets] for l in levels}
    return targets, table


def rho(targets, rows):
    xs = [t for r in rows for t, v in zip(targets, r) if not np.isnan(v)]
    ys = [v for r in rows for v in r if not np.isnan(v)]
    if len(set(ys)) < 2:
        return float("nan"), float("nan")
    r = spearmanr(xs, ys)
    return r.statistic, r.pvalue


def report(d):
    targets, table = load(d)
    if not table:
        print(f"{d}: 結果がありません")
        return
    print("=" * 78)
    print(f"{d}  ({len(table)} ステージ)")
    print("=" * 78)
    print(f"{'stage':22s} " + " ".join(f"{int(t):>5d}" for t in targets) + "    rho")
    for l, row in table.items():
        r, _ = rho(targets, [row])
        rs = f"{r:+.2f}" if not np.isnan(r) else "  n/a"
        print(f"{l:22s} " + " ".join(f"{v:4.0f}%" if not np.isnan(v) else "   - " for v in row) + f"  {rs}")

    rows = list(table.values())
    print("\ntarget 別の平均クリア率:")
    for i, t in enumerate(targets):
        print(f"  target={int(t):4d}: {np.nanmean([r[i] for r in rows]):5.1f}%")
    r, p = rho(targets, rows)
    print(f"  全体 Spearman rho={r:+.2f}  p={p:.4f}")

    tests = {k: v for k, v in table.items() if k.startswith("Level_test_")}
    if tests:
        print("\n未知ステージ（Level_test_*）の難易度別:")
        for diff in ("easy", "medium", "hard"):
            sub = [v for k, v in tests.items() if f"_{diff}" in k]
            if sub:
                r, p = rho(targets, sub)
                print(f"  {diff:7s}({len(sub)}種) " +
                      " ".join(f"{np.nanmean([s[i] for s in sub]):4.0f}%" for i in range(len(targets))) +
                      f"   rho={r:+.2f} p={p:.3f}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for d in sys.argv[1:]:
        report(d)
