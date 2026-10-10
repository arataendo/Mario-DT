"""
高い target だけサンプリングの温度を下げると、DT の腕前の上限が上がり、かつ target による
腕前の条件付けが保たれるかを確かめる（run_temp_check.sh が呼ぶ）。

  [A] 学習用・テスト用ステージ（13個）: target ごとのクリア率、上限、条件付けの強さ
  [B] 評価コーパス（60個）: 難しい帯での上限を他のエージェント群と比べる。
      難易度 D_progress と各パネルの一致（単純・構造を除いた偏相関）が崩れないか

使用例:
    python analyze_temperature.py \
        --levels T1.0=temp_out/levels_t10.json hi0.7=temp_out/levels_hi07.json \
        --corpus T1.0=validity_out/v1/dt.json hi0.7=temp_out/corpus_hi07.json
"""

import argparse
import os

import numpy as np
from scipy.stats import rankdata, spearmanr

from analyze_validity import load, partial_boot, spearman_boot


def pairs(items):
    out = []
    for s in items or []:
        name, path = s.split("=", 1)
        if os.path.exists(path):
            out.append((name, load(path)))
        else:
            print(f"（{path} が無いので {name} は飛ばします）")
    return out


def group_of(path):
    b = os.path.basename(path).replace(".json", "")
    for d in ("easy", "medium", "hard"):
        if b.startswith(f"Level_test_{d}"):
            return f"テスト {d}"
    return "学習用"


def conditioning(res):
    """target を上げるとクリア率が上がる度合い。ステージごとの順位相関の平均と、全体をまとめた順位相関"""
    per, xs, ys = [], [], []
    for r in res.values():
        t = [c["target"] for c in r["curve"]]
        cr = [c["clear_rate"] for c in r["curve"]]
        xs += t
        ys += cr
        if len(set(cr)) > 1:
            per.append(spearmanr(t, cr).statistic)
    return float(np.mean(per)) if per else float("nan"), spearmanr(xs, ys).statistic


def section_levels(runs):
    print("=" * 84)
    print("[A] 学習用・テスト用ステージ: target ごとのクリア率（各グループの平均）")
    print("=" * 84)
    targets = [c["target"] for c in next(iter(runs[0][1].values()))["curve"]]
    groups = ["テスト easy", "テスト medium", "テスト hard", "学習用"]
    for g in groups:
        print(f"\n  {g}")
        print(f"    {'設定':10s} " + " ".join(f"{t:>6.0f}" for t in targets) + "    D_progress")
        for name, res in runs:
            lv = [r for p, r in res.items() if group_of(p) == g]
            if not lv:
                continue
            cr = np.mean([[c["clear_rate"] for c in r["curve"]] for r in lv], axis=0)
            dp = np.mean([r["D_progress"] for r in lv])
            print(f"    {name:10s} " + " ".join(f"{v * 100:5.0f}%" for v in cr) + f"    {dp:.3f}")
    print("\n  条件付けの強さ（target とクリア率の順位相関。基準の設定より大きく下がっていなければ保たれている）")
    print(f"    {'設定':10s} {'ステージごとの平均':>16s} {'全体':>8s} {'最高 target の平均クリア率':>22s}")
    for name, res in runs:
        per, pooled = conditioning(res)
        ceil = np.mean([r["ceiling"] for r in res.values()])
        print(f"    {name:10s} {per:16.2f} {pooled:8.2f} {ceil * 100:21.0f}%")
    print("  テストの易・中・難の D_progress が、この順に大きくなっていれば物差しとして使える")


def section_corpus(runs, manifest, panels):
    print("\n" + "=" * 84)
    print("[B] 評価コーパス 60 ステージ")
    print("=" * 84)
    mm = {m["path"]: m for m in manifest}
    paths = sorted(mm, key=lambda p: mm[p]["difficulty_param"])
    thirds = [("つまみ下位1/3", paths[:20]), ("中位1/3", paths[20:40]), ("上位1/3", paths[40:])]

    print("\n  最高 target でのクリア率（他のエージェント群は各群で最も強いエージェント）")
    head = [n for n, _ in runs] + [n for n, _ in panels]
    print(f"    {'':14s} " + " ".join(f"{h:>10s}" for h in head))
    for label, ps in thirds:
        vals = [np.mean([r[p]["ceiling"] for p in ps]) for _, r in runs]
        vals += [np.mean([max(a["clear_rate"] for a in pr[p]["agents"]) for p in ps]) for _, pr in panels]
        print(f"    {label:14s} " + " ".join(f"{v * 100:9.0f}%" for v in vals))

    base_name, base = runs[0]
    for name, res in runs[1:]:
        same = diff = 0
        for p in paths:
            for c0, c1 in zip(base[p]["curve"], res[p]["curve"]):
                if c0["clear_rate"] == c1["clear_rate"] and abs(c0["progress"] - c1["progress"]) < 1e-9:
                    same += 1
                else:
                    diff += 1
        print(f"\n  確認: {name} と {base_name} で結果が完全に一致した (ステージ, target) の組 {same}/{same + diff}"
              "（温度を変えていない target は一致するはず）")

    S = np.column_stack([rankdata([mm[p][k] for p in paths])
                         for k in ("difficulty_param", "enemies", "gap_tiles", "pipes")])
    print("\n  DT の D_progress と各群の D_progress の一致（評価コーパス、括弧は95%信頼区間）")
    print(f"    {'設定':10s} {'群':10s} {'単純 rho':>22s} {'構造を除いた偏相関':>24s}")
    for name, res in runs:
        x = np.array([res[p]["D_progress"] for p in paths])
        for pname, pr in panels:
            y = np.array([pr[p]["D_panel_progress"] for p in paths])
            r, lo, hi = spearman_boot(x, y)
            q, qlo, qhi = partial_boot(rankdata(x), rankdata(y), S)
            print(f"    {name:10s} {pname:10s} {r:+.2f} [{lo:+.2f}, {hi:+.2f}]    {q:+.2f} [{qlo:+.2f}, {qhi:+.2f}]")
    for name, res in runs[1:]:
        r = spearmanr([base[p]["D_progress"] for p in paths], [res[p]["D_progress"] for p in paths]).statistic
        print(f"  {name} の D_progress と {base_name} の D_progress の順位相関: {r:+.2f}")


def main():
    ap = argparse.ArgumentParser(description="DT の温度を変えたときの上限と条件付けを比べる")
    ap.add_argument("--levels", nargs="*", help="名前=difficulty.py の出力（最初が基準）")
    ap.add_argument("--corpus", nargs="*", help="名前=評価コーパスでの difficulty.py の出力（最初が基準）")
    ap.add_argument("--manifest", default="corpus/v1/manifest.json")
    ap.add_argument("--panels", nargs="*", default=["PPO=validity_out/v1/panel.json",
                                                    "ルール=validity_out/v1/rule_panel.json",
                                                    "プランナー=validity_out/v1/planner_panel.json"])
    a = ap.parse_args()
    lv = pairs(a.levels)
    if lv:
        section_levels(lv)
    cp = pairs(a.corpus)
    if len(cp) >= 1:
        section_corpus(cp, load(a.manifest), pairs(a.panels))


if __name__ == "__main__":
    main()
