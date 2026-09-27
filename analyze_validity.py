"""
DT による難易度 (difficulty.py) の信頼性と妥当性を調べる。

  信頼性: 同じコーパスを別のシードでもう一度測ったとき、同じ順位になるか（再測定の一致）
          ここが低いと、どんな基準とも高い相関は出ようがない
  妥当性: PPO パネルの難易度 (panel_difficulty.py) と、どの指標がよく一致するか

比べる指標:
  DT 由来   : D_clear / D_progress / 1-上限クリア率
  ベースライン: 生成器の難易度つまみ、敵の数、穴のタイル数、パイプ数、構造の合成（z スコアの和）

「DT の指標は、ステージの中身を数えるだけの素朴な指標よりパネルの難易度をよく当てるか」が
最初の主張になる。順位相関 (Spearman) を、ステージを復元抽出するブートストラップで
95% 信頼区間付きで出し、DT の最良指標とベースラインの最良指標の差にも信頼区間を付ける。

使用例:
    python analyze_validity.py --manifest corpus/v1/manifest.json \
        --dt out/dt.json --dt-retest out/dt_retest.json --panel out/panel.json
"""

import argparse
import csv
import json

import numpy as np
from scipy.stats import spearmanr


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _const(v):
    return len(set(np.round(v, 12))) < 2


def spearman_boot(x, y, n_boot=2000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if _const(x) or _const(y):
        return float("nan"), float("nan"), float("nan")  # 値が全部同じだと順位相関は定義できない
    rho = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i])) > 1 and len(set(y[i])) > 1:
            boots.append(spearmanr(x[i], y[i]).statistic)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan"))
    return rho, lo, hi


def diff_boot(a, b, y, n_boot=2000, seed=1):
    """rho(a, y) - rho(b, y) のブートストラップ信頼区間（同じ再標本で両方を計算する）"""
    a, b, y = (np.asarray(v, float) for v in (a, b, y))
    if _const(a) or _const(b) or _const(y):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if min(len(set(a[i])), len(set(b[i])), len(set(y[i]))) > 1:
            d.append(spearmanr(a[i], y[i]).statistic - spearmanr(b[i], y[i]).statistic)
    obs = spearmanr(a, y).statistic - spearmanr(b, y).statistic
    lo, hi = np.percentile(d, [2.5, 97.5]) if d else (float("nan"), float("nan"))
    return obs, lo, hi


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-9)


def main():
    ap = argparse.ArgumentParser(description="DT 難易度の信頼性・妥当性を分析する")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dt", required=True)
    ap.add_argument("--dt-retest", default=None)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--panel-retest", default=None)
    ap.add_argument("--csv", default=None, help="ステージごとの値を CSV に書き出す（図を描く用）")
    args = ap.parse_args()

    man = load(args.manifest)
    dt, panel = load(args.dt), load(args.panel)
    paths = [m["path"] for m in man if m["path"] in dt and m["path"] in panel]
    if len(paths) < len(man):
        print(f"⚠️  DT とパネルの両方に結果があるのは {len(paths)}/{len(man)} ステージ")
    mm = {m["path"]: m for m in man}
    n = len(paths)

    measures = {
        "DT: D_clear": [dt[p]["D_clear"] for p in paths],
        "DT: D_progress": [dt[p]["D_progress"] for p in paths],
        "DT: 1-上限クリア率": [1 - dt[p]["ceiling"] for p in paths],
        "生成器の難易度つまみ": [mm[p]["difficulty_param"] for p in paths],
        "敵の数": [mm[p]["enemies"] for p in paths],
        "穴のタイル数": [mm[p]["gap_tiles"] for p in paths],
        "パイプ数": [mm[p]["pipes"] for p in paths],
    }
    measures["構造の合成 (z和)"] = list(z(measures["敵の数"]) + z(measures["穴のタイル数"]) + z(measures["パイプ数"]))
    dt_keys = [k for k in measures if k.startswith("DT:")]
    base_keys = [k for k in measures if not k.startswith("DT:")]
    refs = {
        "パネル D_clear": [panel[p]["D_panel_clear"] for p in paths],
        "パネル D_progress": [panel[p]["D_panel_progress"] for p in paths],
    }

    print("=" * 78)
    print(f"対象: {n} ステージ")
    print("=" * 78)

    # ---- パネルの腕前のばらつき（パネルが物差しとして機能しているか）----
    agents = panel[paths[0]]["agents"]
    print("\n[パネル] エージェント別の平均クリア率（腕前にばらつきが無いと物差しにならない）")
    for a_i, a in enumerate(agents):
        cr = np.mean([panel[p]["agents"][a_i]["clear_rate"] for p in paths])
        pg = np.mean([panel[p]["agents"][a_i]["progress"] for p in paths])
        print(f"  {a['model']:48s} クリア {cr * 100:5.1f}%  到達 {pg * 100:5.1f}%")

    # ---- 信頼性 ----
    print("\n[信頼性] 別シードで測り直したときの順位の一致 (Spearman)")
    if args.dt_retest:
        rt = load(args.dt_retest)
        for key in ("D_clear", "D_progress"):
            ps = [p for p in paths if p in rt]
            r, lo, hi = spearman_boot([dt[p][key] for p in ps], [rt[p][key] for p in ps])
            print(f"  DT {key:11s}: rho={r:+.2f}  [{lo:+.2f}, {hi:+.2f}]")
    else:
        print("  （--dt-retest 未指定のため省略）")
    if args.panel_retest:
        rt = load(args.panel_retest)
        for key in ("D_panel_clear", "D_panel_progress"):
            ps = [p for p in paths if p in rt]
            r, lo, hi = spearman_boot([panel[p][key] for p in ps], [rt[p][key] for p in ps])
            print(f"  パネル {key:16s}: rho={r:+.2f}  [{lo:+.2f}, {hi:+.2f}]")

    # ---- 妥当性 ----
    for ref_name, ref in refs.items():
        print(f"\n[妥当性] 基準 = {ref_name}  (Spearman rho と 95%信頼区間)")
        rows = []
        for k, v in measures.items():
            r, lo, hi = spearman_boot(v, ref)
            rows.append((k, r, lo, hi))
            print(f"  {k:22s} rho={r:+.2f}  [{lo:+.2f}, {hi:+.2f}]")
        rho_of = {k: (r if np.isfinite(r) else -np.inf) for k, r, *_ in rows}
        best_dt = max(dt_keys, key=rho_of.get)
        best_base = max(base_keys, key=rho_of.get)
        d, lo, hi = diff_boot(measures[best_dt], measures[best_base], ref)
        verdict = ("計算不可（値が一定の指標あり）" if not np.isfinite(lo) else
                   "DT の方が有意に良い" if lo > 0 else
                   "ベースラインの方が有意に良い" if hi < 0 else "有意差なし")
        print(f"  → DT 最良「{best_dt}」 − ベースライン最良「{best_base}」 = {d:+.2f}  "
              f"[{lo:+.2f}, {hi:+.2f}]  ({verdict})")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            cols = list(measures) + list(refs)
            w.writerow(["path"] + cols)
            for i, p in enumerate(paths):
                w.writerow([p] + [measures[c][i] if c in measures else refs[c][i] for c in cols])
        print(f"\n💾 {args.csv}")


if __name__ == "__main__":
    main()
