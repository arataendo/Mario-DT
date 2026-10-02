"""
審判（PPO パネル）自身の測定誤差を見積もり、それを差し引いて3つの生成方法を比べ直す。

v3 では、目的関数（DT + ルール）で見ると最適化でステージのばらつきが半分になったのに、
審判から見ると3つの方法に差が無かった。これが
  (a) 最適化した精度が審判に伝わっていない（評価器固有の難しさしか制御できていない）のか
  (b) 審判の測定誤差が大きく、差を見分けられないだけなのか
を切り分けるため、同じステージを別シードでもう一度審判に測らせ（val_ppo_retest.json）、
2回の差から審判の測定誤差を出す。

  審判の1回あたりの測定誤差 σ_e = SD(1回目 − 2回目) / √2
  観測された目標内のばらつき² = 本当のばらつき² + σ_e²   （平均を取ると σ_e² / 回数）
  → 本当のばらつき = √(観測² − σ_e²/回数)

使用例:
    python panel_difficulty.py --levels-from gen_out/v3/validate_manifest.json --episodes 8 \
        --workers 8 --seed 2000 --output gen_out/v3/val_ppo_retest.json
    python analyze_judge_noise.py --gen-dir gen_out/v3
"""

import argparse

import numpy as np
from scipy.stats import pearsonr, spearmanr

from analyze_generation import load

METHODS = ["naive", "gentest", "search"]
NAMES = {"naive": "つまみ1個", "gentest": "生成して選ぶ", "search": "編集探索"}


def main():
    ap = argparse.ArgumentParser(description="審判の測定誤差を見積もり、差し引いて方法を比べ直す")
    ap.add_argument("--gen-dir", default="gen_out/v3")
    ap.add_argument("--first", default="val_ppo.json")
    ap.add_argument("--retest", default="val_ppo_retest.json")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    g = args.gen_dir.rstrip("/")

    val = load(f"{g}/validate_manifest.json")
    a, b = load(f"{g}/{args.first}"), load(f"{g}/{args.retest}")
    targets = sorted({v["target"] for v in val})
    key = "D_panel_progress"

    d1 = np.array([a[v["path"]][key] for v in val])
    d2 = np.array([b[v["path"]][key] for v in val])
    noise = float(np.std(d1 - d2, ddof=1) / np.sqrt(2))

    print("=" * 80)
    print(f"審判の測定誤差: {g}（{len(val)} ステージを2回ずつ）")
    print("=" * 80)
    print(f"  再測定の一致: Pearson r={pearsonr(d1, d2).statistic:.3f}, Spearman rho={spearmanr(d1, d2).statistic:.3f}")
    print(f"  1回あたりの測定誤差 σ_e = {noise:.3f}   （2回平均なら {noise / np.sqrt(2):.3f}）")
    print(f"  参考: 審判の D の全体のばらつき（SD）{np.std(np.concatenate([d1, d2]), ddof=1):.3f}")

    # run（探索1本）ごとに、3つの方法の値をまとめる
    runs = {}
    for v, x1, x2 in zip(val, d1, d2):
        runs.setdefault(v["run"], {"target": v["target"]})[v["method"]] = (x1, x2)
    run_ids = sorted(runs)
    by_t = {t: [r for r in run_ids if runs[r]["target"] == t] for t in targets}

    def within_sd(method, which, idx):
        """目標ごとのばらつきをプールした SD。which: 0=1回目, 1=2回目, 2=2回の平均"""
        vs = []
        for t in targets:
            xs = [runs[r][method] for r in idx[t]]
            xs = [x[which] if which < 2 else (x[0] + x[1]) / 2 for x in xs]
            vs.append(np.var(xs, ddof=1))
        return float(np.sqrt(np.mean(vs)))

    def true_sd(obs, n_meas):
        return float(np.sqrt(max(obs ** 2 - noise ** 2 / n_meas, 0.0)))

    rng = np.random.default_rng(0)
    full = {t: by_t[t] for t in targets}
    print("\n[1] 同じ目標で作ったステージを、審判がどれだけそろって評価するか（目標内のばらつき SD）")
    print("    観測値には審判の測定誤差が上乗せされている。「誤差を引いた値」が方法本来のばらつき")
    print(f"  {'方法':10s} {'1回目':>7s} {'2回目':>7s} {'2回平均':>8s}   {'誤差を引いた値（2回平均から）':>24s}")
    est = {}
    for m in METHODS:
        o1, o2, om = (within_sd(m, w, full) for w in (0, 1, 2))
        ts = true_sd(om, 2)
        boots = []
        for _ in range(args.boot):
            idx = {t: list(rng.choice(by_t[t], len(by_t[t]))) for t in targets}
            boots.append(true_sd(within_sd(m, 2, idx), 2))
        est[m] = (ts, *np.percentile(boots, [2.5, 97.5]))
        print(f"  {NAMES[m]:10s} {o1:7.3f} {o2:7.3f} {om:8.3f}   {ts:8.3f} [{est[m][1]:.3f}, {est[m][2]:.3f}]")
    print(f"  （誤差を引いた値が 0 に近い ＝ 観測されたばらつきはほぼ審判の測定誤差だけ）")

    print("\n[2] 方法間の差（誤差を引いた目標内のばらつき。run を目標ごとに復元抽出して対で比較）")
    for x, y in [("search", "naive"), ("gentest", "naive"), ("search", "gentest")]:
        diffs = []
        for _ in range(args.boot):
            idx = {t: list(rng.choice(by_t[t], len(by_t[t]))) for t in targets}
            diffs.append(true_sd(within_sd(x, 2, idx), 2) - true_sd(within_sd(y, 2, idx), 2))
        d = est[x][0] - est[y][0]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        v = f"{NAMES[x]}の方が有意にそろっている" if hi < 0 else f"{NAMES[y]}の方が有意にそろっている" if lo > 0 else "有意差なし"
        print(f"  {NAMES[x]} − {NAMES[y]}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  ({v})")

    print("\n[3] 目標を上げると審判の評価も順に上がるか（2回平均で）")
    for m in METHODS:
        xs = [(runs[r]["target"], (runs[r][m][0] + runs[r][m][1]) / 2) for r in run_ids]
        print(f"  {NAMES[m]:10s} 順位相関 {spearmanr([t for t, _ in xs], [x for _, x in xs]).statistic:+.3f}")


if __name__ == "__main__":
    main()
