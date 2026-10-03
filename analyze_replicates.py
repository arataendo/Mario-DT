"""
段階3 v3/v4 の検証: 独立な探索を複数本回した結果で、3つの生成方法を比べる（generate_replicates.py）。

  naive   : つまみで1個作るだけ
  gentest : つまみで N 個作り、目的関数で最良を選ぶ（評価 N 回）
  search  : 編集探索（評価 N 回）

同じ探索（run）から出た3つは同じ候補から出発しているので、run ごとに対にして差を取り、
ブートストラップは run を単位に再標本化する。

パネル（審判の候補）:
  PPO        : val_ppo.json      コーパスの基準は validity_out/v1/panel.json
  ルール     : val_rule.json     同 rule_panel.json
  プランナー : val_planner.json  同 planner_panel.json（先読みプランナー。どの目的関数にも含まれない）
目的関数に使ったパネル（meta.json の panels_used_in_search）は独立な基準ではないので明示する。

指標は2種類:
  写像を使う : D* をコーパスの直線回帰でパネルの尺度に写し、そこからのずれ（v3 で、両端で回帰が
               合わず全方法に共通の誤差が乗ることが分かった）
  写像を使わない: 同じ目標で作ったステージを審判がどれだけそろって評価するか（目標内のばらつき）、
               目標を上げると審判の評価も順に上がるか（順位相関）、目標で説明できる割合

--compare で別の版（例: v3 = 目的関数 DT+ルール）と、同じ審判から見て比べる。
候補のシードが同じなので「つまみ1個」は両方で同じステージになる（審判の値が一致するはず）。

使用例:
    python analyze_replicates.py --gen-dir gen_out/v4 --compare gen_out/v3
"""

import argparse
import itertools
import os

import numpy as np
from scipy.stats import spearmanr

from analyze_generation import load

METHODS = ["naive", "gentest", "search"]
NAMES = {"naive": "つまみ1個", "gentest": "生成して選ぶ", "search": "編集探索"}
PAIRS = [("search", "naive"), ("gentest", "naive"), ("search", "gentest")]
PANEL_FILES = {"PPO": "val_ppo.json", "ルール": "val_rule.json", "プランナー": "val_planner.json"}


def paired_boot(diffs, n=4000, seed=0):
    d = np.asarray(diffs, float)
    rng = np.random.default_rng(seed)
    bs = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return d.mean(), *np.percentile(bs, [2.5, 97.5])


def tile_vec(path):
    from level_editor import EditableLevel
    e = EditableLevel.from_json(load(path))
    pipe, en = np.zeros(e.length), np.zeros(e.length)
    for x, h in e.pipes.items():
        pipe[x] = h
    for x in e.goombas:
        en[x] = 1
    for x in e.koopas:
        en[x] = 2
    return np.concatenate([np.array(e.ground, float), pipe, en])


def build(g, args):
    """1つの版の結果を読み込み、run × method ごとの指標をまとめる"""
    meta = load(f"{g}/meta.json")
    used = set(meta.get("panels_used_in_search", []))
    val = load(f"{g}/validate_manifest.json")
    dt = load(f"{g}/val_dt.json")
    panels = {n: load(f"{g}/{f}") for n, f in PANEL_FILES.items() if os.path.exists(f"{g}/{f}")}

    cm, cdt = load(args.corpus_manifest), load(args.corpus_dt)
    corpus_files = {"PPO": args.corpus_ppo, "ルール": args.corpus_rule, "プランナー": args.corpus_planner}
    cpan = {n: load(f) for n, f in corpus_files.items() if n in panels and os.path.exists(f)}
    xs = np.array([cdt[m["path"]]["D_progress"] for m in cm])
    maps = {}
    for name, pn in cpan.items():
        ys = np.array([pn[m["path"]]["D_panel_progress"] for m in cm])
        s, i = np.polyfit(xs, ys, 1)
        maps[name] = (lambda d, s=s, i=i: i + s * d)

    # 目的関数を別シードで計算し直すための写像（探索時と同じ作り方）
    links = None
    if meta.get("objective") in ("combo", "combo3"):
        from generate_by_difficulty import Linker, dt_progress_over
        hi_t = [float(t) for t in meta["dt_targets"]]
        ref = [cdt[m["path"]]["D_progress"] for m in cm]
        links = {"DT": (Linker([dt_progress_over(cdt[m["path"]], hi_t) for m in cm], ref),
                        lambda p: dt_progress_over(dt[p], hi_t))}
        for name in used:
            corpus = load(corpus_files[name])
            links[name] = (Linker([corpus[m["path"]]["D_panel_progress"] for m in cm], ref),
                           lambda p, n=name: panels[n][p]["D_panel_progress"])

    M = {}
    for v in val:
        p, t = v["path"], v["target"]
        r = dict(target=t, path=p, obj_search_err=abs(v["D_search"] - t),
                 DT_D=dt[p]["D_progress"], dt_err=abs(dt[p]["D_progress"] - t), unclear=dt[p]["ceiling"] <= 0)
        if links is not None:
            r["obj_err"] = abs(float(np.mean([lk(get(p)) for lk, get in links.values()])) - t)
        for name, pn in panels.items():
            pd_ = pn[p]["D_panel_progress"]
            r[f"{name}_D"] = pd_
            if name in maps:
                r[f"{name}_err"] = abs(pd_ - maps[name](t))
                r[f"{name}_resid"] = pd_ - maps[name](dt[p]["D_progress"])
        M.setdefault(v["run"], {})[v["method"]] = r
    runs = sorted(M)
    targets = sorted({M[r]["naive"]["target"] for r in runs})
    by_t = {t: [r for r in runs if M[r]["naive"]["target"] == t] for t in targets}
    return dict(meta=meta, used=used, M=M, runs=runs, targets=targets, by_t=by_t,
                panels=list(panels), maps=set(maps), links=links is not None)


# ---- 写像を使わない指標 ----
def within_sd(B, method, key, idx=None):
    idx = idx or B["by_t"]
    return float(np.sqrt(np.mean([np.var([B["M"][r][method][key] for r in idx[t]], ddof=1)
                                  for t in B["targets"]])))


def rank_corr(B, method, key):
    rs = B["runs"]
    return spearmanr([B["M"][r][method]["target"] for r in rs], [B["M"][r][method][key] for r in rs]).statistic


def eta2(B, method, key):
    allv = np.array([B["M"][r][method][key] for r in B["runs"]])
    sw = sum(np.sum((np.array(x) - np.mean(x)) ** 2) for x in
             ([B["M"][r][method][key] for r in B["by_t"][t]] for t in B["targets"]))
    return 1 - sw / np.sum((allv - allv.mean()) ** 2)


def resample(B, rng):
    return {t: list(rng.choice(B["by_t"][t], len(B["by_t"][t]))) for t in B["targets"]}


def report(g, B, n_boot):
    M, runs, targets, used = B["M"], B["runs"], B["targets"], B["used"]
    meta = B["meta"]
    print("=" * 88)
    print(f"検証: {g}  目的関数 {meta['objective']}"
          + (f"（使ったパネル: {', '.join(sorted(used))}）" if used else "")
          + f"、目標 {len(targets)} 個 × 独立な探索 {meta['replicates']} 本 = {len(runs)} 本、評価回数 N={meta['N']}")
    print("=" * 88)

    print("\n[0] 同じ目標の中での、ステージどうしの違い（タイル単位）")
    for t in targets:
        cells = []
        for m in METHODS:
            V = [tile_vec(M[r][m]["path"]) for r in B["by_t"][t]]
            cells.append(f"{NAMES[m]} {np.mean([int((a != b).sum()) for a, b in itertools.combinations(V, 2)]):5.1f}")
        print(f"  D*={t:.2f}: " + "   ".join(cells))

    def section(title, key, note=""):
        print(f"\n{title}{note}")
        print(f"  {'D*':>5s} " + " ".join(f"{NAMES[m]:>10s}" for m in METHODS))
        for t in targets:
            print(f"  {t:5.2f} " + " ".join(f"{np.mean([M[r][m][key] for r in B['by_t'][t]]):12.3f}" for m in METHODS))
        print(f"  {'全体':>4s} " + " ".join(f"{np.mean([M[r][m][key] for r in runs]):12.3f}" for m in METHODS))
        for a_, b_ in PAIRS:
            d, lo, hi = paired_boot([M[r][a_][key] - M[r][b_][key] for r in runs])
            v = f"{NAMES[a_]}の方が有意に小さい" if hi < 0 else f"{NAMES[b_]}の方が有意に小さい" if lo > 0 else "有意差なし"
            print(f"    {NAMES[a_]} − {NAMES[b_]}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  ({v})")

    section("[1] 目的関数の誤差（探索時のシード）", "obj_search_err", "  ← 探索・選別が直接最小化した量")
    if B["links"]:
        section("[1'] 目的関数の誤差（別シード）", "obj_err", "  ← 探索時のシードへの合わせ込みが無ければ [1] と同程度")
    for name in B["panels"]:
        if name in B["maps"]:
            tag = "  （目的関数に使ったので独立ではない）" if name in used else "  ← どの方法にも使っていない独立な基準"
            section(f"[2] {name}パネルから見た誤差（写像を使う）", f"{name}_err", tag)

    mapped = [n for n in B["panels"] if n in B["maps"]]
    if mapped:
        print("\n[3] 評価器の癖を突いていないか: 残差（パネルの D − DT から予想した値）の、つまみ1個との差")
        for name in mapped:
            tag = "（目的関数に使った）" if name in used else "（独立な審判）"
            for a_, b_ in PAIRS[:2]:
                d, lo, hi = paired_boot([M[r][a_][f"{name}_resid"] - M[r][b_][f"{name}_resid"] for r in runs])
                print(f"  {name}{tag} {NAMES[a_]} − {NAMES[b_]}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                      + ("  ← 0 を含まない（癖を突いている疑い）" if lo > 0 or hi < 0 else ""))

    print("\n[4] 写像を使わない指標: 同じ目標のステージを、そろって・順に評価するか")
    rng = np.random.default_rng(0)
    for name in ["DT"] + B["panels"]:
        key = "DT_D" if name == "DT" else f"{name}_D"
        tag = ("（目的関数の成分）" if name == "DT" else
               "（目的関数に使った）" if name in used else "（独立な審判）")
        print(f"  ■ {name}{tag}")
        print(f"    {'方法':10s} {'目標内のばらつき SD':>24s} {'順位相関':>8s} {'目標で説明できる割合':>18s}")
        boots = {m: [] for m in METHODS}
        for _ in range(n_boot):
            idx = resample(B, rng)
            for m in METHODS:
                boots[m].append(within_sd(B, m, key, idx))
        for m in METHODS:
            lo, hi = np.percentile(boots[m], [2.5, 97.5])
            print(f"    {NAMES[m]:10s} {within_sd(B, m, key):8.3f} [{lo:.3f}, {hi:.3f}] "
                  f"{rank_corr(B, m, key):+9.2f} {eta2(B, m, key):14.2f}")
        for a_, b_ in PAIRS:
            diffs = np.array(boots[a_]) - np.array(boots[b_])
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            d = within_sd(B, a_, key) - within_sd(B, b_, key)
            v = f"{NAMES[a_]}の方が有意にそろう" if hi < 0 else f"{NAMES[b_]}の方が有意にそろう" if lo > 0 else "有意差なし"
            print(f"      ばらつきの差 {NAMES[a_]} − {NAMES[b_]}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  ({v})")

    n_unclear = {m: sum(M[r][m]["unclear"] for r in runs) for m in METHODS}
    print("\n[5] 別シードで最高 target の DT でも一度もクリアできなかったステージ: "
          + ", ".join(f"{NAMES[m]} {n_unclear[m]}/{len(runs)}" for m in METHODS))


def compare(ga, A, gb, B, n_boot):
    """2つの版を、どちらの目的関数にも使っていない同じ審判から見て比べる（run ごとに対）"""
    judges = [n for n in A["panels"] if n in B["panels"] and n not in A["used"] and n not in B["used"]]
    print("\n" + "=" * 88)
    print(f"比較: {gb}（{B['meta']['objective']}） → {ga}（{A['meta']['objective']}）")
    print("  候補のシードが同じなので「つまみ1個」は両方で同じステージ。違いは目的関数だけ")
    print("=" * 88)
    if not judges:
        print("  どちらの目的関数にも使っていない共通のパネルがありません")
        return
    common = [r for r in A["runs"] if r in B["runs"]]
    for name in judges:
        key = f"{name}_D"
        gap = max(abs(A["M"][r]["naive"][key] - B["M"][r]["naive"][key]) for r in common)
        print(f"\n■ 審判: {name}  （確認: 同じ「つまみ1個」での審判の値の最大差 {gap:.4f}。"
              f"{'一致' if gap < 1e-9 else '不一致 ← 測定条件が違う可能性'}）")
        print(f"    {'方法':10s} {'目標内のばらつき ' + gb:>22s} {ga:>10s}   {'順位相関 ' + gb:>14s} {ga:>8s}")
        for m in METHODS:
            print(f"    {NAMES[m]:10s} {within_sd(B, m, key):22.3f} {within_sd(A, m, key):10.3f}   "
                  f"{rank_corr(B, m, key):+14.2f} {rank_corr(A, m, key):+8.2f}")
        rng = np.random.default_rng(1)
        for m in ("gentest", "search"):
            diffs = []
            for _ in range(n_boot):
                idx = resample(A, rng)   # 同じ run を両方の版から取る（対の比較）
                diffs.append(within_sd(A, m, key, idx) - within_sd(B, m, key, idx))
            d = within_sd(A, m, key) - within_sd(B, m, key)
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            v = f"{ga} の方が有意にそろう" if hi < 0 else f"{gb} の方が有意にそろう" if lo > 0 else "有意差なし"
            print(f"      {NAMES[m]} のばらつき {ga} − {gb}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  ({v})")


def main():
    ap = argparse.ArgumentParser(description="複数本の探索で、3つの生成方法を対にして比べる")
    ap.add_argument("--gen-dir", default="gen_out/v3")
    ap.add_argument("--compare", default=None, help="同じ審判で比べる別の版（例: gen_out/v3）")
    ap.add_argument("--corpus-manifest", default="corpus/v1/manifest.json")
    ap.add_argument("--corpus-dt", default="validity_out/v1/dt.json")
    ap.add_argument("--corpus-ppo", default="validity_out/v1/panel.json")
    ap.add_argument("--corpus-rule", default="validity_out/v1/rule_panel.json")
    ap.add_argument("--corpus-planner", default="validity_out/v1/planner_panel.json")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    g = args.gen_dir.rstrip("/")
    A = build(g, args)
    report(g, A, args.boot)
    if args.compare:
        gb = args.compare.rstrip("/")
        compare(g, A, gb, build(gb, args), args.boot)


if __name__ == "__main__":
    main()
