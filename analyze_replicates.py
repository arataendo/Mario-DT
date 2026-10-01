"""
段階3 v3 の検証: 独立な探索を複数本回した結果で、3つの生成方法を比べる（generate_replicates.py）。

  naive   : つまみで1個作るだけ
  gentest : つまみで N 個作り、目的関数で最良を選ぶ（評価 N 回）
  search  : 編集探索（評価 N 回）

同じ探索（run）から出た3つは同じ候補から出発しているので、run ごとに対にして差を取り、
ブートストラップは run を単位に再標本化する（v2 ではほぼ同じ系統のステージを独立とみなしたため、
信頼区間が狭く出すぎていた）。

判定は、どの方法の評価にも使っていない PPO パネル（審判）で行う。
ルールパネルは combo の目的関数に使っているので、独立な基準ではない（参考として表示）。

使用例:
    python analyze_replicates.py --gen-dir gen_out/v3
"""

import argparse
import itertools
import json

import numpy as np

from analyze_generation import load

METHODS = ["naive", "gentest", "search"]
NAMES = {"naive": "つまみ1個", "gentest": "生成して選ぶ", "search": "編集探索"}
PAIRS = [("search", "naive"), ("gentest", "naive"), ("search", "gentest")]


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


def main():
    ap = argparse.ArgumentParser(description="複数本の探索で、3つの生成方法を対にして比べる")
    ap.add_argument("--gen-dir", default="gen_out/v3")
    ap.add_argument("--corpus-manifest", default="corpus/v1/manifest.json")
    ap.add_argument("--corpus-dt", default="validity_out/v1/dt.json")
    ap.add_argument("--corpus-ppo", default="validity_out/v1/panel.json")
    ap.add_argument("--corpus-rule", default="validity_out/v1/rule_panel.json")
    args = ap.parse_args()
    g = args.gen_dir.rstrip("/")

    meta = load(f"{g}/meta.json")
    used = set(meta.get("panels_used_in_search", []))
    val = load(f"{g}/validate_manifest.json")
    dt = load(f"{g}/val_dt.json")
    panels = {"PPO": load(f"{g}/val_ppo.json"), "ルール": load(f"{g}/val_rule.json")}

    # コーパスで DT の尺度 → パネルの尺度 の直線（analyze_generation.py と同じ）
    cm, cdt = load(args.corpus_manifest), load(args.corpus_dt)
    cpan = {"PPO": load(args.corpus_ppo), "ルール": load(args.corpus_rule)}
    xs = np.array([cdt[m["path"]]["D_progress"] for m in cm])
    maps = {}
    for name, pn in cpan.items():
        ys = np.array([pn[m["path"]]["D_panel_progress"] for m in cm])
        s, i = np.polyfit(xs, ys, 1)
        maps[name] = (lambda d, s=s, i=i: i + s * d)
    combo = meta.get("objective") == "combo"
    if combo:
        from generate_by_difficulty import Linker, dt_progress_over
        hi_t = [float(t) for t in meta["dt_targets"]]
        ref = [cdt[m["path"]]["D_progress"] for m in cm]
        link_dt = Linker([dt_progress_over(cdt[m["path"]], hi_t) for m in cm], ref)
        link_rule = Linker([cpan["ルール"][m["path"]]["D_panel_progress"] for m in cm], ref)

    # run × method ごとの指標
    M = {}
    for v in val:
        p, t = v["path"], v["target"]
        r = dict(target=t, path=p, obj_search_err=abs(v["D_search"] - t),
                 dt_err=abs(dt[p]["D_progress"] - t), unclear=dt[p]["ceiling"] <= 0)
        if combo:
            obj = (link_dt(dt_progress_over(dt[p], hi_t)) + link_rule(panels["ルール"][p]["D_panel_progress"])) / 2
            r["obj_err"] = abs(obj - t)
        for name, pn in panels.items():
            pd_ = pn[p]["D_panel_progress"]
            r[f"{name}_D"] = pd_
            r[f"{name}_err"] = abs(pd_ - maps[name](t))
            r[f"{name}_resid"] = pd_ - maps[name](dt[p]["D_progress"])
        M.setdefault(v["run"], {})[v["method"]] = r
    runs = sorted(M)
    targets = sorted({M[r]["naive"]["target"] for r in runs})

    print("=" * 84)
    print(f"検証: {g}  目的関数 {meta['objective']}、目標 {len(targets)} 個 × 独立な探索 {meta['replicates']} 本 "
          f"= {len(runs)} 本、各方法の評価回数 N={meta['N']}")
    print("=" * 84)

    # ---- 0. 独立性の確認: 同じ目標の探索結果どうしが別物か ----
    print("\n[0] 同じ目標の中での、ステージどうしの違い（タイル単位。v2 の探索結果は 4〜53 だった）")
    for t in targets:
        cells = []
        for m in METHODS:
            V = [tile_vec(M[r][m]["path"]) for r in runs if M[r][m]["target"] == t]
            cells.append(f"{NAMES[m]} {np.mean([int((a != b).sum()) for a, b in itertools.combinations(V, 2)]):5.1f}")
        print(f"  D*={t:.2f}: " + "   ".join(cells))

    def section(title, key, note=""):
        print(f"\n{title}{note}")
        print(f"  {'D*':>5s} " + " ".join(f"{NAMES[m]:>10s}" for m in METHODS))
        for t in targets:
            rs = [r for r in runs if M[r]["naive"]["target"] == t]
            print(f"  {t:5.2f} " + " ".join(f"{np.mean([M[r][m][key] for r in rs]):12.3f}" for m in METHODS))
        print(f"  {'全体':>4s} " + " ".join(f"{np.mean([M[r][m][key] for r in runs]):12.3f}" for m in METHODS))
        for a_, b_ in PAIRS:
            d, lo, hi = paired_boot([M[r][a_][key] - M[r][b_][key] for r in runs])
            v = f"{NAMES[a_]}の方が有意に小さい" if hi < 0 else f"{NAMES[b_]}の方が有意に小さい" if lo > 0 else "有意差なし"
            print(f"    {NAMES[a_]} − {NAMES[b_]}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  ({v})")

    section("[1] 目的関数の誤差（探索時のシード）", "obj_search_err", "  ← 探索・選別が直接最小化した量")
    if combo:
        section("[1'] 目的関数の誤差（別シード）", "obj_err", "  ← 探索時のシードへの合わせ込みが無ければ [1] と同程度")
    section("[2] 審判（PPO パネル）から見た誤差", "PPO_err", "  ← どの方法にも使っていない独立な基準。これが本題")
    section("[3] ルールパネルから見た誤差", "ルール_err",
            "  （目的関数に使ったので独立ではない）" if "ルール" in used else "")

    print("\n[4] 評価器の癖を突いていないか: 残差（パネルの D − DT から予想した値）の、方法間の差")
    print("    つまみ1個（何も最適化していない）との差が 0 から外れていれば、最適化で評価器の癖を突いている")
    for name in panels:
        tag = "（目的関数に使った）" if name in used else "（独立な審判）"
        for a_, b_ in PAIRS[:2]:
            d, lo, hi = paired_boot([M[r][a_][f"{name}_resid"] - M[r][b_][f"{name}_resid"] for r in runs])
            print(f"  {name:4s}{tag} {NAMES[a_]} − {NAMES[b_]}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                  + ("  ← 0 を含まない（癖を突いている疑い）" if lo > 0 or hi < 0 else ""))
        for t in targets:
            rs = [r for r in runs if M[r]["naive"]["target"] == t]
            print(f"      D*={t:.2f}: " + "  ".join(
                f"{NAMES[m]} {np.mean([M[r][m][f'{name}_resid'] for r in rs]):+.3f}" for m in METHODS))

    n_unclear = {m: sum(M[r][m]["unclear"] for r in runs) for m in METHODS}
    print("\n[5] 別シードで最高 target の DT でも一度もクリアできなかったステージ: "
          + ", ".join(f"{NAMES[m]} {n_unclear[m]}/{len(runs)}" for m in METHODS))


if __name__ == "__main__":
    main()
