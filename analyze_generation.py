"""
段階3の検証: 探索で作ったステージが、指定した難易度 D* に本当に当たっているかを調べる。

比べるもの（generate_by_difficulty.py の validate_manifest.json）:
  baseline : 生成器の難易度つまみで D* を狙っただけのステージ（探索の初期集団）
  search   : 編集探索で |D - D*| を小さくしたステージ（探索の最終集団）

確かめること:
  1. 別シードでの誤差
     探索は決まったシード (seed=0) の評価に合わせ込んでいる。別シード (seed=1000) で
     測り直し、誤差が膨らんでいないか（シード固有の偶然への過適合）を見る
  2. エージェント群（PPO パネル・ルールパネル）から見た誤差
     DT の D とパネルの D は尺度が違うので、コーパス（段階2）での直線回帰で D* を
     パネルの尺度に写し、その値からのずれを測る
  3. 評価器の癖を突いていないか
     残差 = パネルの D − (DT の D から回帰で予想したパネルの D)。
     探索が「DT だけが難しがる」ステージを作っていれば、探索結果だけ残差が系統的にずれる
     （回帰そのもののずれは両方に同じように効くので、探索とベースラインの差で判断する）

探索の目的関数に使ったパネル（meta.json の panels_used_in_search）は独立な基準ではないので、
その旨を表示する。最終的な判断は、探索に使っていないパネル（審判）で行う。

使用例:
    python analyze_generation.py --gen-dir gen_out/v2
    python analyze_generation.py --gen-dir gen_out/v2 --compare gen_out/v1   # 2つの探索を比べる
"""

import argparse
import json
import os

import numpy as np


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def boot_mean_diff(a, b, n=4000, seed=0):
    """mean(a) - mean(b) と、その 95% ブートストラップ信頼区間"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return a.mean() - b.mean(), *np.percentile(d, [2.5, 97.5])


def boot_mean(a, n=4000, seed=0):
    a = np.asarray(a, float)
    rng = np.random.default_rng(seed)
    return a.mean(), *np.percentile([rng.choice(a, len(a)).mean() for _ in range(n)], [2.5, 97.5])


def verdict(lo, hi, better="探索の方が有意に正確", worse="ベースラインの方が正確"):
    return better if hi < 0 else worse if lo > 0 else "有意差なし"


def analyze(g, args, verbose=True):
    out = print if verbose else (lambda *a, **k: None)
    meta = load(f"{g}/meta.json") if os.path.exists(f"{g}/meta.json") else dict(objective="dt", panels_used_in_search=[])
    used = set(meta.get("panels_used_in_search", []))
    val = load(f"{g}/validate_manifest.json")
    dt = load(f"{g}/val_dt.json")
    panels = {"PPO": load(f"{g}/val_ppo.json"), "ルール": load(f"{g}/val_rule.json")}

    # ---- コーパスで DT の尺度 → パネルの尺度 の直線を求める ----
    cm = load(args.corpus_manifest)
    cdt = load(args.corpus_dt)
    cpan = {"PPO": load(args.corpus_ppo), "ルール": load(args.corpus_rule)}
    xs = np.array([cdt[m["path"]]["D_progress"] for m in cm])
    maps, corpus_resid_sd = {}, {}
    for name, pn in cpan.items():
        ys = np.array([pn[m["path"]]["D_panel_progress"] for m in cm])
        slope, icpt = np.polyfit(xs, ys, 1)
        maps[name] = (lambda d, s=slope, i=icpt: i + s * d)
        corpus_resid_sd[name] = float(np.std(ys - (icpt + slope * xs)))

    # combo の目的関数を別シードで計算し直すための写像（探索時と同じ作り方）
    combo = meta.get("objective") == "combo"
    if combo:
        from generate_by_difficulty import Linker, dt_progress_over
        hi_t = [float(t) for t in meta["dt_targets"]]
        ref = [cdt[m["path"]]["D_progress"] for m in cm]
        link_dt = Linker([dt_progress_over(cdt[m["path"]], hi_t) for m in cm], ref)
        link_rule = Linker([cpan["ルール"][m["path"]]["D_panel_progress"] for m in cm], ref)

    targets = sorted({v["target"] for v in val})
    rows = []
    for v in val:
        p = v["path"]
        r = dict(method=v["method"], target=v["target"], D_search=v["D_search"],
                 D_fresh=dt[p]["D_progress"], ceiling_fresh=dt[p]["ceiling"])
        if combo:
            r["obj_fresh"] = (link_dt(dt_progress_over(dt[p], hi_t))
                              + link_rule(panels["ルール"][p]["D_panel_progress"])) / 2
        for name, pn in panels.items():
            pd_ = pn[p]["D_panel_progress"]
            r[f"{name}_D"] = pd_
            r[f"{name}_err"] = abs(pd_ - maps[name](v["target"]))       # D* をパネルの尺度に写したものからのずれ
            r[f"{name}_resid"] = pd_ - maps[name](r["D_fresh"])         # DT から予想したパネル値からのずれ
        rows.append(r)

    def sel(method, target=None):
        return [r for r in rows if r["method"] == method and (target is None or r["target"] == target)]

    summ = dict(objective=meta.get("objective"), used=used)
    out("=" * 80)
    out(f"検証: {g}  （目的関数: {meta.get('objective')}"
        + (f"、探索に使ったパネル: {', '.join(sorted(used))}" if used else "") + f"、"
        f"ベースライン {len(sel('baseline'))} / 探索 {len(sel('search'))} ステージ）")
    out("=" * 80)

    # ---- 1. 別シードでの誤差 ----
    out("\n[1] DT（全 target）で測った難易度と目標とのずれ |D - D*|")
    out(f"  {'D*':>5s} {'方法':10s} {'探索時のシード':>14s} {'別シード':>10s}   別シードでの D（平均）")
    for t in targets:
        for m in ("baseline", "search"):
            rs = sel(m, t)
            if not rs:
                continue
            es = np.mean([abs(r["D_search"] - t) for r in rs])
            ef = np.mean([abs(r["D_fresh"] - t) for r in rs])
            out(f"  {t:5.2f} {m:10s} {es:14.3f} {ef:10.3f}   {np.mean([r['D_fresh'] for r in rs]):.3f}")
    if combo:
        out("  （combo では「探索時のシード」の列は目的関数の値。DT 単独の値ではない）")
    eb = [abs(r["D_fresh"] - r["target"]) for r in sel("baseline")]
    es = [abs(r["D_fresh"] - r["target"]) for r in sel("search")]
    d, lo, hi = boot_mean_diff(es, eb)
    summ["dt_err"] = (np.mean(eb), np.mean(es), d, lo, hi)
    out(f"  → 別シードでの平均誤差: ベースライン {np.mean(eb):.3f} / 探索 {np.mean(es):.3f}  "
        f"差 {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  ({verdict(lo, hi)})")
    obj_key = "obj_fresh" if combo else "D_fresh"
    over = [abs(r[obj_key] - r["target"]) - abs(r["D_search"] - r["target"]) for r in sel("search")]
    m_, lo, hi = boot_mean(over)
    out(f"  → 探索結果の、シードを変えたことによる目的関数の誤差の増加: {m_:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        f"  （大きいほど、探索時のシードに合わせ込んでいる）")
    if combo:
        ob = [abs(r["obj_fresh"] - r["target"]) for r in sel("baseline")]
        os_ = [abs(r["obj_fresh"] - r["target"]) for r in sel("search")]
        d, lo, hi = boot_mean_diff(os_, ob)
        out(f"  → 目的関数（DT 高 target + ルール）の別シードでの平均誤差: ベースライン {np.mean(ob):.3f} / "
            f"探索 {np.mean(os_):.3f}  差 {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  ({verdict(lo, hi)})")
    n_unclear = sum(r["ceiling_fresh"] <= 0 for r in sel("search"))
    out(f"  → 探索結果のうち、別シードで最高 target でも一度もクリアできなかったもの: {n_unclear}/{len(sel('search'))}")

    # ---- 2. パネルでの誤差 ----
    for name in panels:
        tag = "（探索に使ったので独立ではない）" if name in used else "（探索に使っていない＝独立な審判）"
        out(f"\n[2] {name}パネルから見たずれ {tag}")
        out(f"      D* をコーパスの回帰で{name}の尺度に写した値との差。"
            f"参考: 回帰の残差の標準偏差 {corpus_resid_sd[name]:.3f}（これ以下にはなりにくい）")
        for t in targets:
            vb = [r[f"{name}_err"] for r in sel("baseline", t)]
            vs = [r[f"{name}_err"] for r in sel("search", t)]
            out(f"  D*={t:.2f}: ベースライン {np.mean(vb):.3f}  探索 {np.mean(vs):.3f}   "
                f"（パネルの D 平均: {np.mean([r[f'{name}_D'] for r in sel('baseline', t)]):.3f} → "
                f"{np.mean([r[f'{name}_D'] for r in sel('search', t)]):.3f}, 予想 {maps[name](t):.3f}）")
        eb = [r[f"{name}_err"] for r in sel("baseline")]
        es = [r[f"{name}_err"] for r in sel("search")]
        d, lo, hi = boot_mean_diff(es, eb)
        summ[f"{name}_err"] = (np.mean(eb), np.mean(es), d, lo, hi)
        out(f"  → 平均: ベースライン {np.mean(eb):.3f} / 探索 {np.mean(es):.3f}  差 {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
            f"({verdict(lo, hi)})")

    # ---- 3. 評価器の癖を突いていないか ----
    out("\n[3] 評価器の癖を突いていないか（残差 = パネルの D − DT から予想したパネルの D）")
    out("    探索結果の残差がベースラインからずれていれば、「DT だけが難しがる / 易しがる」ステージを作っている")
    for name in panels:
        tag = "（探索に使った）" if name in used else "（独立な審判）"
        for m in ("baseline", "search"):
            mm, lo, hi = boot_mean([r[f"{name}_resid"] for r in sel(m)])
            out(f"  {name:4s} {m:10s} 残差の平均 {mm:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                + ("  ← 0 を含まない" if lo > 0 or hi < 0 else ""))
        for t in targets:
            mm = np.mean([r[f"{name}_resid"] for r in sel("search", t)])
            out(f"        search D*={t:.2f}: 残差の平均 {mm:+.3f}")
        d, lo, hi = boot_mean_diff([r[f"{name}_resid"] for r in sel("search")],
                                   [r[f"{name}_resid"] for r in sel("baseline")])
        summ[f"{name}_shift"] = (d, lo, hi)
        out(f"        → 探索 − ベースライン の残差の差 {d:+.3f} [{lo:+.3f}, {hi:+.3f}] {tag} "
            f"({'評価器の癖を突いている疑い' if lo > 0 or hi < 0 else '癖を突いている兆候なし'})")
    return summ


def main():
    ap = argparse.ArgumentParser(description="生成したステージの難易度を独立に検証する")
    ap.add_argument("--gen-dir", default="gen_out/v1")
    ap.add_argument("--compare", default=None, help="比べる別の探索結果のディレクトリ（例: gen_out/v1）")
    ap.add_argument("--corpus-manifest", default="corpus/v1/manifest.json")
    ap.add_argument("--corpus-dt", default="validity_out/v1/dt.json")
    ap.add_argument("--corpus-ppo", default="validity_out/v1/panel.json")
    ap.add_argument("--corpus-rule", default="validity_out/v1/rule_panel.json")
    args = ap.parse_args()

    main_s = analyze(args.gen_dir.rstrip("/"), args)
    if not args.compare:
        return
    other = analyze(args.compare.rstrip("/"), args, verbose=False)

    print("\n" + "=" * 80)
    print(f"比較: {args.compare}（{other['objective']}） vs {args.gen_dir}（{main_s['objective']}）")
    print("  ベースラインは同じステージ（つまみ・シードが同じ）。違いは探索の目的関数だけ")
    print("=" * 80)
    print(f"  {'指標':42s} {args.compare:>18s} {args.gen_dir:>18s}")

    def cell(s, key, kind):
        v = s[key]
        if kind == "err":   # 探索 − ベースライン の誤差の差
            return f"{v[2]:+.3f} [{v[3]:+.2f},{v[4]:+.2f}]"
        return f"{v[0]:+.3f} [{v[1]:+.2f},{v[2]:+.2f}]"

    rows = [("DT（全target, 別シード）の誤差: 探索−ベースライン", "dt_err", "err"),
            ("PPO パネルの誤差: 探索−ベースライン", "PPO_err", "err"),
            ("ルールパネルの誤差: 探索−ベースライン", "ルール_err", "err"),
            ("PPO 残差のずれ（評価器の癖を突いた度合い）", "PPO_shift", "shift"),
            ("ルール残差のずれ（同上）", "ルール_shift", "shift")]
    for label, key, kind in rows:
        mark = lambda s, k=key: " *" if any(p in k for p in s["used"]) else ""
        print(f"  {label:42s} {cell(other, key, kind) + mark(other):>18s} {cell(main_s, key, kind) + mark(main_s):>18s}")
    print("  （誤差の差は負ほど探索が正確。残差のずれは 0 に近いほど癖を突いていない。* はそのパネルを探索に使った）")
    print("  （PPO パネルはどちらの探索にも使っていないので、両者を公平に比べられる審判）")


if __name__ == "__main__":
    main()
