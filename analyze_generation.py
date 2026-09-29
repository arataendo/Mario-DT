"""
段階3の検証: 探索で作ったステージが、指定した難易度 D* に本当に当たっているかを調べる。

比べるもの（generate_by_difficulty.py の validate_manifest.json）:
  baseline : 生成器の難易度つまみで D* を狙っただけのステージ（探索の初期集団）
  search   : 編集探索で |D - D*| を小さくしたステージ（探索の最終集団）

確かめること:
  1. 別シードでの DT の誤差
     探索は決まったシード (seed=0) の評価に合わせ込んでいる。別シード (seed=1000) で
     測り直し、誤差が膨らんでいないか（シード固有の偶然への過適合）を見る
  2. 独立なエージェント群（PPO パネル・ルールパネル）から見た誤差
     DT の D とパネルの D は尺度が違うので、コーパス（段階2）での直線回帰で D* を
     パネルの尺度に写し、その値からのずれを測る
  3. DT の癖を突いていないか
     残差 = パネルの D − (DT の D から回帰で予想したパネルの D)。
     探索が「DT だけが難しがる」ステージを作っていれば、探索結果だけ残差が系統的にずれる

使用例:
    python analyze_generation.py --gen-dir gen_out/v1
"""

import argparse
import json

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


def main():
    ap = argparse.ArgumentParser(description="生成したステージの難易度を独立に検証する")
    ap.add_argument("--gen-dir", default="gen_out/v1")
    ap.add_argument("--corpus-manifest", default="corpus/v1/manifest.json")
    ap.add_argument("--corpus-dt", default="validity_out/v1/dt.json")
    ap.add_argument("--corpus-ppo", default="validity_out/v1/panel.json")
    ap.add_argument("--corpus-rule", default="validity_out/v1/rule_panel.json")
    args = ap.parse_args()
    g = args.gen_dir.rstrip("/")

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

    targets = sorted({v["target"] for v in val})
    rows = []
    for v in val:
        p = v["path"]
        r = dict(method=v["method"], target=v["target"], D_search=v["D_search"],
                 D_fresh=dt[p]["D_progress"], ceiling_fresh=dt[p]["ceiling"])
        for name, pn in panels.items():
            pd_ = pn[p]["D_panel_progress"]
            r[f"{name}_D"] = pd_
            r[f"{name}_err"] = abs(pd_ - maps[name](v["target"]))       # D* をパネルの尺度に写したものからのずれ
            r[f"{name}_resid"] = pd_ - maps[name](r["D_fresh"])         # DT から予想したパネル値からのずれ
        rows.append(r)

    def sel(method, target=None):
        return [r for r in rows if r["method"] == method and (target is None or r["target"] == target)]

    print("=" * 80)
    print(f"検証: {g}  （ベースライン {len(sel('baseline'))} / 探索 {len(sel('search'))} ステージ）")
    print("=" * 80)

    # ---- 1. DT での誤差（探索に使ったシード / 別シード）----
    print("\n[1] DT で測った難易度と目標とのずれ |D - D*|")
    print(f"  {'D*':>5s} {'方法':10s} {'探索時のシード':>14s} {'別シード':>10s}   別シードでの D（平均）")
    for t in targets:
        for m in ("baseline", "search"):
            rs = sel(m, t)
            if not rs:
                continue
            es = np.mean([abs(r["D_search"] - t) for r in rs])
            ef = np.mean([abs(r["D_fresh"] - t) for r in rs])
            print(f"  {t:5.2f} {m:10s} {es:14.3f} {ef:10.3f}   {np.mean([r['D_fresh'] for r in rs]):.3f}")
    eb = [abs(r["D_fresh"] - r["target"]) for r in sel("baseline")]
    es = [abs(r["D_fresh"] - r["target"]) for r in sel("search")]
    d, lo, hi = boot_mean_diff(es, eb)
    print(f"  → 別シードでの平均誤差: ベースライン {np.mean(eb):.3f} / 探索 {np.mean(es):.3f}  "
          f"差 {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
          f"({'探索の方が有意に正確' if hi < 0 else 'ベースラインの方が正確' if lo > 0 else '有意差なし'})")
    over = [abs(r["D_fresh"] - r["target"]) - abs(r["D_search"] - r["target"]) for r in sel("search")]
    m_, lo, hi = boot_mean(over)
    print(f"  → 探索結果の、シードを変えたことによる誤差の増加: {m_:+.3f} [{lo:+.3f}, {hi:+.3f}]"
          f"  （大きいほど、探索時のシードに合わせ込んでいる）")
    n_unclear = sum(r["ceiling_fresh"] <= 0 for r in sel("search"))
    print(f"  → 探索結果のうち、別シードで最高 target でも一度もクリアできなかったもの: {n_unclear}/{len(sel('search'))}")

    # ---- 2. 独立なパネルでの誤差 ----
    for name in panels:
        print(f"\n[2] {name}パネルから見たずれ（D* をコーパスの回帰で{name}の尺度に写した値との差）")
        print(f"      参考: コーパスでの回帰の残差の標準偏差 {corpus_resid_sd[name]:.3f}（これ以下にはなりにくい）")
        for t in targets:
            vb = [r[f"{name}_err"] for r in sel("baseline", t)]
            vs = [r[f"{name}_err"] for r in sel("search", t)]
            print(f"  D*={t:.2f}: ベースライン {np.mean(vb):.3f}  探索 {np.mean(vs):.3f}   "
                  f"（パネルの D 平均: {np.mean([r[f'{name}_D'] for r in sel('baseline', t)]):.3f} → "
                  f"{np.mean([r[f'{name}_D'] for r in sel('search', t)]):.3f}, 予想 {maps[name](t):.3f}）")
        eb = [r[f"{name}_err"] for r in sel("baseline")]
        es = [r[f"{name}_err"] for r in sel("search")]
        d, lo, hi = boot_mean_diff(es, eb)
        print(f"  → 平均: ベースライン {np.mean(eb):.3f} / 探索 {np.mean(es):.3f}  差 {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
              f"({'探索の方が有意に正確' if hi < 0 else 'ベースラインの方が正確' if lo > 0 else '有意差なし'})")

    # ---- 3. DT の癖を突いていないか ----
    print("\n[3] DT の癖を突いていないか（残差 = パネルの D − DT から予想したパネルの D）")
    print("    探索結果の残差が 0 から系統的にずれていれば、「DT だけが難しがる / 易しがる」ステージを作っている")
    for name in panels:
        for m in ("baseline", "search"):
            mm, lo, hi = boot_mean([r[f"{name}_resid"] for r in sel(m)])
            print(f"  {name:4s} {m:10s} 残差の平均 {mm:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                  + ("  ← 0 を含まない" if lo > 0 or hi < 0 else ""))
        for t in targets:
            mm = np.mean([r[f"{name}_resid"] for r in sel("search", t)])
            print(f"        search D*={t:.2f}: 残差の平均 {mm:+.3f}")
        # 回帰そのものがずれていれば両方が同じようにずれるので、探索とベースラインの差で判断する
        d, lo, hi = boot_mean_diff([r[f"{name}_resid"] for r in sel("search")],
                                   [r[f"{name}_resid"] for r in sel("baseline")])
        print(f"        → 探索 − ベースライン の残差の差 {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
              f"({'探索が DT の癖を突いている疑い' if lo > 0 or hi < 0 else '癖を突いている兆候なし'})")


if __name__ == "__main__":
    main()
