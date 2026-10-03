"""
段階3: 指定した難易度 D* のステージを、タイル単位の編集探索で作る。

難易度は DT による D_progress（difficulty.py）。段階2で、再測定の一致 0.88、
独立なエージェント群とも構造を超えて一致することを確かめた指標。

探索: 目標ごとに (μ+λ) 進化戦略
  1. 初期集団 = 生成器の難易度つまみで D* を狙って作ったステージ（= ベースライン）
     つまみ→D の関係はコーパス (段階2) の直線回帰を逆算して決める
  2. 各世代、親をランダムに選んで編集 (level_editor.py) した子を λ 個作る
  3. 子をまとめて DT で評価（全目標の子を1バッチにして GPU / 並列環境を効率よく使う）
  4. 親と子から |D - D*| の小さい μ 個を残す
     「最高 target の DT でも一度もクリアできない」ステージは不合格（遊べないステージの排除）

目的関数 (--objective):
  dt    : DT の D_progress だけ（v1）。検証で、探索が DT の弱点を突くことが分かった:
          ・スタート直後に障害物を置き、わざと下手にプレイする低い target の DT だけを止める
          ・パネルが難しがる敵を減らし、DT が苦手な穴・パイプを増やす
  combo : DT（低い target 0・60 を外したもの）とルールパネルの平均。
          DT だけが苦手な要素を増やしても、ルールパネルが難しがらなければ適応度が上がらない。
          PPO パネルは探索に使わず、最終検証の審判として取っておく。
          各成分はコーパス上の分布の対応（同じ順位の値どうし）で D* の尺度（DT の全 target 版）にそろえる。
          回帰で写すと値が平均に縮み、目標値の意味がずれるため。

各ステージは決まったシードで測る（同じステージには常に同じ値がつく）。
探索がそのシード固有の偶然に合わせ込む可能性があるので、最後に
validate_generated.py で別のシードと独立なエージェント群で測り直す。

世代ごとに状態を保存するので、途中で止まっても同じコマンドで続きから再開できる。

使用例:
    python generate_by_difficulty.py --model models/mario_dt_20260924_111016_epoch20.pth \
        --targets 0.25,0.45,0.65 --out gen_out/v1 --workers 8
"""

import argparse
import csv
import json
import os
import time

import numpy as np

from generate_level import generate_level
from level_editor import EditableLevel

UNCLEARABLE_PENALTY = 1.0


def knob_mapping(corpus_manifest, corpus_dt):
    """コーパスでの「つまみ → D_progress」の直線を求め、D* からつまみを逆算する関数を返す"""
    man = json.load(open(corpus_manifest, encoding="utf-8"))
    dt = json.load(open(corpus_dt, encoding="utf-8"))
    k = np.array([m["difficulty_param"] for m in man])
    d = np.array([dt[m["path"]]["D_progress"] for m in man])
    slope, intercept = np.polyfit(k, d, 1)
    return (lambda target: float(np.clip((target - intercept) / slope, 0.05, 0.95))), (intercept, slope)


def fitness(r, target):
    return abs(r["D"] - target) + (UNCLEARABLE_PENALTY if r["ceiling"] <= 0 else 0.0)


class Linker:
    """コーパス上の分布の対応で、ある指標の値を基準の尺度（DT の全 target 版 D_progress）に写す。
    x がその指標の分布の何 % 点にあたるかを求め、基準の分布の同じ % 点の値を返す"""

    def __init__(self, values, ref):
        self.src = np.sort(np.asarray(values, float))
        self.ref = np.sort(np.asarray(ref, float))
        self.q = np.linspace(0, 1, len(self.src))

    def __call__(self, x):
        return float(np.interp(np.interp(x, self.src, self.q), self.q, self.ref))


def dt_progress_over(res_entry, targets):
    """difficulty.py の結果から、指定した target だけで D_progress を計算し直す"""
    pg = [c["progress"] for c in res_entry["curve"] if c["target"] in targets]
    return 1 - float(np.mean(pg))


class Objective:
    """ステージ群をまとめて評価し、目的関数の値を返す。
    generate_by_difficulty.py（1本の探索）と generate_replicates.py（独立な探索を複数本）で共有する。

      dt     : DT だけ
      combo  : DT（高い target）+ ルールパネル
      combo3 : DT（高い target）+ ルールパネル + PPO パネル
               評価器の種類を増やすと、どの目的関数にも含まれない審判（先読みプランナー）への
               伝わり方が上がるかを確かめるため
    各成分はコーパス上の分布の対応で D* の尺度（DT の全 target 版）にそろえて平均する。
    """

    def __init__(self, args, out_dir):
        self.args = args
        self.out = out_dir
        self.ev = None
        self.rule_pool = None
        self.ppo = None
        self.dt_targets = [float(t) for t in args.dt_targets.split(",")]
        self.use_rule = args.objective in ("combo", "combo3")
        self.use_ppo = args.objective == "combo3"
        if self.use_rule or self.use_ppo:
            # コーパスで各成分を D* の尺度にそろえる写像を作る
            man = json.load(open(args.corpus_manifest, encoding="utf-8"))
            cdt = json.load(open(args.corpus_dt, encoding="utf-8"))
            ref = [cdt[m["path"]]["D_progress"] for m in man]
            self.link_dt = Linker([dt_progress_over(cdt[m["path"]], self.dt_targets) for m in man], ref)
        if self.use_rule:
            crule = json.load(open(args.corpus_rule, encoding="utf-8"))
            self.link_rule = Linker([crule[m["path"]]["D_panel_progress"] for m in man], ref)
            from rule_agent import PRESETS
            self.rule_agents = list(PRESETS)
        if self.use_ppo:
            cppo = json.load(open(args.corpus_ppo, encoding="utf-8"))
            self.link_ppo = Linker([cppo[m["path"]]["D_panel_progress"] for m in man], ref)

    def panels_used(self):
        return (["ルール"] if self.use_rule else []) + (["PPO"] if self.use_ppo else [])

    def start_pool(self):
        """パネル用のプロセス群は、DT を GPU に載せる前に作る（CUDA 初期化後の fork を避ける）。
        PPO パネルは目的関数の中では CPU で推論する（GPU を初期化させないため）"""
        a = self.args
        if self.use_rule and a.workers > 0 and self.rule_pool is None:
            import multiprocessing as mp
            self.rule_pool = mp.get_context("fork" if hasattr(os, "fork") else "spawn").Pool(a.workers)
        if self.use_ppo and self.ppo is None:
            from panel_difficulty import DEFAULT_PANEL, PanelEvaluator
            self.ppo = PanelEvaluator(DEFAULT_PANEL, device="cpu", workers=a.workers,
                                      max_steps=a.max_steps, batch_size=a.batch_size)

    def evaluator(self):
        if self.ev is None:
            from difficulty import DTDifficultyEvaluator
            self.ev = DTDifficultyEvaluator(self.args.model, workers=self.args.workers,
                                            max_steps=self.args.max_steps, batch_size=self.args.batch_size)
        return self.ev

    def evaluate(self, items):
        """items: [(ラベル, EditableLevel)] → {ラベル: {"D", "ceiling", 成分 "D_dt"/"D_rule"/"D_ppo"}}"""
        paths = {}
        for label, lvl in items:
            path = f"{self.out}/levels/{label}.json"
            lvl.save(path)
            paths[label] = path
        a = self.args
        plist = list(paths.values())
        res = self.evaluator().evaluate(plist, targets=self.dt_targets, episodes=a.episodes, seed=a.seed)
        out = {}
        if not (self.use_rule or self.use_ppo):
            for label, p in paths.items():
                out[label] = dict(D=res[p]["D_progress"], ceiling=res[p]["ceiling"])
            return out
        rres = pres = None
        if self.use_rule:
            from rule_panel import evaluate as rule_evaluate
            rres = rule_evaluate(plist, self.rule_agents, episodes=a.rule_episodes,
                                 seed=a.seed, max_steps=a.max_steps, pool=self.rule_pool)
        if self.use_ppo:
            if self.ppo is None:
                self.start_pool()
            pres = self.ppo.evaluate(plist, episodes=a.ppo_episodes, seed=a.seed)
        for label, p in paths.items():
            comp = dict(D_dt=self.link_dt(res[p]["D_progress"]))
            if rres is not None:
                comp["D_rule"] = self.link_rule(rres[p]["D_panel_progress"])
            if pres is not None:
                comp["D_ppo"] = self.link_ppo(pres[p]["D_panel_progress"])
            out[label] = dict(comp, D=float(np.mean(list(comp.values()))), ceiling=res[p]["ceiling"])
        return out

    def close(self):
        if self.ev is not None:
            self.ev.close()
        if self.rule_pool is not None:
            self.rule_pool.close()
        if self.ppo is not None:
            self.ppo.close()


class Search:
    def __init__(self, args):
        self.args = args
        self.out = args.out.rstrip("/")
        self.targets = [float(t) for t in args.targets.split(",")]
        self.state_path = f"{self.out}/state.json"
        self.obj = Objective(args, self.out)
        self.dt_targets = self.obj.dt_targets

    def evaluate(self, items):
        return self.obj.evaluate(items)

    # ---- 状態の保存・再開 ----
    def save_state(self, gen, pops, rng, history):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dict(generation=gen, targets=self.targets, history=history,
                           rng=rng.bit_generator.state,
                           pops={str(t): [dict({k: v for k, v in p.items() if k != "lvl"},
                                               level=p["lvl"].to_json())
                                          for p in pop] for t, pop in pops.items()}), f)
        os.replace(tmp, self.state_path)

    def load_state(self):
        with open(self.state_path, encoding="utf-8") as f:
            s = json.load(f)
        rng = np.random.default_rng()
        rng.bit_generator.state = s["rng"]
        pops = {float(t): [dict(p, lvl=EditableLevel.from_json(p.pop("level"))) for p in pop]
                for t, pop in s["pops"].items()}
        return s["generation"], pops, rng, s["history"]

    # ---- 本体 ----
    def run(self):
        a = self.args
        os.makedirs(f"{self.out}/levels", exist_ok=True)
        meta = dict(objective=a.objective, dt_targets=self.dt_targets, targets=self.targets,
                    rule_episodes=a.rule_episodes if a.objective == "combo" else None,
                    panels_used_in_search=self.obj.panels_used())
        meta_path = f"{self.out}/meta.json"
        if os.path.exists(meta_path):
            old = json.load(open(meta_path, encoding="utf-8"))
            if old != meta:
                raise SystemExit(f"❌ {self.out} は別の設定で探索した結果です（{old}）。--out を変えてください")
        else:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        self.obj.start_pool()
        if os.path.exists(self.state_path):
            gen, pops, rng, history = self.load_state()
            print(f"🔄 {self.state_path} から再開（第 {gen} 世代まで完了）")
        else:
            gen, history = 0, []
            rng = np.random.default_rng(a.seed + 12345)
            to_knob, (b0, b1) = knob_mapping(a.corpus_manifest, a.corpus_dt)
            print(f"つまみ→D_progress（コーパス）: D = {b0:.3f} + {b1:.3f}×つまみ")
            items, meta = [], {}
            for ti, t in enumerate(self.targets):
                knob = to_knob(t)
                print(f"  目標 D*={t:.2f} → つまみ {knob:.3f} で初期集団（ベースライン）を生成")
                for i in range(a.mu):
                    seed = 500000 + ti * 1000 + i
                    lvl = EditableLevel.from_json(generate_level(f"{knob:.4f}", 194, seed)).repair()
                    label = f"t{ti}_g000_{i:02d}"
                    items.append((label, lvl))
                    meta[label] = (t, lvl, f"knob={knob:.3f},seed={seed}")
            res = self.evaluate(items)
            pops = {t: [] for t in self.targets}
            for label, (t, lvl, origin) in meta.items():
                r = res[label]
                pops[t].append(dict(r, label=label, lvl=lvl, fit=fitness(r, t), origin=origin))
            for t in pops:
                pops[t].sort(key=lambda p: p["fit"])
            # 初期集団 = ベースライン（つまみで狙っただけのステージ）。検証で探索結果と比べる
            with open(f"{self.out}/baseline.json", "w", encoding="utf-8") as f:
                json.dump([dict(path=f"{self.out}/levels/{p['label']}.json", method="baseline",
                                target=t, D_search=p["D"], ceiling=p["ceiling"], origin=p["origin"])
                           for t, pop in pops.items() for p in pop], f, ensure_ascii=False, indent=2)
            history += self.log(0, pops)
            self.save_state(0, pops, rng, history)

        while gen < a.generations:
            gen += 1
            t0 = time.time()
            items, parent_of = [], {}
            for ti, t in enumerate(self.targets):
                for j in range(a.lam):
                    parent = pops[t][int(rng.integers(len(pops[t])))]
                    label = f"t{ti}_g{gen:03d}_{j:02d}"
                    items.append((label, parent["lvl"].mutate(rng)))
                    parent_of[label] = (t, parent["label"])
            res = self.evaluate(items)
            for label, lvl in items:
                t, plabel = parent_of[label]
                r = res[label]
                pops[t].append(dict(r, label=label, lvl=lvl, fit=fitness(r, t), origin=f"mutate({plabel})"))
            for t in pops:
                pops[t] = sorted(pops[t], key=lambda p: p["fit"])[:a.mu]
            history += self.log(gen, pops, time.time() - t0)
            self.save_state(gen, pops, rng, history)

        self.write_results(pops, history)
        self.obj.close()

    def log(self, gen, pops, elapsed=None):
        rows = []
        msg = [f"世代 {gen:3d}" + (f" ({elapsed / 60:.1f}分)" if elapsed else "")]
        for t, pop in pops.items():
            best = pop[0]
            rows.append(dict(generation=gen, target=t, best_D=best["D"], best_err=abs(best["D"] - t),
                             mean_err=float(np.mean([abs(p["D"] - t) for p in pop])),
                             best_label=best["label"]))
            names = {"D_dt": "DT", "D_rule": "ルール", "D_ppo": "PPO"}
            comp = (" [" + " / ".join(f"{names[k]} {best[k]:.3f}" for k in names if k in best) + "]"
                    if "D_dt" in best else "")
            msg.append(f"D*={t:.2f}: 最良 D={best['D']:.3f}{comp} (誤差 {abs(best['D'] - t):.3f}) "
                       f"集団平均誤差 {rows[-1]['mean_err']:.3f}")
        print("  ".join(msg), flush=True)
        return rows

    def write_results(self, pops, history):
        with open(f"{self.out}/history.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]))
            w.writeheader()
            w.writerows(history)
        final = {}
        for ti, (t, pop) in enumerate(pops.items()):
            for rank, p in enumerate(pop):
                path = f"{self.out}/final/t{ti}_rank{rank}.json"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                p["lvl"].save(path)
                final[path] = dict(target=t, rank=rank, D_search=p["D"], ceiling=p["ceiling"],
                                   origin=p["origin"], **p["lvl"].summary())
        with open(f"{self.out}/final/manifest.json", "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)
        # 検証用: ベースラインと探索結果を並べた一覧（difficulty.py 等の --levels-from に渡せる形式）
        with open(f"{self.out}/baseline.json", encoding="utf-8") as f:
            val = json.load(f)
        val += [dict(path=path, method="search", target=m["target"], D_search=m["D_search"],
                     ceiling=m["ceiling"], origin=m["origin"]) for path, m in final.items()]
        with open(f"{self.out}/validate_manifest.json", "w", encoding="utf-8") as f:
            json.dump(val, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 最終集団を {self.out}/final/ に保存（validate_generated.py で検証）")


def main():
    ap = argparse.ArgumentParser(description="指定した難易度のステージを編集探索で作る")
    ap.add_argument("--model", required=True, help="難易度の評価に使う DT")
    ap.add_argument("--targets", default="0.25,0.45,0.65", help="目標の D_progress（カンマ区切り）")
    ap.add_argument("--out", default="gen_out/v1")
    ap.add_argument("--mu", type=int, default=6, help="残す親の数")
    ap.add_argument("--lam", type=int, default=8, help="1世代あたりの子の数（目標ごと）")
    ap.add_argument("--generations", type=int, default=15)
    ap.add_argument("--episodes", type=int, default=6, help="評価1回あたりの target ごとのエピソード数")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0, help="評価のシード（検証では別の値を使う）")
    ap.add_argument("--corpus-manifest", default="corpus/v1/manifest.json")
    ap.add_argument("--corpus-dt", default="validity_out/v1/dt.json")
    ap.add_argument("--corpus-rule", default="validity_out/v1/rule_panel.json",
                    help="combo の尺度合わせに使うコーパスのルールパネル結果")
    ap.add_argument("--objective", choices=["dt", "combo", "combo3"], default="dt",
                    help="dt: DT だけ（v1） / combo: DT（高い target）とルールパネルの平均")
    ap.add_argument("--dt-targets", default=None,
                    help="DT を遊ばせる target。既定は dt なら 0,60,120,180,235、combo なら 120,180,235"
                         "（わざと下手な低い target の到達率を操作する抜け道を塞ぐ）")
    ap.add_argument("--corpus-ppo", default="validity_out/v1/panel.json",
                    help="combo3 の尺度合わせに使うコーパスの PPO パネル結果")
    ap.add_argument("--ppo-episodes", type=int, default=2,
                    help="combo3 で、PPO の各エージェントに遊ばせるエピソード数")
    ap.add_argument("--rule-episodes", type=int, default=4,
                    help="combo で、ルールの各エージェントに遊ばせるエピソード数")
    args = ap.parse_args()
    if args.dt_targets is None:
        args.dt_targets = "0,60,120,180,235" if args.objective == "dt" else "120,180,235"
    Search(args).run()


if __name__ == "__main__":
    main()
