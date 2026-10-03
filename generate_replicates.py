"""
段階3 v3: 独立な探索を目標ごとに複数本回し、3つの生成方法を計算量をそろえて比べる。

v2 の反省:
  - 探索の最終集団はほぼ1つの系統に収束していた（目標ごとの6ステージの違いは4〜8タイル）。
    18ステージを独立な標本として扱ったため、信頼区間が実際より狭く出ていた
  - 第0世代の時点で、つまみで作った6個から目的関数で選んだ最良のものが既に目標にほぼ当たっていた
    （D*=0.25, 0.45 で誤差 0.006, 0.016）。改善の多くが「評価器で選んだ効果」で、
    編集の効果ではない可能性がある

そこで v3 では、目標ごとに独立な探索を R 本回し、各探索から1個ずつ出す（= 独立な標本）。
各探索は、別々のシードでつまみから作った N 個の候補を持ち、3つの方法をこの候補から作る:

  naive   : 候補の1番目（つまみで1個作るだけ。何も選ばない）
  gentest : N 個すべてを目的関数で評価し、最良を選ぶ（生成して選ぶ）
  search  : 候補の最初の μ 個から始め、λ 個 × G 世代を編集探索（評価は μ + λG = N 回）

gentest と search は評価回数が同じ N 回なので、差は「評価を新しい候補の選別に使うか、
編集に使うか」だけになる。3つが同じ候補から出発するので、探索ごとに対にして比べられる。

候補の評価と探索の各世代で状態を保存するので、途中で止まっても同じコマンドで再開できる。

使用例:
    python generate_replicates.py --model models/mario_dt_20260924_111016_epoch20.pth \
        --out gen_out/v3 --workers 8
"""

import argparse
import json
import os
import time

import numpy as np

from generate_by_difficulty import Objective, fitness, knob_mapping
from generate_level import generate_level
from level_editor import EditableLevel


class Replicates:
    def __init__(self, args):
        self.args = args
        self.out = args.out.rstrip("/")
        self.targets = [float(t) for t in args.targets.split(",")]
        self.N = args.mu + args.lam * args.generations
        self.runs = [(f"t{ti}_r{r:02d}", ti, t) for ti, t in enumerate(self.targets)
                     for r in range(args.replicates)]
        self.state_path = f"{self.out}/state.json"
        self.obj = Objective(args, self.out)

    # ---- 状態 ----
    def save(self, st):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f)
        os.replace(tmp, self.state_path)

    def load(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, encoding="utf-8") as f:
                return json.load(f)
        return dict(cands={}, pops=None, generation=0, history=[], rng=None)

    def candidate(self, ti, r, i, knob):
        seed = 600000 + ti * 10000 + r * 100 + i
        return EditableLevel.from_json(generate_level(f"{knob:.4f}", 194, seed)).repair(), seed

    # ---- 本体 ----
    def run(self):
        a = self.args
        os.makedirs(f"{self.out}/levels", exist_ok=True)
        meta = dict(design="replicates", objective=a.objective, dt_targets=self.obj.dt_targets,
                    targets=self.targets, replicates=a.replicates, mu=a.mu, lam=a.lam,
                    generations=a.generations, N=self.N, episodes=a.episodes,
                    rule_episodes=a.rule_episodes if a.objective in ("combo", "combo3") else None,
                    panels_used_in_search=self.obj.panels_used())
        if a.objective == "combo3":
            meta["ppo_episodes"] = a.ppo_episodes
        meta_path = f"{self.out}/meta.json"
        if os.path.exists(meta_path):
            if json.load(open(meta_path, encoding="utf-8")) != meta:
                raise SystemExit(f"❌ {self.out} は別の設定の結果です。--out を変えてください")
        else:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        self.obj.start_pool()
        st = self.load()
        to_knob, (b0, b1) = knob_mapping(a.corpus_manifest, a.corpus_dt)
        knobs = {ti: to_knob(t) for ti, t in enumerate(self.targets)}
        print(f"目標 {self.targets} × 独立な探索 {a.replicates} 本 = {len(self.runs)} 本。"
              f"各探索の評価回数 N = μ + λG = {a.mu} + {a.lam}×{a.generations} = {self.N}")
        print("つまみ: " + ", ".join(f"D*={t:.2f}→{knobs[ti]:.3f}" for ti, t in enumerate(self.targets)))

        # ---- 1. 候補の評価（gentest 用。search の初期集団と naive もここから取る）----
        todo = [(run_id, ti, r, i) for run_id, ti, t in self.runs
                for r in [int(run_id.split("_r")[1])] for i in range(self.N)
                if f"{run_id}_c{i:02d}" not in st["cands"]]
        if todo:
            print(f"\n[1/2] 候補の評価: 残り {len(todo)} / {len(self.runs) * self.N}")
        for s in range(0, len(todo), a.chunk):
            t0 = time.time()
            items, seeds = [], {}
            for run_id, ti, r, i in todo[s:s + a.chunk]:
                lvl, seed = self.candidate(ti, r, i, knobs[ti])
                label = f"{run_id}_c{i:02d}"
                items.append((label, lvl))
                seeds[label] = seed
            res = self.obj.evaluate(items)
            for label, _ in items:
                t = self.targets[int(label[1:label.index("_")])]
                st["cands"][label] = dict(res[label], fit=fitness(res[label], t), seed=seeds[label])
            self.save(st)
            print(f"  {min(s + a.chunk, len(todo))}/{len(todo)} 評価済み ({(time.time() - t0) / 60:.1f}分)", flush=True)

        # ---- 2. 編集探索（全探索を同じ世代で進め、子をまとめて評価する）----
        rng = np.random.default_rng(a.seed + 777)
        if st["rng"] is not None:
            rng.bit_generator.state = st["rng"]
        if st["pops"] is None:
            st["pops"] = {}
            for run_id, ti, t in self.runs:
                r = int(run_id.split("_r")[1])
                st["pops"][run_id] = sorted(
                    [dict(st["cands"][f"{run_id}_c{i:02d}"], label=f"{run_id}_c{i:02d}",
                          level=self.candidate(ti, r, i, knobs[ti])[0].to_json()) for i in range(a.mu)],
                    key=lambda p: p["fit"])
            st["generation"] = 0
            self.save(st)
        if st["generation"] < a.generations:
            print(f"\n[2/2] 編集探索: 第 {st['generation']} 世代まで完了 / {a.generations}")
        while st["generation"] < a.generations:
            g = st["generation"] + 1
            t0 = time.time()
            items, owner = [], {}
            for run_id, ti, t in self.runs:
                pop = st["pops"][run_id]
                for j in range(a.lam):
                    parent = pop[int(rng.integers(len(pop)))]
                    label = f"{run_id}_g{g:02d}_{j:02d}"
                    items.append((label, EditableLevel.from_json(parent["level"]).mutate(rng)))
                    owner[label] = (run_id, t)
            res = self.obj.evaluate(items)
            for label, lvl in items:
                run_id, t = owner[label]
                st["pops"][run_id].append(dict(res[label], fit=fitness(res[label], t), label=label,
                                               level=lvl.to_json()))
            for run_id in st["pops"]:
                st["pops"][run_id] = sorted(st["pops"][run_id], key=lambda p: p["fit"])[:a.mu]
            errs = [st["pops"][rid][0]["fit"] for rid, _, _ in self.runs]
            st["history"].append(dict(generation=g, mean_best_fit=float(np.mean(errs)),
                                      max_best_fit=float(np.max(errs))))
            st["generation"] = g
            st["rng"] = rng.bit_generator.state
            self.save(st)
            print(f"  世代 {g}: 各探索の最良の適応度 平均 {np.mean(errs):.4f} / 最悪 {np.max(errs):.4f}"
                  f" ({(time.time() - t0) / 60:.1f}分)", flush=True)

        self.write_results(st, knobs)
        self.obj.close()

    def write_results(self, st, knobs):
        os.makedirs(f"{self.out}/final", exist_ok=True)
        val = []
        for run_id, ti, t in self.runs:
            r = int(run_id.split("_r")[1])
            cands = [(i, st["cands"][f"{run_id}_c{i:02d}"]) for i in range(self.N)]
            best_i = min(cands, key=lambda c: c[1]["fit"])[0]
            chosen = {
                "naive": (self.candidate(ti, r, 0, knobs[ti])[0], cands[0][1], f"候補0 (seed={cands[0][1]['seed']})"),
                "gentest": (self.candidate(ti, r, best_i, knobs[ti])[0], cands[best_i][1],
                            f"候補{best_i} (seed={cands[best_i][1]['seed']})"),
                "search": (EditableLevel.from_json(st["pops"][run_id][0]["level"]), st["pops"][run_id][0],
                           st["pops"][run_id][0]["label"]),
            }
            for method, (lvl, ev, origin) in chosen.items():
                path = f"{self.out}/final/{run_id}_{method}.json"
                lvl.save(path)
                val.append(dict(path=path, run=run_id, target=t, method=method, D_search=ev["D"],
                                ceiling=ev["ceiling"], origin=origin, **lvl.summary()))
        with open(f"{self.out}/validate_manifest.json", "w", encoding="utf-8") as f:
            json.dump(val, f, ensure_ascii=False, indent=2)
        print(f"\n✅ {len(val)} ステージ（{len(self.runs)} 本 × 3 方法）を {self.out}/final/ に保存")


def main():
    ap = argparse.ArgumentParser(description="独立な探索を複数本回し、3つの生成方法を比べる")
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", default="0.25,0.35,0.45,0.55,0.65")
    ap.add_argument("--replicates", type=int, default=6, help="目標ごとの独立な探索の本数")
    ap.add_argument("--mu", type=int, default=4)
    ap.add_argument("--lam", type=int, default=6)
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--rule-episodes", type=int, default=4)
    ap.add_argument("--objective", choices=["dt", "combo", "combo3"], default="combo")
    ap.add_argument("--ppo-episodes", type=int, default=2, help="combo3 で PPO の各エージェントのエピソード数")
    ap.add_argument("--dt-targets", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=60, help="候補の評価を何個ずつ保存するか")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gen_out/v3")
    ap.add_argument("--corpus-manifest", default="corpus/v1/manifest.json")
    ap.add_argument("--corpus-dt", default="validity_out/v1/dt.json")
    ap.add_argument("--corpus-rule", default="validity_out/v1/rule_panel.json")
    ap.add_argument("--corpus-ppo", default="validity_out/v1/panel.json")
    args = ap.parse_args()
    if args.dt_targets is None:
        args.dt_targets = "0,60,120,180,235" if args.objective == "dt" else "120,180,235"
    Replicates(args).run()


if __name__ == "__main__":
    main()
