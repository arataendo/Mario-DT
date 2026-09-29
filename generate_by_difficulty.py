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
    return abs(r["D_progress"] - target) + (UNCLEARABLE_PENALTY if r["ceiling"] <= 0 else 0.0)


class Search:
    def __init__(self, args):
        self.args = args
        self.out = args.out.rstrip("/")
        self.targets = [float(t) for t in args.targets.split(",")]
        self.state_path = f"{self.out}/state.json"
        self.ev = None

    # ---- 評価 ----
    def evaluator(self):
        if self.ev is None:
            from difficulty import DTDifficultyEvaluator
            self.ev = DTDifficultyEvaluator(self.args.model, workers=self.args.workers,
                                            max_steps=self.args.max_steps, batch_size=self.args.batch_size)
        return self.ev

    def evaluate(self, items):
        """items: [(ラベル, EditableLevel)] → {ラベル: {"D_progress", "ceiling", ...}}"""
        paths = {}
        for label, lvl in items:
            path = f"{self.out}/levels/{label}.json"
            lvl.save(path)
            paths[label] = path
        res = self.evaluator().evaluate(list(paths.values()), episodes=self.args.episodes, seed=self.args.seed)
        return {label: res[p] for label, p in paths.items()}

    # ---- 状態の保存・再開 ----
    def save_state(self, gen, pops, rng, history):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dict(generation=gen, targets=self.targets, history=history,
                           rng=rng.bit_generator.state,
                           pops={str(t): [dict(label=p["label"], D=p["D"], ceiling=p["ceiling"],
                                               fit=p["fit"], level=p["lvl"].to_json(), origin=p["origin"])
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
                pops[t].append(dict(label=label, lvl=lvl, D=r["D_progress"], ceiling=r["ceiling"],
                                    fit=fitness(r, t), origin=origin))
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
                pops[t].append(dict(label=label, lvl=lvl, D=r["D_progress"], ceiling=r["ceiling"],
                                    fit=fitness(r, t), origin=f"mutate({plabel})"))
            for t in pops:
                pops[t] = sorted(pops[t], key=lambda p: p["fit"])[:a.mu]
            history += self.log(gen, pops, time.time() - t0)
            self.save_state(gen, pops, rng, history)

        self.write_results(pops, history)
        if self.ev is not None:
            self.ev.close()

    def log(self, gen, pops, elapsed=None):
        rows = []
        msg = [f"世代 {gen:3d}" + (f" ({elapsed / 60:.1f}分)" if elapsed else "")]
        for t, pop in pops.items():
            best = pop[0]
            rows.append(dict(generation=gen, target=t, best_D=best["D"], best_err=abs(best["D"] - t),
                             mean_err=float(np.mean([abs(p["D"] - t) for p in pop])),
                             best_label=best["label"]))
            msg.append(f"D*={t:.2f}: 最良 D={best['D']:.3f} (誤差 {abs(best['D'] - t):.3f}) "
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
    Search(ap.parse_args()).run()


if __name__ == "__main__":
    main()
