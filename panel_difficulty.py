"""
DT とは別のエージェント群（パネル）にステージを遊ばせ、難易度の「物差し」を作る。

DT による難易度 (difficulty.py) が妥当かを確かめるには、DT とは別の基準が要る。
人間の被験者実験の代わりに、腕前の違う複数の PPO チェックポイントをパネルとし、
「パネルのうち何割がクリアできたか」をステージの難易度とみなす。

    D_panel_clear    = 1 - (全エージェント・全エピソードの平均クリア率)
    D_panel_progress = 1 - (同 平均到達率)

パネルの選び方の注意:
  最終 DT (v10 系) の教師は hard 強化ランの 6.5M〜8.0M。旧系統の 1.04M〜5.0M は教師ではないので
  既定ではこちらを使う。ただし 5.0M は hard 強化ランの出発点なので完全に独立ではない。
  2.3M 以前は Level1-1 だけで学習しているので、生成ステージではほぼ何もできない可能性が高い
  （--probe で各エージェントの腕前のばらつきを先に確かめること）。

使用例:
    python panel_difficulty.py --levels corpus/v1/lvl_000.json,... --episodes 8 --workers 8
"""

import argparse
import json
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import torch

from classes.wrappers import StallGuard
from difficulty import EnvPool, read_levels

# 旧系統（最終 DT の教師ではない）。2.57M 以降は --random-level で学習したもの
DEFAULT_PANEL = [f"models/mario_ppo_level11_checkpoint_{s}_steps.zip"
                 for s in (2570000, 3110000, 3650000, 4190000, 4730000, 5000000)]


class PanelEvaluator:
    def __init__(self, model_paths, device=None, workers=0, max_steps=500,
                 deterministic=False, batch_size=256):
        from stable_baselines3 import PPO
        # CUDA 初期化前に環境プロセスを fork しておく（difficulty.py と同じ理由）
        self.pool = EnvPool(workers, n_stack=4)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.paths = list(model_paths)
        self.models = [PPO.load(p, device=self.device) for p in self.paths]
        self.max_steps = max_steps
        self.deterministic = deterministic
        self.batch_size = batch_size

    def close(self):
        self.pool.close()

    def _run_batch(self, jobs):
        B = len(jobs)
        rngs = [np.random.default_rng(j["sample_seed"]) for j in jobs]
        guards = [StallGuard() for _ in jobs]
        mario_x = np.zeros(B, dtype=np.int64)
        results = [dict(job, cleared=False, reason="timeout", total_reward=0.0,
                        steps=0, progress=0.0) for job in jobs]
        active = np.ones(B, dtype=bool)
        latest = dict(self.pool.reset([(i, j["level"], j["env_seed"], self.max_steps)
                                       for i, j in enumerate(jobs)]))
        for t in range(self.max_steps):
            idx = np.flatnonzero(active)
            if len(idx) == 0:
                break
            chosen = {}
            # エージェントごとにまとめて方策を1回だけ計算する
            for a_i, model in enumerate(self.models):
                sub = [i for i in idx if jobs[i]["agent"] == a_i]
                if not sub:
                    continue
                obs = np.stack([latest[i] for i in sub])
                with torch.no_grad():
                    obs_t, _ = model.policy.obs_to_tensor(obs)
                    probs = model.policy.get_distribution(obs_t).distribution.probs
                    probs = probs.double().cpu().numpy()
                for n, i in enumerate(sub):
                    if self.deterministic:
                        a = int(probs[n].argmax())
                    else:
                        # エピソードごとの乱数で引く（バッチの組み方に結果を依存させない）
                        cdf = np.cumsum(probs[n])
                        a = int(min(np.searchsorted(cdf, rngs[i].random() * cdf[-1]), len(cdf) - 1))
                    chosen[i] = guards[i].choose(a, int(mario_x[i]))
            for slot, obs, reward, done, mx, reason, prog in self.pool.step(
                    [(int(i), chosen[i]) for i in idx]):
                res = results[slot]
                latest[slot] = obs
                mario_x[slot] = mx
                res["total_reward"] += reward
                res["steps"] = t + 1
                res["progress"] = prog
                if done:
                    active[slot] = False
                    res["reason"] = reason or "timeout"
                    res["cleared"] = res["reason"] == "level_complete"
        return results

    def run(self, jobs):
        out = []
        for s in range(0, len(jobs), self.batch_size):
            out += self._run_batch(jobs[s:s + self.batch_size])
        return out

    def evaluate(self, levels, episodes=8, seed=0):
        # 環境シードは DT の評価 (difficulty.py) と同じ規則。ステージ間で共通にする
        jobs = [dict(level=lv, agent=a, episode=e, env_seed=seed + e,
                     sample_seed=(seed + e) * 1000 + 500 + a)
                for lv in levels for a in range(len(self.models)) for e in range(episodes)]
        return summarize_panel(self.run(jobs), self.paths)


def summarize_panel(results, paths):
    by = {}
    for r in results:
        by.setdefault(r["level"], {}).setdefault(r["agent"], []).append(r)
    out = {}
    for lv, per_a in by.items():
        agents = []
        for a, p in enumerate(paths):
            rs = per_a.get(a, [])
            agents.append(dict(model=os.path.basename(p), n=len(rs),
                               clear_rate=float(np.mean([r["cleared"] for r in rs])),
                               progress=float(np.mean([r["progress"] for r in rs]))))
        out[lv] = dict(
            D_panel_clear=float(1 - np.mean([x["clear_rate"] for x in agents])),
            D_panel_progress=float(1 - np.mean([x["progress"] for x in agents])),
            agents=agents,
        )
    return out


def main():
    ap = argparse.ArgumentParser(description="PPO パネルでステージの難易度を測る")
    ap.add_argument("--levels", default="", help="カンマ区切り。ステージ名か .json のパス")
    ap.add_argument("--levels-from", default=None, help="make_corpus.py の manifest.json")
    ap.add_argument("--models", default=",".join(DEFAULT_PANEL))
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    paths = [p.strip() for p in args.models.split(",") if p.strip()]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"見つからないモデル: {missing}")
    levels = read_levels(args.levels, args.levels_from)
    ev = PanelEvaluator(paths, device=args.device, workers=args.workers, max_steps=args.max_steps,
                        deterministic=args.deterministic, batch_size=args.batch_size)
    t0 = time.time()
    try:
        res = ev.evaluate(levels, args.episodes, seed=args.seed)
    finally:
        ev.close()
    el = time.time() - t0
    print(f"{len(levels) * len(paths) * args.episodes} エピソードを {el:.0f} 秒で評価 "
          f"({el / len(levels):.0f} 秒/ステージ)\n")
    names = [os.path.basename(p).replace("mario_ppo_level11_checkpoint_", "").replace("_steps.zip", "")
             for p in paths]
    print(f"{'stage':24s} {'D_clear':>8s} {'D_prog':>7s}   エージェント別クリア率: " + " ".join(names))
    for lv in levels:
        r = res[lv]
        print(f"{os.path.basename(lv):24s} {r['D_panel_clear']:8.2f} {r['D_panel_progress']:7.2f}   "
              + " ".join(f"{a['clear_rate'] * 100:3.0f}%" for a in r["agents"]))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\n💾 {args.output}")


if __name__ == "__main__":
    main()
