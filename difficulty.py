"""
Decision Transformer を「腕前を指定できるプレイヤー」として使い、ステージの難易度を測る。

考え方:
  DT は目標収益 (target) で腕前を指定できる。低い target ではほとんど何もクリアせず、
  target を上げると易しいステージから順にクリアできるようになる（eval_out/v10_sample 参照）。
  そこで、いくつかの target でそれぞれ数エピソード遊ばせ、
  「腕前をいろいろ変えたときに、どれくらいクリアできたか / どこまで進めたか」を難易度とする。

    D_clear    = 1 - (全 target の平均クリア率)
    D_progress = 1 - (全 target の平均到達率)   ← 0/1 のクリアより分散が小さい
    ceiling    = 最高 target でのクリア率       ← 腕前を最大にしても届かない上限

  どの定義を最終的に使うかは、別エージェント (PPO パネル) との一致で決める（研究の論点）。

eval_dt_matrix.py との違い（ステージ探索で何百個も評価するための工夫）:
  1. 多数のエピソードを同時に進め、DT の推論をまとめて1回のバッチで行う（GPU 向け）
  2. 各フレームの CNN 出力をキャッシュする。従来は毎ステップ過去30フレーム全部を
     CNN に通し直しており、計算の大半が無駄だった（新しいのは最新の1枚だけ）
  3. 環境の実行を複数プロセスに分散できる (--workers)
  4. エピソードごとに環境シードと行動サンプリングの乱数を固定するので、
     同じステージ・同じ設定なら何度測っても同じ値になり、バッチの組み方にも依存しない

使用例:
    python difficulty.py --model models/mario_dt_20260924_111016_epoch20.pth \
        --levels Level_test_easy,Level_test_hard --episodes 10 --workers 8
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import deque

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import torch

from classes.wrappers import StallGuard
from infer_dt import load_model

SKIP = 4
DEFAULT_TARGETS = (0, 60, 120, 180, 235)


# ---------------------------------------------------------------------------
# 環境の実行（同一プロセス内 / 複数プロセス）
# ---------------------------------------------------------------------------

def _make_env(level, max_steps):
    from classes.MarioGymEnv import MarioEnv
    from classes.wrappers import SkipFrame, MarioImageWrapper
    env = MarioEnv(level=level, render_mode=None, max_episode_steps=max_steps * SKIP)
    return MarioImageWrapper(SkipFrame(env, skip=SKIP))


class _Slots:
    """スロット番号 → 環境。reset / step の結果を必要な項目だけに絞って返す"""

    def __init__(self):
        self.envs = {}

    def reset(self, items):
        out = []
        for slot, level, seed, max_steps in items:
            if slot in self.envs:
                self.envs.pop(slot).close()
            env = _make_env(level, max_steps)
            obs, _ = env.reset(seed=seed)
            self.envs[slot] = env
            out.append((slot, obs))
        return out

    def step(self, items):
        out = []
        for slot, action in items:
            obs, reward, term, trunc, info = self.envs[slot].step(action)
            done = bool(term or trunc)
            out.append((slot, obs, float(reward), done,
                        int(info.get("mario_x", 0)),
                        info.get("reason", "timeout" if trunc else None),
                        float(info.get("progress", 0.0))))
            if done:
                self.envs.pop(slot).close()
        return out

    def close(self):
        for env in self.envs.values():
            env.close()
        self.envs.clear()


def _worker(conn):
    slots = _Slots()
    while True:
        cmd, payload = conn.recv()
        if cmd == "reset":
            conn.send(slots.reset(payload))
        elif cmd == "step":
            conn.send(slots.step(payload))
        else:
            slots.close()
            conn.send(None)
            return


class EnvPool:
    """workers=0 なら同一プロセス、>0 ならスロットを workers 個のプロセスに分散して並列に進める。

    環境どうしは完全に独立している（同じプロセスで同時に動かしても結果が変わらない）ことを
    確認済みなので、どちらで動かしても同じ結果になる。
    """

    def __init__(self, workers=0):
        self.workers = workers
        if workers <= 0:
            self.local = _Slots()
            return
        ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
        self.conns, self.procs = [], []
        for _ in range(workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child,), daemon=True)
            p.start()
            self.conns.append(parent)
            self.procs.append(p)

    def _dispatch(self, cmd, items):
        if self.workers <= 0:
            return getattr(self.local, cmd)(items)
        shards = [[] for _ in range(self.workers)]
        for it in items:
            shards[it[0] % self.workers].append(it)
        busy = [w for w, sh in enumerate(shards) if sh]
        for w in busy:  # 先に全部送ってから受け取るので、各プロセスが並列に動く
            self.conns[w].send((cmd, shards[w]))
        out = []
        for w in busy:
            out += self.conns[w].recv()
        return out

    def reset(self, items):
        return self._dispatch("reset", items)

    def step(self, items):
        return self._dispatch("step", items)

    def close(self):
        if self.workers <= 0:
            self.local.close()
            return
        for c in self.conns:
            c.send(("close", None))
            c.recv()
        for p in self.procs:
            p.join(timeout=5)


# ---------------------------------------------------------------------------
# DT によるバッチ評価
# ---------------------------------------------------------------------------

class DTDifficultyEvaluator:
    def __init__(self, model_path, device=None, workers=0, max_steps=500,
                 sample=True, temperature=1.0, batch_size=256):
        # 環境用のプロセスは、モデルを GPU に載せる「前」に fork しておく。
        # CUDA を初期化した後に fork すると、子プロセスが固まることがあるため。
        self.pool = EnvPool(workers)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.rtg_min, self.rtg_max, self.ctx = load_model(model_path, self.device)
        self.k = getattr(self.model, "frame_stack", 1)
        self.max_steps = max_steps
        self.sample = sample
        self.temperature = temperature
        self.batch_size = batch_size
        with torch.no_grad():
            # 推論ループ (infer_dt.py) は足りない過去をゼロ画像で埋めるので、その埋め込みを用意しておく
            zero = torch.zeros(1, 1, 3 * self.k, 84, 84, device=self.device)
            self.pad_emb = self.model.encode_states(zero)[0, 0]

    def close(self):
        self.pool.close()

    # ---- 1バッチ分のエピソードを同時に進める ----
    def _run_batch(self, jobs):
        B, H, T = len(jobs), self.model.hidden_size, self.max_steps
        dev = self.device
        span = max(self.rtg_max - self.rtg_min, 1e-5)

        emb = torch.zeros(B, T, H, device=dev)
        act = torch.zeros(B, T, dtype=torch.long, device=dev)
        rtg = torch.zeros(B, T, device=dev)
        cur_rtg = np.array([float(j["target"]) for j in jobs])
        rngs = [np.random.default_rng(j["sample_seed"]) for j in jobs]
        guards = [StallGuard() for _ in jobs]
        mario_x = np.zeros(B, dtype=np.int64)
        recent = [deque(maxlen=self.k) for _ in jobs]
        results = [dict(job, cleared=False, reason="timeout", total_reward=0.0,
                        steps=0, progress=0.0) for job in jobs]
        active = np.ones(B, dtype=bool)

        latest = {}
        for slot, obs in self.pool.reset([(i, j["level"], j["env_seed"], T) for i, j in enumerate(jobs)]):
            latest[slot] = obs

        with torch.no_grad():
            for t in range(T):
                idx = np.flatnonzero(active)
                if len(idx) == 0:
                    break
                # 1) 最新フレームだけ CNN に通してキャッシュに積む
                for i in idx:
                    frame = latest[i].astype(np.float32) / 255.0
                    if not recent[i]:
                        for _ in range(self.k):  # 先頭は過去が無いので同じフレームで埋める
                            recent[i].append(frame)
                    else:
                        recent[i].append(frame)
                stacked = np.stack([np.concatenate(list(recent[i]), axis=0) for i in idx])
                idx_t = torch.as_tensor(idx, device=dev)
                emb[idx_t, t] = self.model.encode_states(
                    torch.as_tensor(stacked, device=dev).unsqueeze(1))[:, 0]
                rtg[idx_t, t] = torch.as_tensor((cur_rtg[idx] - self.rtg_min) / span,
                                                dtype=torch.float32, device=dev)
                act[idx_t, t] = 0  # 予測前のダミー。推論後に実際の行動で置き換える

                # 2) 直近 ctx ステップの窓を作り、まとめて推論
                lo = max(0, t - self.ctx + 1)
                w = t - lo + 1
                pad = self.ctx - w
                e, a, r = emb[idx_t, lo:t + 1], act[idx_t, lo:t + 1], rtg[idx_t, lo:t + 1]
                ts = torch.arange(lo, t + 1, device=dev)
                if pad > 0:
                    n = len(idx)
                    e = torch.cat([self.pad_emb.expand(n, pad, H), e], dim=1)
                    a = torch.cat([torch.zeros(n, pad, dtype=torch.long, device=dev), a], dim=1)
                    r = torch.cat([torch.zeros(n, pad, device=dev), r], dim=1)
                    ts = torch.cat([torch.zeros(pad, dtype=torch.long, device=dev), ts])
                mask = torch.cat([torch.zeros(pad, device=dev), torch.ones(w, device=dev)])
                logits = self.model(None, a, r.unsqueeze(-1), ts.expand(len(idx), -1),
                                    attention_mask=mask.expand(len(idx), -1),
                                    state_embeddings=e)[:, -1]

                # 3) 行動を決める（サンプリングはエピソードごとの乱数で行い、バッチの組み方に依存させない）
                if self.sample:
                    probs = torch.softmax(logits / self.temperature, dim=-1).double().cpu().numpy()
                    cdf = np.cumsum(probs, axis=1)
                    chosen = [int(min(np.searchsorted(cdf[n], rngs[i].random() * cdf[n, -1]),
                                      cdf.shape[1] - 1)) for n, i in enumerate(idx)]
                else:
                    chosen = logits.argmax(dim=-1).tolist()
                chosen = [guards[i].choose(c, int(mario_x[i])) for c, i in zip(chosen, idx)]
                act[idx_t, t] = torch.as_tensor(chosen, device=dev)

                # 4) 環境を進める
                for slot, obs, reward, done, mx, reason, prog in self.pool.step(
                        [(int(i), c) for i, c in zip(idx, chosen)]):
                    res = results[slot]
                    latest[slot] = obs
                    mario_x[slot] = mx
                    res["total_reward"] += reward
                    res["steps"] = t + 1
                    res["progress"] = prog
                    cur_rtg[slot] = float(np.clip(cur_rtg[slot] - reward, self.rtg_min, self.rtg_max))
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

    def evaluate(self, levels, targets=DEFAULT_TARGETS, episodes=10, seed=0):
        """各ステージを targets × episodes だけ遊ばせ、難易度をまとめて返す。

        シードはステージによらず共通（エピソード e は env_seed=seed+e）。
        ステージ間の差が「敵の向きの引き運」ではなくステージ自体の差になるようにするため。
        """
        jobs = [dict(level=lv, target=float(tg), episode=e,
                     env_seed=seed + e, sample_seed=(seed + e) * 1000 + ti)
                for lv in levels for ti, tg in enumerate(targets) for e in range(episodes)]
        res = self.run(jobs)
        return summarize(res, targets)


def summarize(results, targets):
    by = {}
    for r in results:
        by.setdefault(r["level"], {}).setdefault(r["target"], []).append(r)
    out = {}
    for lv, per_t in by.items():
        curve = []
        for tg in targets:
            rs = per_t.get(float(tg), [])
            curve.append(dict(
                target=float(tg), n=len(rs),
                clear_rate=float(np.mean([r["cleared"] for r in rs])) if rs else float("nan"),
                progress=float(np.mean([r["progress"] for r in rs])) if rs else float("nan"),
                total_reward=float(np.mean([r["total_reward"] for r in rs])) if rs else float("nan"),
            ))
        cr = np.array([c["clear_rate"] for c in curve])
        pg = np.array([c["progress"] for c in curve])
        out[lv] = dict(
            D_clear=float(1 - np.nanmean(cr)),
            D_progress=float(1 - np.nanmean(pg)),
            ceiling=float(cr[-1]),
            curve=curve,
        )
    return out


def main():
    ap = argparse.ArgumentParser(description="DT でステージの難易度を測る")
    ap.add_argument("--model", required=True)
    ap.add_argument("--levels", required=True,
                    help="カンマ区切り。levels/ のステージ名か .json のパス")
    ap.add_argument("--targets", default=",".join(str(t) for t in DEFAULT_TARGETS))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--workers", type=int, default=0,
                    help="環境を動かすプロセス数。0 なら同一プロセス（Windows ではこちら）")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--argmax", action="store_true", help="サンプリングせず argmax で行動する")
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", default=None, help="結果 JSON の保存先")
    args = ap.parse_args()

    ev = DTDifficultyEvaluator(args.model, device=args.device, workers=args.workers,
                               max_steps=args.max_steps, sample=not args.argmax,
                               batch_size=args.batch_size)
    levels = [s.strip() for s in args.levels.split(",") if s.strip()]
    targets = [float(t) for t in args.targets.split(",")]
    t0 = time.time()
    try:
        res = ev.evaluate(levels, targets, args.episodes, seed=args.seed)
    finally:
        ev.close()
    el = time.time() - t0
    n = len(levels) * len(targets) * args.episodes
    print(f"{n} エピソードを {el:.0f} 秒で評価 ({el / len(levels):.0f} 秒/ステージ, device={ev.device}, workers={args.workers})\n")
    print(f"{'stage':24s} {'D_clear':>8s} {'D_prog':>7s} {'上限':>5s}   クリア率 (target 順)")
    for lv in levels:
        r = res[lv]
        print(f"{os.path.basename(lv):24s} {r['D_clear']:8.2f} {r['D_progress']:7.2f} {r['ceiling'] * 100:4.0f}%   "
              + " ".join(f"{c['clear_rate'] * 100:3.0f}%" for c in r["curve"]))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\n💾 {args.output}")


if __name__ == "__main__":
    main()
