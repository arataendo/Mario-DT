"""
ルールベースのエージェント群（rule_agent.py）でステージの難易度を測る。

PPO パネル (panel_difficulty.py) と同じ形式の JSON を出力するので、
analyze_validity.py にそのまま渡せる。

環境シードの規則も DT・PPO パネルと同じ（エピソード e は env_seed=seed+e）で、
敵の初期の向きがステージ・エージェント間で共通になる。

使用例:
    python rule_panel.py --levels-from corpus/v1/manifest.json --episodes 8 --workers 8 \
        --output validity_out/v1/rule_panel.json
    python rule_panel.py --calibrate          # 学習用ステージで各プリセットの腕前を確かめる
"""

import argparse
import json
import multiprocessing as mp
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from eval_common import read_levels, summarize_panel
from rule_agent import PRESETS, run_episode

# 腕前の確認用。評価に使うコーパスとは別の、学習用ステージで調整する（評価データで調整しない）
CALIB_LEVELS = ["Level_easy_01", "Level_easy_02", "Level_medium_01", "Level_medium_02",
                "Level_hard_01", "Level_hard_02", "Level_hard_03"]


def _job_key(j):
    return f"{j['level']}|{j['agent']}|{j['episode']}"


def run_jobs_robust(jobs, run_fn, workers, checkpoint=None, stall_timeout=1800,
                    max_restarts=5, progress_every=50, start_method=None, quiet=False):
    """長時間かかる評価を、止まらないように回す。

    multiprocessing.Pool.map は、子プロセスが（共有サーバーのメモリ逼迫などで）OOM で殺されると
    そのジョブの結果を待ったまま永久に止まる。研究室PCで、先読みプランナーの評価が
    1日以上止まったのはこれだと考えられる。ここでは:
      - ProcessPoolExecutor を使う（子プロセスが死ぬと BrokenProcessPool で検知できる）
      - stall_timeout 秒どのジョブも終わらなければ、固まったとみなす
      - いずれの場合もプロセス群を作り直し、残りのジョブだけを続ける（最大 max_restarts 回）
      - 終わったジョブは checkpoint (JSON Lines) に1件ずつ追記し、再実行時は飛ばす
      - progress_every 件ごとに進み具合を表示する
    start_method: 呼び出し側が既に CUDA を初期化している場合（探索中など）は "spawn" を渡す。
                  fork だと CUDA 初期化後の親を複製してしまい、子が固まることがある
    """
    from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutTimeout, as_completed
    from concurrent.futures.process import BrokenProcessPool

    done = {}
    if checkpoint and os.path.exists(checkpoint):
        with open(checkpoint, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[_job_key(r)] = r
        print(f"🔄 {checkpoint} から {len(done)} 件を読み込み、続きから評価します", flush=True)
    todo = [j for j in jobs if _job_key(j) not in done]
    ckf = open(checkpoint, "a", encoding="utf-8") if checkpoint else None
    t0, n0 = time.time(), len(done)
    ctx = mp.get_context(start_method or ("fork" if hasattr(os, "fork") else "spawn"))
    restarts = 0
    try:
        while todo:
            ex = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
            futs = {ex.submit(run_fn, j): j for j in todo}
            try:
                for fut in as_completed(futs, timeout=stall_timeout):
                    r = fut.result()
                    done[_job_key(r)] = r
                    if ckf:
                        ckf.write(json.dumps(r, ensure_ascii=False) + "\n")
                        ckf.flush()
                    n = len(done) - n0
                    if not quiet and n % progress_every == 0:
                        el = time.time() - t0
                        rest = (len(jobs) - len(done)) * el / max(n, 1)
                        print(f"  {len(done)}/{len(jobs)} 件  経過 {el / 60:.0f}分  残り約 {rest / 60:.0f}分", flush=True)
                ex.shutdown(wait=True)
                break
            except (BrokenProcessPool, FutTimeout) as e:
                why = "子プロセスが異常終了" if isinstance(e, BrokenProcessPool) else f"{stall_timeout}秒どのジョブも終わらない"
                for p in list(getattr(ex, "_processes", {}).values()):   # 固まった子プロセスを確実に止める
                    p.terminate()
                ex.shutdown(wait=False, cancel_futures=True)
                todo = [j for j in jobs if _job_key(j) not in done]
                restarts += 1
                print(f"⚠️  {why}。プロセス群を作り直して残り {len(todo)} 件を続けます"
                      f"（{restarts}/{max_restarts} 回目）", flush=True)
                if restarts > max_restarts:
                    raise RuntimeError(f"作り直しが {max_restarts} 回を超えました。メモリ (free -g) を確認してください")
    finally:
        if ckf:
            ckf.close()
    return [done[_job_key(j)] for j in jobs]


def evaluate(levels, agent_names, episodes=8, seed=0, workers=0, max_steps=500, pool=None,
             run_fn=run_episode, seed_offset=700, checkpoint=None, start_method=None, quiet=False):
    """pool を渡すとそれを使う（探索のように何度も呼ぶ場合、プロセス群を作り直さずに済む。
    また、呼び出し側で CUDA を初期化する前に作っておけば、CUDA 初期化後の fork を避けられる）。
    pool を渡さず workers>0 なら、止まらない回し方 (run_jobs_robust) を使う"""
    jobs = [dict(level=lv, agent=a, agent_name=name, episode=e, env_seed=seed + e,
                 sample_seed=(seed + e) * 1000 + seed_offset + a, max_steps=max_steps)
            for lv in levels for a, name in enumerate(agent_names) for e in range(episodes)]
    if pool is not None:
        results = pool.map(run_fn, jobs, chunksize=4)
    elif workers > 0:
        results = run_jobs_robust(jobs, run_fn, workers, checkpoint=checkpoint,
                                  start_method=start_method, quiet=quiet)
    else:
        results = [run_fn(j) for j in jobs]
    return summarize_panel(results, agent_names)


def main(presets=PRESETS, run_fn=run_episode, seed_offset=700,
         desc="ルールベースのエージェント群でステージの難易度を測る"):
    """planner_panel.py もこれを使う（プリセットと1エピソードの実行関数を差し替える）"""
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--levels", default="")
    ap.add_argument("--levels-from", default=None, help="make_corpus.py の manifest.json")
    ap.add_argument("--agents", default=",".join(presets), help="使うプリセット名（カンマ区切り）")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true",
                    help="学習用ステージで各プリセットの平均クリア率を確かめる（評価コーパスは使わない）")
    ap.add_argument("--output", default=None)
    ap.add_argument("--checkpoint", default=None,
                    help="終わったエピソードを1件ずつ保存するファイル。既定は --output に .partial.jsonl を付けたもの。"
                         "途中で止まっても、同じコマンドで続きから再開できる")
    args = ap.parse_args()
    if args.checkpoint is None and args.output:
        args.checkpoint = args.output + ".partial.jsonl"

    names = [s.strip() for s in args.agents.split(",") if s.strip()]
    unknown = [n for n in names if n not in presets]
    if unknown:
        raise SystemExit(f"未知のプリセット: {unknown}（選べるもの: {list(presets)}）")
    levels = CALIB_LEVELS if args.calibrate else read_levels(args.levels, args.levels_from)

    t0 = time.time()
    res = evaluate(levels, names, args.episodes, args.seed, args.workers, args.max_steps,
                   run_fn=run_fn, seed_offset=seed_offset, checkpoint=args.checkpoint)
    el = time.time() - t0
    print(f"{len(levels) * len(names) * args.episodes} エピソードを {el:.0f} 秒で評価\n")

    print(f"{'stage':24s} " + " ".join(f"{n:>8s}" for n in names))
    for lv in levels:
        print(f"{os.path.basename(lv):24s} "
              + " ".join(f"{a['clear_rate'] * 100:7.0f}%" for a in res[lv]["agents"]))
    print(f"{'平均クリア率':22s} " + " ".join(
        f"{np.mean([res[lv]['agents'][i]['clear_rate'] for lv in levels]) * 100:7.0f}%"
        for i in range(len(names))))
    print(f"{'平均到達率':23s} " + " ".join(
        f"{np.mean([res[lv]['agents'][i]['progress'] for lv in levels]) * 100:7.0f}%"
        for i in range(len(names))))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\n💾 {args.output}")
        if args.checkpoint and os.path.exists(args.checkpoint):
            os.remove(args.checkpoint)   # 完走したので途中保存は不要（git add に混ざらないように消す）


if __name__ == "__main__":
    main()
