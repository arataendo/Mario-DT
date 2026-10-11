"""
先読みプランナー（planner_agent.py）にプレイさせ、DT 学習用のトラジェクトリを集める。

背景:
  今の DT（教師 = PPO、学習ステージ 11 個）は、学習に使ったステージでは温度を下げると
  上限が上がるのに、未知のステージではほとんど上がらなかった（temp_out/report.txt）。
  上限の低さは行動選択のランダムさではなく、知らないステージで動けないこと（汎化）が原因と考えられる。
  教師の PPO 自身も、評価コーパスの難しい帯では 48% しかクリアできない。
  そこで、
    - 教師を、難しい帯でも 82% クリアするプランナー（plan_h8）にする
    - 生成器で作った多数のステージで集めて、ステージの多様さを増やす
  収益の幅は従来（collect_dt_dataset.py）と同じく、同じ教師の行動を確率 ε でランダムに
  置き換えて作る（別系統の教師を混ぜると模倣対象が二峰化して崩れた、v5/v7 の教訓）。

出力は collect_dt_dataset.py と同じ形式（metadata.pkl + フレーム PNG）なので、
rebalance_dt_dataset.py と train_dt.py にそのまま渡せる。
エピソードは子プロセスごとに1本ずつ pickle で保存し、終わったものは再実行時に飛ばす
（rule_panel.run_jobs_robust を使う。子プロセスが死んだり固まったりしても作り直して続ける）。

使用例:
    python collect_planner_dataset.py --levels-from corpus/train_gen/manifest.json \
        --episodes-per-level 6 --epsilons 0,0,0,0.3,0.6,0.9 --workers 8 --out dt_dataset_planner
"""

import argparse
import os
import pickle
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from eval_common import read_levels

SKIP = 4


def run_episode(job):
    """1エピソードを集めて pickle に保存し、要約だけを返す（子プロセスで呼ばれる）"""
    from PIL import Image
    from classes.MarioGymEnv import MarioEnv
    from classes.wrappers import SkipFrame, MarioImageWrapper
    from planner_agent import PRESETS, PlannerAgent, N_ACTIONS

    # DT の推論・評価（difficulty.py, infer_dt.py）と同じ観測の作り方
    env = MarioImageWrapper(SkipFrame(MarioEnv(level=job["level"], render_mode=None,
                                               max_episode_steps=job["max_steps"] * SKIP), skip=SKIP))
    obs, info = env.reset(seed=job["env_seed"])
    agent = PlannerAgent(**PRESETS[job["agent_name"]], seed=job["sample_seed"])
    rng = np.random.default_rng(job["sample_seed"] + 1)

    frame_dir = job["frame_dir"]
    os.makedirs(frame_dir, exist_ok=True)
    image_paths, actions, rewards = [], [], []
    reason = "timeout"
    for t in range(job["max_steps"]):
        path = f"{frame_dir}/t{t:05d}.png"
        Image.fromarray(np.transpose(obs, (1, 2, 0))).save(path)
        if job["epsilon"] > 0 and rng.random() < job["epsilon"]:
            action = int(rng.integers(N_ACTIONS))   # 先読みせずにランダム行動（収益を下げる）
        else:
            action = agent.act(env)
        obs, reward, term, trunc, info = env.step(action)
        image_paths.append(path)
        actions.append(action)
        rewards.append(float(reward))
        if term or trunc:
            reason = info.get("reason", "timeout" if trunc else None) or "timeout"
            break
    env.close()

    rewards = np.array(rewards, dtype=np.float32)
    ep = dict(image_paths=image_paths, actions=np.array(actions, dtype=np.int64), rewards=rewards,
              returns_to_go=np.cumsum(rewards[::-1])[::-1].astype(np.float32),
              level=job["level"], final_reason=reason, progress=float(info.get("progress", 0.0)),
              epsilon=float(job["epsilon"]), teacher=job["agent_name"])
    tmp = job["ep_path"] + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(ep, f)
    os.replace(tmp, job["ep_path"])
    return dict(job, steps=len(actions), total_reward=float(rewards.sum()), reason=reason,
                cleared=reason == "level_complete", progress=ep["progress"])


def main():
    ap = argparse.ArgumentParser(description="プランナーのプレイから DT 学習用データを集める")
    ap.add_argument("--levels", default="")
    ap.add_argument("--levels-from", default=None, help="make_corpus.py の manifest.json")
    ap.add_argument("--teacher", default="plan_h8", help="planner_agent.PRESETS の名前")
    ap.add_argument("--episodes-per-level", type=int, default=6)
    ap.add_argument("--epsilons", default="0,0,0,0.3,0.6,0.9",
                    help="ステージごとの各エピソードのランダム行動確率（長さ = --episodes-per-level で順に使う）")
    ap.add_argument("--max-steps", type=int, default=2000,
                    help="1エピソードの最大行動回数（collect_dt_dataset.py の 8000 フレームと同じ）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="dt_dataset_planner")
    a = ap.parse_args()

    from rule_panel import run_jobs_robust
    # 子プロセス（spawn）に渡す関数は、名前で import できるモジュールのものにする。
    # __main__ の関数を渡すと、子プロセスがそれを見つけられずに異常終了する（Windows で確認）
    import collect_planner_dataset as this
    levels = read_levels(a.levels, a.levels_from)
    eps = [float(x) for x in a.epsilons.split(",")]
    out = a.out.rstrip("/")
    os.makedirs(f"{out}/episodes", exist_ok=True)
    jobs = []
    for li, lv in enumerate(levels):
        name = os.path.splitext(os.path.basename(lv))[0]
        for e in range(a.episodes_per_level):
            tag = f"{name}_e{e:02d}"
            jobs.append(dict(level=lv, agent=0, agent_name=a.teacher, episode=e,
                             epsilon=eps[e % len(eps)], env_seed=a.seed + li * 100 + e,
                             sample_seed=(a.seed + li * 100 + e) * 1000 + 17, max_steps=a.max_steps,
                             frame_dir=f"{out}/frames/{tag}", ep_path=f"{out}/episodes/{tag}.pkl"))
    print(f"📋 {len(levels)} ステージ × {a.episodes_per_level} 本 = {len(jobs)} エピソード"
          f"（教師 {a.teacher}、ε {eps}）", flush=True)

    t0 = time.time()
    res = run_jobs_robust(jobs, this.run_episode, a.workers, checkpoint=f"{out}/progress.jsonl",
                          stall_timeout=3600, progress_every=20)
    print(f"収集 {time.time() - t0:.0f} 秒", flush=True)

    episodes = []
    for j in jobs:
        with open(j["ep_path"], "rb") as f:
            episodes.append(pickle.load(f))
    with open(f"{out}/metadata.pkl", "wb") as f:
        pickle.dump(episodes, f)

    ret = np.array([r["total_reward"] for r in res])
    print(f"\n✅ {len(episodes)} エピソード、総ステップ {sum(r['steps'] for r in res)} -> {out}/metadata.pkl")
    for e in sorted(set(eps)):
        rs = [r for r in res if r["epsilon"] == e]
        print(f"   ε={e:.1f}: {len(rs):4d} 本  クリア {np.mean([r['cleared'] for r in rs]) * 100:5.1f}%"
              f"  到達 {np.mean([r['progress'] for r in rs]) * 100:5.1f}%"
              f"  収益 平均 {np.mean([r['total_reward'] for r in rs]):6.1f}")
    print(f"   収益 {ret.min():.1f} 〜 {ret.max():.1f}")


if __name__ == "__main__":
    main()
