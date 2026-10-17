"""
教師（先読みプランナー または PPO）に多数のステージをプレイさせ、DT 学習用のトラジェクトリを並列に集める。

背景:
  今の DT（教師 = PPO、学習ステージ 11 個）は、学習に使ったステージでは温度を下げると
  上限が上がるのに、未知のステージではほとんど上がらなかった（temp_out/report.txt）。
  上限の低さは行動選択のランダムさではなく、知らないステージで動けないこと（汎化）が原因と考えられる。
  そこで、生成器で作った多数のステージで集めて、ステージの多様さを増やす。

  教師は2通り:
    - プランナー（planner_agent の PRESETS / TEACHER_PRESETS の名前）。plan_h8 は跳ぶタイミングが
      真似できず、跳ぶ教師 teach_h8 は教師自体が弱くなり、どちらも上限を上げられなかった（planner_out/, teach_out/）
    - PPO（.zip のパスをカンマ区切りで。エピソードごとに順に使う）。今の DT の教師と同じ系統を使えば、
      変わるのは「ステージの多様さ」だけになる
  収益の幅は従来（collect_dt_dataset.py）と同じく、同じ教師の行動を確率 ε でランダムに
  置き換えて作る（別系統の教師を混ぜると模倣対象が二峰化して崩れた、v5/v7 の教訓）。
  PPO の行動の選び方も collect_dt_dataset.py と同じ（確率的に選ぶ割合 0.7、停滞の安全弁あり）。

出力は collect_dt_dataset.py と同じ形式（metadata.pkl + フレーム PNG）なので、
rebalance_dt_dataset.py と train_dt.py にそのまま渡せる。
エピソードは子プロセスごとに1本ずつ pickle で保存し、終わったものは再実行時に飛ばす
（rule_panel.run_jobs_robust を使う。子プロセスが死んだり固まったりしても作り直して続ける）。

使用例:
    python collect_planner_dataset.py --levels-from corpus/train_gen/manifest.json \
        --episodes-per-level 6 --epsilons 0,0,0,0.3,0.6,0.9 --workers 8 --out dt_dataset_planner
    python collect_planner_dataset.py --levels-from corpus/train_div/manifest.json \
        --teacher models/a.zip,models/b.zip --workers 48 --out dt_dataset_div
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
_PPO_CACHE = {}   # 子プロセスごとに、読み込んだ PPO を使い回す


def _play_planner(job, save_frame):
    from classes.MarioGymEnv import MarioEnv
    from classes.wrappers import SkipFrame, MarioImageWrapper
    from planner_agent import PRESETS, TEACHER_PRESETS, PlannerAgent, N_ACTIONS

    # DT の推論・評価（difficulty.py, infer_dt.py）と同じ観測の作り方
    env = MarioImageWrapper(SkipFrame(MarioEnv(level=job["level"], render_mode=None,
                                               max_episode_steps=job["max_steps"] * SKIP), skip=SKIP))
    obs, info = env.reset(seed=job["env_seed"])
    agent = PlannerAgent(**{**PRESETS, **TEACHER_PRESETS}[job["teacher"]], seed=job["sample_seed"])
    rng = np.random.default_rng(job["sample_seed"] + 1)
    actions, rewards, reason = [], [], "timeout"
    for t in range(job["max_steps"]):
        save_frame(t, obs)
        if job["epsilon"] > 0 and rng.random() < job["epsilon"]:
            action = int(rng.integers(N_ACTIONS))   # 先読みせずにランダム行動（収益を下げる）
        else:
            action = agent.act(env)
        obs, reward, term, trunc, info = env.step(action)
        actions.append(action)
        rewards.append(float(reward))
        if term or trunc:
            reason = info.get("reason", "timeout" if trunc else None) or "timeout"
            break
    env.close()
    return actions, rewards, reason, info


def _play_ppo(job, save_frame):
    import torch
    from stable_baselines3 import PPO
    from classes.wrappers import make_mario_env, latest_frame, StallGuard

    torch.set_num_threads(1)   # 多数の子プロセスで並列に回すので、1プロセス1スレッド
    torch.manual_seed(job["sample_seed"])
    path = job["teacher"]
    if path not in _PPO_CACHE:
        _PPO_CACHE[path] = PPO.load(path, device="cpu")
    model = _PPO_CACHE[path]
    # PPO の学習時と同じ環境（フレームを重ねた観測）。DT 用には最新の1フレームだけ保存する
    env = make_mario_env(level=job["level"], max_episode_steps=job["max_steps"] * SKIP,
                         render_mode=None, random_level=False)
    obs, info = env.reset(seed=job["env_seed"])
    rng = np.random.default_rng(job["sample_seed"] + 1)
    deterministic = rng.random() >= job["stochastic_ratio"]
    guard, mario_x = StallGuard(), 0
    actions, rewards, reason = [], [], "timeout"
    for t in range(job["max_steps"]):
        save_frame(t, latest_frame(obs, channels=3))
        action, _ = model.predict(obs, deterministic=deterministic)
        if job["epsilon"] > 0 and rng.random() < job["epsilon"]:
            action = int(rng.integers(env.action_space.n))
        action = guard.choose(int(action), mario_x)
        obs, reward, term, trunc, info = env.step(action)
        mario_x = info.get("mario_x", mario_x)
        actions.append(action)
        rewards.append(float(reward))
        if term or trunc:
            reason = info.get("reason", "timeout" if trunc else None) or "timeout"
            break
    env.close()
    return actions, rewards, reason, info


def run_episode(job):
    """1エピソードを集めて pickle に保存し、要約だけを返す（子プロセスで呼ばれる）"""
    from PIL import Image

    frame_dir = job["frame_dir"]
    os.makedirs(frame_dir, exist_ok=True)
    image_paths = []

    def save_frame(t, frame):
        path = f"{frame_dir}/t{t:05d}.png"
        Image.fromarray(np.transpose(frame, (1, 2, 0))).save(path)
        image_paths.append(path)

    play = _play_ppo if job["teacher"].endswith(".zip") else _play_planner
    actions, rewards, reason, info = play(job, save_frame)

    rewards = np.array(rewards, dtype=np.float32)
    ep = dict(image_paths=image_paths, actions=np.array(actions, dtype=np.int64), rewards=rewards,
              returns_to_go=np.cumsum(rewards[::-1])[::-1].astype(np.float32),
              level=job["level"], final_reason=reason, progress=float(info.get("progress", 0.0)),
              epsilon=float(job["epsilon"]), teacher=os.path.basename(job["teacher"]))
    tmp = job["ep_path"] + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(ep, f)
    os.replace(tmp, job["ep_path"])
    return dict(job, steps=len(actions), total_reward=float(rewards.sum()), reason=reason,
                cleared=reason == "level_complete", progress=ep["progress"])


def main():
    ap = argparse.ArgumentParser(description="教師（プランナー / PPO）のプレイから DT 学習用データを集める")
    ap.add_argument("--levels", default="")
    ap.add_argument("--levels-from", default=None, help="make_corpus.py の manifest.json")
    ap.add_argument("--teacher", default="plan_h8",
                    help="planner_agent のプリセット名、または PPO の .zip のパス（カンマ区切りで複数。エピソードごとに順に使う）")
    ap.add_argument("--episodes-per-level", type=int, default=6)
    ap.add_argument("--epsilons", default="0,0,0,0.3,0.6,0.9",
                    help="ステージごとの各エピソードのランダム行動確率（長さ = --episodes-per-level で順に使う）")
    ap.add_argument("--stochastic-ratio", type=float, default=0.7,
                    help="PPO 教師で、行動を確率的に選ぶエピソードの割合（collect_dt_dataset.py と同じ）")
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
    teachers = [t.strip() for t in a.teacher.split(",") if t.strip()]
    for t in teachers:
        if t.endswith(".zip") and not os.path.exists(t):
            raise SystemExit(f"❌ 教師のモデル {t} がありません")
    out = a.out.rstrip("/")
    os.makedirs(f"{out}/episodes", exist_ok=True)
    jobs = []
    for li, lv in enumerate(levels):
        name = os.path.splitext(os.path.basename(lv))[0]
        for e in range(a.episodes_per_level):
            tag = f"{name}_e{e:02d}"
            jobs.append(dict(level=lv, agent=0, agent_name=a.teacher, episode=e,
                             teacher=teachers[(li * a.episodes_per_level + e) % len(teachers)],
                             epsilon=eps[e % len(eps)], stochastic_ratio=a.stochastic_ratio,
                             env_seed=a.seed + li * 100 + e,
                             sample_seed=(a.seed + li * 100 + e) * 1000 + 17, max_steps=a.max_steps,
                             frame_dir=f"{out}/frames/{tag}", ep_path=f"{out}/episodes/{tag}.pkl"))
    print(f"📋 {len(levels)} ステージ × {a.episodes_per_level} 本 = {len(jobs)} エピソード"
          f"（教師 {len(teachers)} 個: {', '.join(os.path.basename(t) for t in teachers)}、ε {eps}）", flush=True)

    t0 = time.time()
    res = run_jobs_robust(jobs, this.run_episode, a.workers, checkpoint=f"{out}/progress.jsonl",
                          stall_timeout=3600, progress_every=max(20, len(jobs) // 50))
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
