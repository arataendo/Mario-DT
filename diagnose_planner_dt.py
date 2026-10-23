"""
プランナー教師の DT（run_planner_teacher.sh）がどの target でもほぼクリアできなかった原因を切り分ける。

  [1] 集めたデータ: ε ごとのクリア率・到達率・収益・長さ、行動の分布（PPO 教師のデータと比べる）
  [2] 再生の確認: 記録した行動を新しい環境で同じシードから再生し、保存した画像・結末と一致するか
      （プランナーの先読みが本体の画面や状態を汚していれば、ここでずれる）
  [3] 学習: エポックごとの損失、チェックポイントの RTG 範囲
  [4] 学習データ上の正解率: 教師の行動を DT がどれだけ当てるか（学習できているか）
  [5] 実際のプレイ: target 235 で遊ばせたときの行動の分布と結末（PPO 教師の DT と比べる）

使用例（研究室PC）:
    python diagnose_planner_dt.py > planner_out/diagnose.txt
"""

import argparse
import json
import os
import pickle
import random
import re
from collections import Counter

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import torch
from PIL import Image

# classes/AgentInput.py の対応表
ACTION_NAMES = {0: "NOP", 1: "L", 2: "R", 3: "J", 4: "L+J", 5: "R+J", 6: "Dash", 7: "R+Dash", 8: "R+Dash+J", 9: "L+Dash"}


def hist(actions):
    c = Counter(int(a) for a in actions)
    n = max(sum(c.values()), 1)
    return "  ".join(f"{a}:{c[a] / n * 100:4.1f}%" for a in sorted(c))


def section_data(eps, ref_eps):
    print("=" * 80)
    print("[1] 集めたデータ")
    print("=" * 80)
    for e in sorted({round(x.get("epsilon", 0), 2) for x in eps}):
        es = [x for x in eps if round(x.get("epsilon", 0), 2) == e]
        print(f"  ε={e:.1f}: {len(es):4d} 本  クリア {np.mean([x['final_reason'] == 'level_complete' for x in es]) * 100:5.1f}%"
              f"  到達 {np.mean([x['progress'] for x in es]) * 100:5.1f}%"
              f"  収益 {np.mean([x['rewards'].sum() for x in es]):6.1f}"
              f"  長さ {np.mean([len(x['actions']) for x in es]):6.0f}"
              f"  結末 {dict(Counter(x['final_reason'] for x in es))}")
    # 今のデータと合わせて学習した場合は、出どころ別にも見る（新しく集めたものには teacher が付いている）
    src = [("新しく集めた分", [x for x in eps if "teacher" in x]), ("今のデータ（11ステージ）", [x for x in eps if "teacher" not in x])]
    if all(g for _, g in src):
        for name, g in src:
            g0 = [x for x in g if x.get("epsilon", 0) == 0]
            clears = [x["rewards"].sum() for x in g0 if x["final_reason"] == "level_complete"]
            print(f"  {name:14s} {len(g):5d} 本  ε=0 のクリア {np.mean([x['final_reason'] == 'level_complete' for x in g0]) * 100:5.1f}%"
                  f"  クリアした回の収益 中央 {np.median(clears):6.1f}（{np.percentile(clears, 10):.1f}〜{np.percentile(clears, 90):.1f}）")
    allr = np.concatenate([x["returns_to_go"] for x in eps])
    print(f"  RTG の範囲: {allr.min():.1f} 〜 {allr.max():.1f}")
    print(f"  行動の分布（ε=0）  新しいデータ  : {hist([a for x in eps if x.get('epsilon', 0) == 0 for a in x['actions']])}")
    if ref_eps:
        print(f"                     PPO 教師      : {hist([a for x in ref_eps if x.get('epsilon', 0) == 0 for a in x['actions']])}")
    print(f"  （行動番号: {ACTION_NAMES}）")


def section_replay(eps, progress_path, n_per_eps=2):
    print("\n" + "=" * 80)
    print("[2] 再生の確認（記録した行動を新しい環境で再生）")
    print("=" * 80)
    from classes.MarioGymEnv import MarioEnv
    from classes.wrappers import SkipFrame, MarioImageWrapper
    seeds = {}
    with open(progress_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                j = json.loads(line)
                seeds[j["ep_path"].replace("\\", "/")] = j
    picked, count = [], Counter()
    for x in eps:
        e = round(x.get("epsilon", 0), 2)
        if count[e] >= n_per_eps:
            continue
        # 画像のパス .../frames/<tag>/t00000.png から job を引く
        tag = x["image_paths"][0].replace("\\", "/").split("/")[-2]
        job = next((j for j in seeds.values() if j["ep_path"].replace("\\", "/").endswith(f"/{tag}.pkl")), None)
        if job is None:
            continue
        picked.append((x, job))
        count[e] += 1
    for x, job in picked:
        env = MarioImageWrapper(SkipFrame(MarioEnv(level=job["level"], render_mode=None,
                                                   max_episode_steps=job["max_steps"] * 4), skip=4))
        obs, info = env.reset(seed=job["env_seed"])
        first_bad, maxdiff, reason = None, 0.0, "timeout"
        for t, a in enumerate(x["actions"]):
            saved = np.asarray(Image.open(x["image_paths"][t]).convert("RGB")).transpose(2, 0, 1)
            d = float(np.abs(saved.astype(np.int16) - obs.astype(np.int16)).mean())
            maxdiff = max(maxdiff, d)
            if d > 0 and first_bad is None:
                first_bad = t
            obs, r, term, trunc, info = env.step(int(a))
            if term or trunc:
                reason = info.get("reason", "timeout") or "timeout"
                break
        env.close()
        print(f"  ε={job['epsilon']:.1f} {os.path.basename(job['level']):12s} 長さ {len(x['actions']):4d}"
              f"  画像が最初にずれた時刻 {first_bad}  最大の平均画素差 {maxdiff:5.2f}"
              f"  結末 記録={x['final_reason']}/{x['progress'] * 100:.0f}%"
              f"  再生={reason}/{info.get('progress', 0) * 100:.0f}%")
    print("  （画像が一度もずれず、結末も一致すれば、データは本物のプレイを正しく記録している）")


def section_train(model_path, log_path):
    print("\n" + "=" * 80)
    print("[3] 学習")
    print("=" * 80)
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    print(f"  {model_path}: rtg_min={ck['rtg_min']:.1f} rtg_max={ck['rtg_max']:.1f} frame_stack={ck.get('frame_stack')}")
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8", errors="replace") as f:
            losses = re.findall(r"Epoch (\d+) Complete \| Average Loss: ([0-9.]+)", f.read())
        print("  エポックごとの平均損失: " + " ".join(f"{e}:{float(l):.3f}" for e, l in losses))
    else:
        print(f"  （{log_path} が無いので損失は表示しない）")


def section_accuracy(model_path, dataset_path, n=300, label=""):
    from infer_dt import load_model
    from train_dt import MarioDTDataset
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _, ctx = load_model(model_path, dev)
    ds = MarioDTDataset(dataset_path, context_len=ctx, frame_stack=model.frame_stack)
    random.seed(0)
    hit = tot = 0
    for _ in range(n):
        b = ds[0]   # __getitem__ はエピソードと開始位置をランダムに選ぶ
        with torch.no_grad():
            logits = model(b["states"][None].to(dev), b["actions"][None].to(dev), b["returns_to_go"][None].to(dev),
                           b["timesteps"][None].to(dev), attention_mask=b["attention_mask"][None].to(dev))[0]
        m = b["attention_mask"].bool()
        pred = logits.argmax(-1).cpu()[m]
        true = b["actions"][m]
        hit += int((pred == true).sum())
        tot += int(m.sum())
    print(f"  {label}: 学習データの窓 {n} 個で、教師の行動を argmax で当てた割合 {hit / max(tot, 1) * 100:.1f}%")


def section_play(models, levels, target=235.0, episodes=2):
    print("\n" + "=" * 80)
    print(f"[5] 実際のプレイ（target {target:.0f}、サンプリング、各 {episodes} 本）")
    print("=" * 80)
    from infer_dt import load_model, run_episode
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for label, path in models:
        model, rmin, rmax, ctx = load_model(path, dev)
        for lv in levels:
            for e in range(episodes):
                log = []
                total, steps, x, reason = run_episode(model, rmin, rmax, ctx, lv, target, 500, dev, False,
                                                      sample=True, seed=e, action_log=log)
                print(f"  {label:10s} {os.path.basename(lv):22s} 結末 {reason:15s} x={x:5d} 長さ {steps:3d}  行動 {hist(log)}")


def main():
    ap = argparse.ArgumentParser(description="プランナー教師の DT の診断")
    ap.add_argument("--data", default="dt_dataset_planner")
    ap.add_argument("--model", default="models/mario_dt_planner_epoch20.pth")
    ap.add_argument("--ref-model", default="models/mario_dt_20260924_111016_epoch20.pth")
    ap.add_argument("--ref-data", default="dt_dataset_v10/metadata_rebalanced.pkl")
    ap.add_argument("--train-log", default="planner_out/train.log")
    ap.add_argument("--train-manifest", default="corpus/train_gen/manifest.json")
    ap.add_argument("--label", default="プランナー教師", help="診断する DT の呼び名（表示用）")
    ap.add_argument("--play-target", type=float, default=235.0, help="[5] 実際のプレイの目標収益")
    a = ap.parse_args()

    with open(f"{a.data}/metadata_rebalanced.pkl", "rb") as f:
        eps = pickle.load(f)
    ref = None
    if os.path.exists(a.ref_data):
        with open(a.ref_data, "rb") as f:
            ref = pickle.load(f)
    section_data(eps, ref)
    if os.path.exists(f"{a.data}/progress.jsonl"):
        section_replay(eps, f"{a.data}/progress.jsonl")
    else:
        print(f"\n（{a.data}/progress.jsonl が無いので、[2] 再生の確認は飛ばす）")
    section_train(a.model, a.train_log)

    print("\n" + "=" * 80)
    print("[4] 学習データ上の正解率")
    print("=" * 80)
    section_accuracy(a.model, f"{a.data}/metadata_rebalanced.pkl", label=f"{a.label}の DT（自分の学習データ）")
    if os.path.exists(a.ref_data) and os.path.exists(a.ref_model):
        section_accuracy(a.ref_model, a.ref_data, label="PPO 教師の DT（自分の学習データ）")

    with open(a.train_manifest, encoding="utf-8") as f:
        train_lv = json.load(f)[0]["path"]
    models = [(a.label, a.model)] + ([("今の DT", a.ref_model)] if os.path.exists(a.ref_model) else [])
    section_play(models, ["Level_easy_01", train_lv, "Level_test_medium", "Level_test_hard"], target=a.play_target)


if __name__ == "__main__":
    main()
