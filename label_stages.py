"""
配置から難易度を予測するモデル（difficulty_model.py）の学習用に、多数のステージを作って DT で難易度を付ける。

既存のラベル付きステージ（評価コーパスと生成実験、261 個）は、学習には少ない。ここで数千個に増やす。
  - 半分は生成器のまま（つまみ 0.05〜0.95 を一様に）
  - 半分はそれに編集（段階3の探索と同じ 12 種類）を数回かけたもの。探索が作るような
    「生成器だけでは出てこない配置」にも予測が効くようにする
  - 1ステージあたり 5 target × episodes 本（既定 5 本。評価コーパスの 10 本より粗いが、数で補う）

結果は 1 ステージ 1 行の JSON Lines（label_out/<name>/labels.jsonl）に追記していき、止まっても続きから再開する。
各行にステージの配置そのもの（コンパクトな形）も入れるので、このファイルだけで自宅 PC でも学習できる。

使用例（研究室PC）:
    python label_stages.py --model models/mario_dt_20260924_111016_epoch20.pth --count 3000 --workers 8
"""

import argparse
import json
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from generate_level import generate_level
from level_editor import EditableLevel

SEED_BASE = 900000   # 評価コーパス 300000〜、生成実験 600000〜、学習用ステージ 800000〜 と重ならない


def to_compact(lv):
    return dict(length=lv.length, holes=[x for x in range(lv.length) if not lv.ground[x]],
                pipes={str(x): h for x, h in lv.pipes.items()}, goombas=lv.goombas, koopas=lv.koopas,
                extras=lv.extras)


def from_compact(c):
    holes = set(c["holes"])
    return EditableLevel(c["length"], [x not in holes for x in range(c["length"])],
                         {int(x): h for x, h in c["pipes"].items()}, c["goombas"], c["koopas"], c["extras"])


def make_stage(i, mutated_every=2):
    """i 番目のステージ（シードから完全に決まる）。i が mutated_every の倍数でなければ編集をかける"""
    seed = SEED_BASE + i
    rng = np.random.default_rng(seed)
    knob = float(rng.uniform(0.05, 0.95))
    lv = EditableLevel.from_json(generate_level(f"{knob:.4f}", 194, seed)).repair()
    n_edits = 0
    if i % mutated_every != 0:
        n_edits = int(rng.integers(1, 6))
        for _ in range(n_edits):
            lv = lv.mutate(rng)
    return lv, dict(id=i, seed=seed, knob=round(knob, 4), n_edits=n_edits)


def main():
    ap = argparse.ArgumentParser(description="多数のステージを作り、DT で難易度を付ける")
    ap.add_argument("--model", required=True)
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--targets", default="0,60,120,180,235")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=100, help="何ステージずつ評価・保存するか")
    ap.add_argument("--seed", type=int, default=0, help="評価のシード（環境・行動の乱数）")
    ap.add_argument("--name", default="v1")
    a = ap.parse_args()

    out_dir, lvl_dir = f"label_out/{a.name}", f"corpus/label_{a.name}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(lvl_dir, exist_ok=True)
    path = f"{out_dir}/labels.jsonl"
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
        print(f"🔄 {path} から {len(done)} 個を読み込み、続きから付けます", flush=True)
    todo = [i for i in range(a.count) if i not in done]
    if not todo:
        print("✅ すべて付け終わっています")
        return

    from difficulty import DTDifficultyEvaluator
    targets = [float(t) for t in a.targets.split(",")]
    ev = DTDifficultyEvaluator(a.model, workers=a.workers)
    t0, n0 = time.time(), 0
    try:
        with open(path, "a", encoding="utf-8") as fo:
            for s in range(0, len(todo), a.chunk):
                ids = todo[s:s + a.chunk]
                stages = {}
                for i in ids:
                    lv, rec = make_stage(i)
                    p = f"{lvl_dir}/s{i:05d}.json"
                    lv.save(p)
                    stages[p] = (lv, rec)
                res = ev.evaluate(list(stages), targets, a.episodes, seed=a.seed)
                for p, (lv, rec) in stages.items():
                    r = res[p]
                    fo.write(json.dumps(dict(rec, path=p, D_progress=r["D_progress"], D_clear=r["D_clear"],
                                             ceiling=r["ceiling"], episodes=a.episodes,
                                             progress=[c["progress"] for c in r["curve"]],
                                             clear=[c["clear_rate"] for c in r["curve"]],
                                             level=to_compact(lv)), ensure_ascii=False) + "\n")
                fo.flush()
                n0 += len(ids)
                el = time.time() - t0
                print(f"  {len(done) + n0}/{a.count} 個  経過 {el / 60:.0f}分  残り約 {(len(todo) - n0) * el / n0 / 60:.0f}分",
                      flush=True)
    finally:
        ev.close()
    print(f"✅ {path}")


if __name__ == "__main__":
    main()
