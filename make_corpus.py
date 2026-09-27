"""
難易度関数の妥当性検証に使うステージ群（コーパス）を作る。

generate_level.py の難易度つまみ (0〜1) を層別に均等に振って N 個生成し、
各ステージの構造特徴（敵・穴・パイプの数など）と一緒に manifest.json に記録する。

- 出力先は levels/ の外（corpus/<名前>/）。levels/ に置くと PPO 学習の抽選に混ざる
  （Level.loadLevel は .json のパスを直接読めるので、評価はパスで渡せばよい）
- シードは既存ステージ（90210〜90218 など）と重ならない範囲から採る
- 同じ引数なら同じコーパスが再現される

使用例:
    python make_corpus.py --name v1 --count 60
    # → corpus/v1/lvl_000.json ... lvl_059.json, corpus/v1/manifest.json
"""

import argparse
import json
import os

import numpy as np

from generate_level import GROUND_ROW, generate_level, validate_level


def structural_features(data):
    """ステージの素朴な構造特徴。DT の難易度と比べる「ベースライン」に使う"""
    lv = data["level"]
    obj, ent = lv["objects"], lv["entities"]
    ground_top = {x for x, y in obj["ground"] if y == GROUND_ROW}
    length = data.get("length") or (max(x for x, _ in obj["ground"]) + 1)
    gap_xs = [x for x in range(length) if x not in ground_top]
    widths, cur = [], 0
    for x in range(length):
        if x in ground_top:
            if cur:
                widths.append(cur)
            cur = 0
        else:
            cur += 1
    if cur:
        widths.append(cur)
    goombas, koopas = len(ent.get("Goomba", [])), len(ent.get("Koopa", []))
    pipes = obj.get("pipe", [])
    return dict(
        length=int(length),
        enemies=goombas + koopas, goombas=goombas, koopas=koopas,
        gap_tiles=len(gap_xs), gaps=len(widths), max_gap=max(widths, default=0),
        pipes=len(pipes), max_pipe=max((p[2] for p in pipes), default=0),
    )


def main():
    ap = argparse.ArgumentParser(description="難易度検証用のステージ群を生成する")
    ap.add_argument("--name", default="v1")
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--dmin", type=float, default=0.05)
    ap.add_argument("--dmax", type=float, default=0.95)
    ap.add_argument("--length", type=int, default=194)
    ap.add_argument("--seed-base", type=int, default=300000)
    ap.add_argument("--out-root", default="corpus")
    args = ap.parse_args()

    # manifest のパスが評価結果のキーになるので、OS によらず "/" 区切りに固定する
    out_dir = f"{args.out_root.rstrip('/')}/{args.name}"
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed_base)
    # 層別サンプリング: [dmin, dmax] を count 等分し、各区間から1つずつ一様に取る
    edges = np.linspace(args.dmin, args.dmax, args.count + 1)
    diffs = [float(rng.uniform(edges[i], edges[i + 1])) for i in range(args.count)]

    manifest = []
    for i, d in enumerate(diffs):
        seed = args.seed_base + i
        data = generate_level(f"{d:.4f}", args.length, seed)
        validate_level(data)
        path = f"{out_dir}/lvl_{i:03d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        manifest.append(dict(path=path, difficulty_param=round(d, 4), seed=seed,
                             **structural_features(data)))

    with open(f"{out_dir}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"✅ {len(manifest)} ステージを {out_dir}/ に生成（難易度つまみ {args.dmin}〜{args.dmax}）")
    for key in ("enemies", "gap_tiles", "pipes"):
        v = [m[key] for m in manifest]
        print(f"   {key:10s}: {min(v)}〜{max(v)} (平均 {np.mean(v):.1f})")


if __name__ == "__main__":
    main()
