"""
研究室PC（Linux）へ学習を移すために、Git に載せない大容量ファイルを1本の tar にまとめる。

なぜ tar 1本にするか:
  dt_dataset_v8 / v9 のフレームは合計約37万個の小さな PNG（1個 約1.5KB）で、
  scp / rsync で個別に送ると1ファイルごとの往復オーバーヘッドで非常に遅い。
  PNG は既に圧縮済みなので gzip はかけず、無圧縮 tar で束ねるだけにしている。

metadata のパスも同時に直す:
  Windows で収集したため image_paths が "dt_dataset_v8/frames\\ep00000_t00000.png" の
  ようにバックスラッシュを含む。Linux ではこれが区切りと解釈されず、画像が1枚も読めない。
  tar に入れる metadata はすべて "/" 区切りに書き換える（元ファイルは変更しない）。

使い方（リポジトリのルートで）:
    python pack_for_transfer.py
    # → transfer/mario_dt_data.tar と transfer/mario_dt_data.tar.sha256 ができる

研究室PC側では、リポジトリのルートで
    sha256sum -c mario_dt_data.tar.sha256 && tar -xf mario_dt_data.tar
とすれば、このPCと同じ相対パスに展開される（setup_lab.sh が自動でやる）。
"""

import argparse
import hashlib
import io
import os
import pickle
import sys
import tarfile
import time

# 続きの作業（v10 系の DT 学習・評価、ε 収集の追加）に必要なもの
DEFAULT_DATASETS = ["dt_dataset_v8", "dt_dataset_v9"]
DEFAULT_METADATA = [
    "dt_dataset_v8/metadata.pkl",
    "dt_dataset_v9/metadata.pkl",
    "dt_dataset_v10/metadata.pkl",              # v8 + v9 のマージ（リバランス前）
    "dt_dataset_v10/metadata_rebalanced.pkl",   # 現行の学習データ (time_penalty=0.01)
    "dt_dataset_v10/metadata_tp010.pkl",        # 次に試す time_penalty=0.10 版
]
DEFAULT_MODELS = [
    # 現時点の最良 DT（推論は --sample 付きで使う）
    "models/mario_dt_20260921_175028_epoch20.pth",
    # frame_stack=2 の学習途中（epoch 4 で停止）。--resume-from で再開できる
    "models/mario_dt_20260924_111016_epoch4.pth",
    # 追加データ収集に使う単一系統の PPO チェックポイント群 (6.5M〜8.0M)
    *[f"models/mario_ppo_level11_checkpoint_{s}_steps.zip"
      for s in (6500000, 6800000, 7100000, 7400000, 7700000, 8000000)],
]


def normalized_metadata_bytes(path):
    """metadata を読み込み、image_paths を "/" 区切りにしてバイト列で返す"""
    with open(path, "rb") as f:
        episodes = pickle.load(f)
    n_fixed = 0
    for ep in episodes:
        new_paths = [p.replace("\\", "/") for p in ep["image_paths"]]
        if new_paths != ep["image_paths"]:
            n_fixed += 1
        ep["image_paths"] = new_paths
    return pickle.dumps(episodes), len(episodes), n_fixed


def referenced_frames(metadata_paths):
    """metadata が参照している全フレームのパス（"/" 区切り）"""
    refs = set()
    for p in metadata_paths:
        with open(p, "rb") as f:
            for ep in pickle.load(f):
                refs.update(q.replace("\\", "/") for q in ep["image_paths"])
    return refs


def main():
    parser = argparse.ArgumentParser(description="研究室PCへ送る学習データ・モデルを1本の tar にまとめる")
    parser.add_argument("--output", default="transfer/mario_dt_data.tar")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                        help="frames/ を丸ごと入れるデータセットディレクトリ（カンマ区切り）")
    parser.add_argument("--skip-verify", action="store_true",
                        help="metadata が参照する画像が全部そろっているかの事前確認を省く")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    missing = [p for p in DEFAULT_METADATA + DEFAULT_MODELS if not os.path.exists(p)]
    if missing:
        print("❌ 見つからないファイルがあります:")
        for p in missing:
            print("   ", p)
        sys.exit(1)

    # 送った先で「画像が足りない」と分かるのが一番痛いので、先に全部そろっているか確認する
    frame_files = []
    for d in datasets:
        fdir = os.path.join(d, "frames")
        names = sorted(os.listdir(fdir))
        frame_files += [f"{d}/frames/{n}" for n in names]
        print(f"📁 {d}/frames: {len(names):,} 枚")
    if not args.skip_verify:
        refs = referenced_frames(DEFAULT_METADATA)
        lacking = refs - set(frame_files)
        if lacking:
            print(f"❌ metadata が参照しているのに tar に入らない画像が {len(lacking):,} 枚あります。例:")
            for p in sorted(lacking)[:5]:
                print("   ", p)
            print("   --datasets に該当ディレクトリを追加してください")
            sys.exit(1)
        print(f"✅ metadata が参照する {len(refs):,} 枚はすべて含まれます")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    tmp = args.output + ".partial"
    t0 = time.time()
    with tarfile.open(tmp, "w") as tar:
        for p in DEFAULT_METADATA:
            data, n_eps, n_fixed = normalized_metadata_bytes(p)
            info = tarfile.TarInfo(name=p)
            info.size = len(data)
            info.mtime = int(os.path.getmtime(p))
            tar.addfile(info, io.BytesIO(data))
            print(f"📝 {p}: {n_eps} エピソード（パスを '/' に直したもの {n_fixed}）")
        for p in DEFAULT_MODELS:
            tar.add(p, arcname=p)
            print(f"🤖 {p}")
        for i, p in enumerate(frame_files):
            tar.add(p, arcname=p)
            if (i + 1) % 20000 == 0 or i + 1 == len(frame_files):
                el = time.time() - t0
                print(f"🖼️  {i + 1:,}/{len(frame_files):,} 枚 ({el / 60:.1f} 分経過)", flush=True)
    os.replace(tmp, args.output)

    # 転送後に壊れていないか確認できるよう、sha256 を sha256sum -c 形式で書き出す
    h = hashlib.sha256()
    with open(args.output, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    with open(args.output + ".sha256", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{h.hexdigest()}  {os.path.basename(args.output)}\n")

    size = os.path.getsize(args.output) / 1e9
    print("\n" + "=" * 60)
    print(f"✅ 完成: {args.output} ({size:.2f} GB, {(time.time() - t0) / 60:.1f} 分)")
    print(f"   sha256: {h.hexdigest()}")
    print("\n研究室PCへの転送例:")
    print(f"   scp {args.output} {args.output}.sha256 <ユーザー名>@<研究室PC>:~/Mario-DT/")


if __name__ == "__main__":
    main()
