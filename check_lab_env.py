"""
研究室PC（Linux + NVIDIA GPU）で学習を回せる状態になっているかを確認する。

長時間の学習を仕掛けてから数時間後に落ちているのが一番痛いので、
つまずきやすい所を数十秒で一通り踏んでおく。

使い方（リポジトリのルートで、venv を有効にした状態で）:
    python check_lab_env.py
"""

import os
import sys
import traceback

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

results = []


def check(name):
    def deco(fn):
        try:
            detail = fn()
            results.append((True, name, detail or ""))
            print(f"✅ {name}" + (f": {detail}" if detail else ""))
        except Exception as e:  # noqa: BLE001 — どこで落ちたかを全部並べて見せたい
            results.append((False, name, str(e)))
            print(f"❌ {name}: {e}")
            hint = HINTS.get(name)
            if hint:
                print(f"   → {hint}")
            if os.environ.get("VERBOSE"):
                traceback.print_exc()
        return fn
    return deco


HINTS = {
    "OpenCV (cv2)": "GUI の無いサーバーでは libGL が無いことが多い。"
                    "sudo apt install -y libgl1 libglib2.0-0 を入れるか、"
                    "sudo が無ければ pip uninstall -y opencv-python && pip install opencv-python-headless",
    "PyTorch と GPU": "nvidia-smi で見える CUDA バージョンに合う torch を入れ直す。例: "
                      "pip install torch --index-url https://download.pytorch.org/whl/cu121 "
                      "(ドライバが古い場合は cu118)",
    "学習データ (dt_dataset_v10)": "mario_dt_data.tar を展開していない。"
                                   "リポジトリのルートで tar -xf mario_dt_data.tar",
}


@check("PyTorch と GPU")
def _():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"torch {torch.__version__} から GPU が見えない（CPU で動いてしまう）")
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"torch {torch.__version__} / {name} ({mem:.0f} GB)"


@check("OpenCV (cv2)")
def _():
    import cv2
    return cv2.__version__


@check("transformers")
def _():
    import transformers
    return transformers.__version__


@check("ゲーム環境 (画面なしで 1 エピソード分の step)")
def _():
    from classes.wrappers import make_mario_env
    env = make_mario_env(level="Level_easy_01", max_episode_steps=400)
    obs, _ = env.reset()
    for _ in range(50):
        obs, r, term, trunc, info = env.step(8)  # 右ダッシュジャンプ
        if term or trunc:
            break
    env.close()
    return f"obs {tuple(obs.shape)}, mario_x={info.get('mario_x')}"


@check("学習データ (dt_dataset_v10)")
def _():
    from train_dt import MarioDTDataset
    ds = MarioDTDataset("dt_dataset_v10/metadata_rebalanced.pkl", virtual_len=4, frame_stack=2)
    sample = ds[0]
    return f"{ds.num_episodes} エピソード, states {tuple(sample['states'].shape)}"


@check("DT の学習 1 ステップ (GPU, frame_stack=2)")
def _():
    import torch
    from torch.utils.data import DataLoader
    from train_dt import MarioDTDataset, DecisionTransformer
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MarioDTDataset("dt_dataset_v10/metadata_rebalanced.pkl", virtual_len=64, frame_stack=2)
    # Linux では num_workers>0 が使える（Windows 機では使えなかった）。ここで動くか確かめる
    dl = DataLoader(ds, batch_size=16, num_workers=2)
    batch = next(iter(dl))
    model = DecisionTransformer(frame_stack=2).to(dev)
    logits = model(batch["states"].to(dev), batch["actions"].to(dev),
                   batch["returns_to_go"].to(dev), batch["timesteps"].to(dev),
                   attention_mask=batch["attention_mask"].to(dev))
    logits.sum().backward()
    return f"device={dev}, logits {tuple(logits.shape)}, DataLoader num_workers=2 OK"


@check("最良モデルの読み込みと推論 (--sample)")
def _():
    import torch
    from infer_dt import load_model, run_episode
    path = "models/mario_dt_20260921_175028_epoch20.pth"
    model, rmin, rmax, ctx = load_model(path, torch.device("cpu"))
    total, steps, x, reason = run_episode(model, rmin, rmax, ctx, "Level_easy_01", 200.0,
                                          60, torch.device("cpu"), render=False, sample=True)
    return f"60 ステップ走行 OK (X={x}, reason={reason})"


print()
n_ng = sum(1 for ok, *_ in results if not ok)
if n_ng == 0:
    print("🎉 すべて OK。LAB_SETUP.md の「学習を回す」に進めます。")
else:
    print(f"⚠️  {n_ng} 件の問題があります。上の → の対処をしてから再実行してください。")
sys.exit(1 if n_ng else 0)
