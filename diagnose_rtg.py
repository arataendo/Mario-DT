"""
Decision Transformer の returns-to-go (RTG) 条件付けが実際に機能しているかを診断する。

eval_dt_matrix.py で「target_return を上げてもクリア率が上がらない（むしろ下がる）」
という非単調な結果が再現したため、そもそもモデルが RTG トークンを見て行動を
変えているのかを直接測定する。

診断内容:
  1. 行動分布の RTG 感度
     データセットから実際の (状態列, 行動履歴) ウィンドウを取り出し、
     RTG 系列「だけ」を target 値ごとに差し替えて行動分布を比較する。
     状態・行動・タイムステップは完全に固定なので、分布の差は RTG のみに由来する。
       - argmax 一致率  : target を変えても同じ行動を選ぶ割合
       - Total Variation: 行動分布の距離 0.5*Σ|p-q|
     参照スケールとして「別の状態に変えたとき」の TV も測る。
     RTG由来のTV が 状態由来のTV に比べて極端に小さければ、RTG は無視されている。

  2. 埋め込みの寄与量
     embed_rtg(rtg) のノルムを state_encoder(s) のノルムと比較する。
     RTG 埋め込みが状態埋め込みに比べて桁違いに小さければ、
     LayerNorm 後に信号が埋もれて学習が RTG を使わなくなる。

  3. 勾配感度
     d(行動logits)/d(rtg) の大きさ。0 に近ければ RTG は出力に効いていない。

  4. データセット側の RTG 分布
     そもそも収益に幅があるか（失敗エピソードに偏っていないか）を確認する。

使用方法:
    python diagnose_rtg.py --model models/mario_dt_20260822_233744_epoch30.pth \
        --data dt_dataset_v3/metadata.pkl --num-windows 200
"""

import os
import argparse
import pickle
import random

import numpy as np
import torch
from PIL import Image

from train_dt import DecisionTransformer, ACTION_VOCAB_SIZE


def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = DecisionTransformer(
        action_vocab_size=ckpt.get("action_vocab_size", ACTION_VOCAB_SIZE),
        hidden_size=ckpt["hidden_size"],
        context_len=ckpt["context_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["rtg_min"], ckpt["rtg_max"], ckpt["context_len"]


def load_window(episode, start_t, context_len, image_size=(84, 84)):
    """MarioDTDataset.__getitem__ と同じ手順で 1 ウィンドウを作る（RTG は生値のまま返す）"""
    ep_len = len(episode["actions"])
    end_t = min(start_t + context_len, ep_len)
    seq_len = end_t - start_t

    images = []
    for img_path in episode["image_paths"][start_t:end_t]:
        try:
            img = Image.open(img_path).convert("RGB")
            if img.size != image_size:
                img = img.resize(image_size, Image.BILINEAR)
            images.append((np.array(img, dtype=np.float32) / 255.0).transpose(2, 0, 1))
        except Exception:
            images.append(np.zeros((3, image_size[1], image_size[0]), dtype=np.float32))
    images = np.array(images, dtype=np.float32)

    actions = episode["actions"][start_t:end_t]
    rtg_raw = episode["returns_to_go"][start_t:end_t].astype(np.float32)
    timesteps = np.arange(start_t, end_t)

    pad = context_len - seq_len
    if pad > 0:
        images = np.concatenate([np.zeros((pad, 3, *image_size), dtype=np.float32), images], axis=0)
        actions = np.concatenate([np.zeros(pad, dtype=np.int64), actions], axis=0)
        rtg_raw = np.concatenate([np.zeros(pad, dtype=np.float32), rtg_raw], axis=0)
        timesteps = np.concatenate([np.zeros(pad, dtype=np.int64), timesteps], axis=0)
    mask = np.concatenate([np.zeros(pad, dtype=np.float32), np.ones(seq_len, dtype=np.float32)], axis=0)

    return images, actions, rtg_raw, timesteps, mask


def build_rtg_for_target(rtg_raw, mask, target, rtg_min, rtg_max):
    """ウィンドウ内の RTG の「減り方」は実データのまま、最後の値が target になるよう平行移動する。

    推論時 (infer_dt.py) は cur_rtg_raw を target から報酬分だけ減らしていくので、
    「直近の RTG 水準が target 相当」という状況を再現するのがこの形。
    """
    shifted = rtg_raw + (target - rtg_raw[-1])
    norm = (shifted - rtg_min) / (rtg_max - rtg_min + 1e-5)
    norm = norm * mask  # パディング位置は 0 のまま
    return norm.astype(np.float32)


def forward_probs(model, images, actions, rtg_norm, timesteps, mask, device):
    with torch.no_grad():
        s = torch.tensor(images, dtype=torch.float32, device=device).unsqueeze(0)
        a = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(0)
        r = torch.tensor(rtg_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
        t = torch.tensor(timesteps, dtype=torch.long, device=device).unsqueeze(0)
        m = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
        logits = model(s, a, r, t, attention_mask=m)
        return torch.softmax(logits[0, -1], dim=-1).cpu().numpy()


def tv_distance(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def main():
    parser = argparse.ArgumentParser(description="DT の RTG 条件付けが機能しているか診断する")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, default="dt_dataset_v3/metadata.pkl")
    parser.add_argument("--num-windows", type=int, default=200)
    parser.add_argument("--targets", type=str, default="0,75,150,225,300")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model, rtg_min, rtg_max, context_len = load_model(args.model, device)
    print(f"モデル: {args.model}")
    print(f"  context_len={context_len}  学習時RTG範囲=[{rtg_min:.1f}, {rtg_max:.1f}]\n")

    with open(args.data, "rb") as f:
        episodes = [ep for ep in pickle.load(f) if len(ep["actions"]) > 0]
    print(f"データセット: {args.data} ({len(episodes)} エピソード)\n")

    targets = [float(t) for t in args.targets.split(",")]

    # ------------------------------------------------------------------
    # 4. データセット側の RTG / return 分布
    # ------------------------------------------------------------------
    print("=" * 70)
    print("[4] データセットの収益分布")
    print("=" * 70)
    returns = np.array([float(ep["rewards"].sum()) for ep in episodes])
    clears = np.array([ep.get("final_reason") == "level_complete" for ep in episodes])
    print(f"  エピソード return: min={returns.min():.1f} / 中央値={np.median(returns):.1f} / "
          f"max={returns.max():.1f} / 平均={returns.mean():.1f}")
    for lo, hi in [(-100, 0), (0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 400)]:
        n = int(((returns >= lo) & (returns < hi)).sum())
        bar = "#" * int(60 * n / len(returns))
        print(f"    return [{lo:4d},{hi:4d}): {n:4d} ({n / len(returns) * 100:5.1f}%) {bar}")
    print(f"  クリア率: {clears.mean() * 100:.1f}% ({int(clears.sum())}/{len(episodes)})")
    print(f"  クリア時の return: 平均={returns[clears].mean():.1f}" if clears.any() else "  クリア無し")
    print(f"  非クリア時の return: 平均={returns[~clears].mean():.1f}\n")

    # ------------------------------------------------------------------
    # ウィンドウをサンプリング
    # ------------------------------------------------------------------
    print(f"実データから {args.num_windows} ウィンドウをサンプリング中...")
    windows = []
    for _ in range(args.num_windows):
        ep = episodes[random.randrange(len(episodes))]
        start_t = random.randrange(len(ep["actions"]))
        windows.append(load_window(ep, start_t, context_len))
    print("完了\n")

    # ------------------------------------------------------------------
    # 1. 行動分布の RTG 感度
    # ------------------------------------------------------------------
    print("=" * 70)
    print("[1] 行動分布の RTG 感度（状態・行動履歴を固定し RTG だけ変える）")
    print("=" * 70)

    # probs[w][ti] = ウィンドウ w を target ti で評価した行動分布
    probs = np.zeros((len(windows), len(targets), ACTION_VOCAB_SIZE), dtype=np.float64)
    for wi, (images, actions, rtg_raw, timesteps, mask) in enumerate(windows):
        for ti, target in enumerate(targets):
            rtg_norm = build_rtg_for_target(rtg_raw, mask, target, rtg_min, rtg_max)
            probs[wi, ti] = forward_probs(model, images, actions, rtg_norm, timesteps, mask, device)

    argmaxes = probs.argmax(axis=2)  # (W, T)

    print("\n  ● argmax 一致率（target ペアごと。100% なら RTG を完全に無視している）")
    header = "        " + " ".join(f"{t:>7.0f}" for t in targets)
    print(header)
    for i, ti in enumerate(targets):
        cells = []
        for j, tj in enumerate(targets):
            agree = float((argmaxes[:, i] == argmaxes[:, j]).mean())
            cells.append(f"{agree * 100:6.1f}%")
        print(f"  {ti:5.0f} " + " ".join(cells))

    print("\n  ● Total Variation 距離（行動分布の差。0 なら完全に同じ分布）")
    print(header)
    for i, ti in enumerate(targets):
        cells = []
        for j, tj in enumerate(targets):
            tv = float(np.mean([tv_distance(probs[w, i], probs[w, j]) for w in range(len(windows))]))
            cells.append(f"{tv:7.4f}")
        print(f"  {ti:5.0f} " + " ".join(cells))

    # 参照スケール: 状態を変えたときに行動分布がどれだけ動くか
    mid = len(targets) // 2
    ref_tvs = []
    for _ in range(len(windows)):
        a, b = random.randrange(len(windows)), random.randrange(len(windows))
        if a != b:
            ref_tvs.append(tv_distance(probs[a, mid], probs[b, mid]))
    ref_tv = float(np.mean(ref_tvs))
    rtg_tv = float(np.mean([tv_distance(probs[w, 0], probs[w, -1]) for w in range(len(windows))]))

    print(f"\n  ● 参照スケール")
    print(f"      状態を変えたときの平均TV (同一target)        : {ref_tv:.4f}")
    print(f"      RTGを {targets[0]:.0f}→{targets[-1]:.0f} に変えたときの平均TV : {rtg_tv:.4f}")
    print(f"      → RTG の影響力は状態の {rtg_tv / max(ref_tv, 1e-9) * 100:.1f}% 相当")

    argmax_flip = float((argmaxes[:, 0] != argmaxes[:, -1]).mean())
    print(f"      target {targets[0]:.0f} と {targets[-1]:.0f} で選ぶ行動が変わったウィンドウ: "
          f"{argmax_flip * 100:.1f}%\n")

    # ------------------------------------------------------------------
    # 2. 埋め込みの寄与量
    # ------------------------------------------------------------------
    print("=" * 70)
    print("[2] 埋め込みノルムの比較（RTG信号が状態信号に埋もれていないか）")
    print("=" * 70)
    with torch.no_grad():
        sample_imgs = torch.tensor(
            np.array([w[0] for w in windows[:32]]), dtype=torch.float32, device=device
        ).view(-1, 3, 84, 84)
        state_emb = model.state_encoder(sample_imgs)
        state_norm = float(state_emb.norm(dim=-1).mean())

        print(f"  state_encoder(s) の平均ノルム : {state_norm:8.4f}")
        for target in targets:
            v = (target - rtg_min) / (rtg_max - rtg_min + 1e-5)
            e = model.embed_rtg(torch.tensor([[v]], dtype=torch.float32, device=device))
            print(f"  embed_rtg(target={target:5.0f}) のノルム : {float(e.norm()):8.4f}  "
                  f"(正規化値 {v:.3f})")

        w_norm = float(model.embed_rtg.weight.norm())
        b_norm = float(model.embed_rtg.bias.norm())
        print(f"\n  embed_rtg.weight のノルム : {w_norm:8.4f}  ← RTG値に応じて変化する成分")
        print(f"  embed_rtg.bias   のノルム : {b_norm:8.4f}  ← RTG値に依存しない定数成分")
        print(f"  → RTG を 0→1 動かしたときの埋め込み変化量は {w_norm:.4f}、"
              f"状態埋め込みの {w_norm / max(state_norm, 1e-9) * 100:.1f}% 相当")

    # ------------------------------------------------------------------
    # 3. 勾配感度
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[3] 勾配感度 d(行動logits)/d(rtg)")
    print("=" * 70)
    grads = []
    for images, actions, rtg_raw, timesteps, mask in windows[:64]:
        rtg_norm = build_rtg_for_target(rtg_raw, mask, targets[mid], rtg_min, rtg_max)
        s = torch.tensor(images, dtype=torch.float32, device=device).unsqueeze(0)
        a = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(0)
        r = torch.tensor(rtg_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
        r.requires_grad_(True)
        t = torch.tensor(timesteps, dtype=torch.long, device=device).unsqueeze(0)
        m = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)

        logits = model(s, a, r, t, attention_mask=m)
        logits[0, -1].abs().sum().backward()
        grads.append(float(r.grad.abs().mean()))
    print(f"  |d(logits)/d(rtg)| の平均 : {np.mean(grads):.6f}")
    print(f"  （0 に近いほど RTG は出力に効いていない）\n")

    print("=" * 70)
    print("判定")
    print("=" * 70)
    if rtg_tv < 0.05 * ref_tv:
        print("  ❌ RTG はほぼ完全に無視されている。行動は状態だけで決まっている。")
    elif rtg_tv < 0.25 * ref_tv:
        print("  ⚠️  RTG の影響は残っているが状態に比べて弱い。条件付けの制御性は低い。")
    else:
        print("  ✅ RTG は行動分布に十分効いている。非単調性は別要因の可能性が高い。")


if __name__ == "__main__":
    main()
