"""
Decision Transformer の学習スクリプト。

collect_dt_dataset.py で収集した (state画像・action・returns_to_go) の
トラジェクトリから、GPT2 バックボーンの Decision Transformer を学習する。
アーキテクチャは Nature CNN の状態エンコーダ + GPT2Model で、
(RTG, State, Action) を1タイムステップとして交互に並べた系列を入力する
標準的な Decision Transformer 構成。

使用方法:
    python train_dt.py --dataset dt_dataset/metadata.pkl --epochs 20
"""

import os
import argparse
import pickle
import random
import re
from datetime import datetime

# 注意: この環境では DataLoader(num_workers>0) が Windows の spawn 起動時に
# 無関係なスクリプトを "__main__" として子プロセスに渡してしまい
# (AttributeError: Can't get attribute 'MarioDTDataset' on <module '__main__' from
#  'C:\\AI\\smfix\\sm_dummy_main.py'>)、確実にクラッシュする問題を確認済み。
# multiprocessing.set_executable(sys.executable) を試したが解消しなかった
# (VSCodeのPython環境検出ツールなど、こちら側のコードで制御できない外部要因の疑い)。
# そのため Windows 機では --num-workers は 0 固定で運用すること。
# Linux (研究室PC) ではこの問題は起きない（fork で起動するため）。PNG のデコードが
# 律速になるので、むしろ --num-workers 8 程度にすると GPU を遊ばせずに済む。

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import GPT2Config, GPT2Model

from classes.AgentInput import AgentInput

# アクション数は環境側 (AgentInput) と常に一致させる。ここで固定値を
# 持つと、行動空間を変更したときに気付かずズレたまま学習してしまう。
ACTION_VOCAB_SIZE = AgentInput.N_ACTIONS


class MarioDTDataset(Dataset):
    def __init__(self, pkl_path, context_len=30, image_size=(84, 84), virtual_len=50000,
                 frame_stack=1):
        print(f"Loading metadata from {pkl_path}...")
        with open(pkl_path, "rb") as f:
            self.episodes = pickle.load(f)

        self.context_len = context_len
        self.image_size = image_size
        self.virtual_len = virtual_len
        # frame_stack: 1ステップの状態として直近 k フレームを重ねる。
        #   DT は単一フレーム入力だと速度が分からず、1マス穴のジャンプ踏切りを外して落ちる
        #   （教師の PPO は 4フレームスタックなので同じ穴で落ちない）。
        #   収集時に毎ステップ「その時点の最新フレーム」を保存しているので、
        #   同一エピソード内の連続する image_paths がそのまま過去フレームになる。再収集は不要。
        self.frame_stack = frame_stack

        # Windows で収集した metadata は "dt_dataset_v8/frames\ep00000.png" のように
        # バックスラッシュを含む。Linux ではこれが区切りと解釈されず画像が読めないため、
        # "/" に揃える（"/" は Windows でも区切りとして通る）。
        for ep in self.episodes:
            ep["image_paths"] = [p.replace("\\", "/") for p in ep["image_paths"]]

        self.lengths = [len(ep["actions"]) for ep in self.episodes]
        self.episodes = [ep for ep, length in zip(self.episodes, self.lengths) if length > 0]
        self.lengths = [len(ep["actions"]) for ep in self.episodes]
        self.num_episodes = len(self.episodes)
        if self.num_episodes == 0:
            raise ValueError("空でないエピソードがデータセットにありません")

        all_rtg = np.concatenate([ep["returns_to_go"] for ep in self.episodes])
        self.rtg_max, self.rtg_min = float(np.max(all_rtg)), float(np.min(all_rtg))
        print(f"Dataset Loaded: {self.num_episodes} episodes. "
              f"RTG range: {self.rtg_min:.2f} to {self.rtg_max:.2f}")

    def __len__(self):
        return self.virtual_len

    def __getitem__(self, idx):
        ep_idx = random.randint(0, self.num_episodes - 1)
        episode = self.episodes[ep_idx]
        ep_len = self.lengths[ep_idx]

        start_t = random.randint(0, ep_len - 1)
        end_t = min(start_t + self.context_len, ep_len)
        seq_len = end_t - start_t

        # ウィンドウ内の各tに対し [t-k+1 .. t] を重ねる。必要な画像は
        # start_t-(k-1) 〜 end_t-1 の連続領域だけなので、まとめて1回読み込んで
        # スライスで組み立てる（t ごとに k 枚読むと k 倍遅くなる）。
        k = self.frame_stack
        load_from = max(0, start_t - (k - 1))
        image_paths = episode["image_paths"][load_from:end_t]
        actions = episode["actions"][start_t:end_t]
        rtg = episode["returns_to_go"][start_t:end_t]
        rtg = (rtg - self.rtg_min) / (self.rtg_max - self.rtg_min + 1e-5)
        timesteps = np.arange(start_t, end_t)

        images = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                if img.size != self.image_size:
                    img = img.resize(self.image_size, Image.BILINEAR)
                img_arr = np.array(img, dtype=np.float32) / 255.0
                images.append(img_arr.transpose(2, 0, 1))  # HWC -> CHW
            except Exception:
                images.append(np.zeros((3, self.image_size[1], self.image_size[0]), dtype=np.float32))
        images = np.array(images, dtype=np.float32)

        if k > 1:
            # エピソード先頭では過去フレームが無いので先頭フレームを複製して埋める
            head_pad = (k - 1) - (start_t - load_from)
            if head_pad > 0:
                images = np.concatenate([np.repeat(images[:1], head_pad, axis=0), images], axis=0)
            # images[i : i+k] が時刻 start_t+i の状態になる
            images = np.stack(
                [np.concatenate(images[i:i + k], axis=0) for i in range(seq_len)], axis=0
            )

        pad_len = self.context_len - seq_len
        images = np.concatenate(
            [np.zeros((pad_len, 3 * k, *self.image_size), dtype=np.float32), images], axis=0
        )
        actions = np.concatenate([np.zeros(pad_len, dtype=np.int64), actions], axis=0)
        rtg = np.concatenate([np.zeros(pad_len, dtype=np.float32), rtg], axis=0)
        timesteps = np.concatenate([np.zeros(pad_len, dtype=np.int64), timesteps], axis=0)
        attention_mask = np.concatenate(
            [np.zeros(pad_len, dtype=np.float32), np.ones(seq_len, dtype=np.float32)], axis=0
        )

        return {
            "states": torch.tensor(images),
            "actions": torch.tensor(actions, dtype=torch.long),
            "returns_to_go": torch.tensor(rtg, dtype=torch.float32).unsqueeze(-1),
            "timesteps": torch.tensor(timesteps, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
        }


class DecisionTransformer(nn.Module):
    def __init__(self, action_vocab_size=ACTION_VOCAB_SIZE, hidden_size=128,
                 context_len=30, max_ep_len=2000, n_layer=3, n_head=4, frame_stack=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_len = context_len
        self.frame_stack = frame_stack
        in_ch = 3 * frame_stack

        config = GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=4 * hidden_size,
            activation_function="relu",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )
        self.transformer = GPT2Model(config)

        # Nature CNN: (B, 3, 84, 84) -> (B, hidden_size)
        self.state_encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, hidden_size),
            nn.Tanh(),
        )

        self.embed_rtg = nn.Linear(1, hidden_size)
        self.embed_action = nn.Embedding(action_vocab_size, hidden_size)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_action = nn.Sequential(nn.Linear(hidden_size, action_vocab_size))

    def encode_states(self, states):
        """(B, T, 3k, 84, 84) の画像列を (B, T, hidden) の埋め込みにする。

        CNN はフレームごとに独立に掛かるので、この結果は時刻ごとにキャッシュできる。
        推論で毎ステップ過去 context_len 枚すべてを通し直すと、計算の大半がここで
        無駄になる（新しいのは最新の1枚だけ）。difficulty.py はこれを使って高速化している。
        """
        batch_size, seq_length = states.shape[0], states.shape[1]
        flat = states.reshape(-1, 3 * self.frame_stack, 84, 84)
        return self.state_encoder(flat).view(batch_size, seq_length, self.hidden_size)

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None,
                state_embeddings=None):
        # state_embeddings を渡した場合は states を使わない（encode_states 済みのキャッシュ）
        if state_embeddings is None:
            state_embeddings = self.encode_states(states)
        batch_size, seq_length = state_embeddings.shape[0], state_embeddings.shape[1]

        timesteps = timesteps.clamp(max=self.embed_timestep.num_embeddings - 1)
        action_embeddings = self.embed_action(actions)
        rtg_embeddings = self.embed_rtg(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        rtg_embeddings = rtg_embeddings + time_embeddings

        stacked_inputs = torch.stack((rtg_embeddings, state_embeddings, action_embeddings), dim=2)
        inputs_embeds = stacked_inputs.view(batch_size, seq_length * 3, self.hidden_size)
        inputs_embeds = self.embed_ln(inputs_embeds)

        if attention_mask is not None:
            stacked_mask = torch.stack((attention_mask, attention_mask, attention_mask), dim=2)
            stacked_mask = stacked_mask.view(batch_size, seq_length * 3)
        else:
            stacked_mask = None

        transformer_outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=stacked_mask)
        x = transformer_outputs["last_hidden_state"]

        # RTG_0, State_0, Action_0, RTG_1, State_1, Action_1, ... の並びなので
        # State の位置 (index 1, 4, 7, ...) の出力から次の Action を予測する
        state_pred_idx = torch.arange(1, seq_length * 3, 3, device=x.device)
        state_representations = x[:, state_pred_idx, :]

        return self.predict_action(state_representations)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.resume_from:
        # 再開時はチェックポイント側の frame_stack を正とする。
        # --frame-stack の指定を忘れると conv 層の形状が合わずに落ちるため。
        ck_fs = torch.load(args.resume_from, map_location="cpu", weights_only=False).get("frame_stack", 1)
        if ck_fs != args.frame_stack:
            print(f"ℹ️  チェックポイントの frame_stack={ck_fs} に合わせます (指定値 {args.frame_stack})")
            args.frame_stack = ck_fs
    dataset = MarioDTDataset(args.dataset, context_len=args.context_len,
                             frame_stack=args.frame_stack)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )

    model = DecisionTransformer(
        action_vocab_size=ACTION_VOCAB_SIZE,
        frame_stack=args.frame_stack,
        hidden_size=args.hidden_size,
        context_len=args.context_len,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(args.model_dir, exist_ok=True)

    start_epoch = 0
    if args.resume_from:
        print(f"🔄 チェックポイントから再開します: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            print("   ⚠️ 保存されたチェックポイントに optimizer 状態が無いため、"
                  "optimizer は初期状態から再開します（学習自体は継続可能）")
        # ファイル名の "_epochN" から再開位置を推定し、エポック番号を継続する
        base = os.path.basename(args.resume_from).replace(".pth", "")
        match = re.search(r"^(.*)_epoch(\d+)$", base)
        if match:
            run_name = match.group(1)
            start_epoch = int(match.group(2))
        else:
            run_name = f"{base}_resumed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"   run_name={run_name}, epoch {start_epoch} から再開")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"mario_dt_{timestamp}"

    model.train()
    for epoch in range(start_epoch, start_epoch + args.epochs):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            if step >= args.steps_per_epoch:
                break

            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rtg = batch["returns_to_go"].to(device)
            timesteps = batch["timesteps"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            action_logits = model(states, actions, rtg, timesteps, attention_mask=attention_mask)

            logits_flat = action_logits.view(-1, ACTION_VOCAB_SIZE)
            actions_flat = actions.view(-1)
            mask_flat = attention_mask.view(-1)

            loss = F.cross_entropy(logits_flat, actions_flat, reduction="none")
            loss = (loss * mask_flat).sum() / mask_flat.sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()

            total_loss += loss.item()
            if step % 100 == 0:
                print(f"Epoch {epoch+1}/{start_epoch + args.epochs} | Step {step} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / args.steps_per_epoch
        print(f"=== Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f} ===")

        ckpt_path = os.path.join(args.model_dir, f"{run_name}_epoch{epoch+1}.pth")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rtg_min": dataset.rtg_min,
            "rtg_max": dataset.rtg_max,
            "hidden_size": args.hidden_size,
            "context_len": args.context_len,
            "action_vocab_size": ACTION_VOCAB_SIZE,
            "frame_stack": args.frame_stack,
        }, ckpt_path)
        print(f"💾 保存: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Mario Decision Transformer の学習")
    parser.add_argument("--dataset", type=str, default="./dt_dataset/metadata.pkl")
    parser.add_argument("--model-dir", type=str, default="./models")
    parser.add_argument("--context-len", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=1,
                         help="状態として重ねる直近フレーム数。1なら従来通り。"
                              "2以上にすると速度情報が入り、1マス穴の踏切りを外しにくくなる")
    parser.add_argument("--resume-from", type=str, default=None,
                         help="このチェックポイント(.pth)からモデル/optimizer状態を読み込み、"
                              "ファイル名の epoch 番号から続けて学習する")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
