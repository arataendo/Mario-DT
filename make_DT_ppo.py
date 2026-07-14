import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

try:
    from make_DT import DecisionTransformer
except ImportError:  # pragma: no cover
    DecisionTransformer = None


class PPORolloutDataset(Dataset):
    """PPO 推論ログから Decision Transformer 用のサンプルを生成するデータセット。"""

    def __init__(self, episodes: List[Dict[str, Any]], context_len: int = 30, image_size: Tuple[int, int] = (84, 84), gamma: float = 0.99):
        self.episodes = episodes
        self.context_len = context_len
        self.image_size = image_size
        self.gamma = gamma
        self.lengths = [len(ep.get("rewards", [])) for ep in self.episodes]
        self.num_episodes = len(self.episodes)

        if self.num_episodes == 0:
            raise ValueError("No episodes found in the PPO log file")

        all_rtg = np.concatenate([np.asarray(ep["returns_to_go"], dtype=np.float32) for ep in self.episodes])
        self.rtg_max, self.rtg_min = np.max(all_rtg), np.min(all_rtg)
        print(f"Dataset Loaded: {self.num_episodes} episodes. RTG range: {self.rtg_min:.2f} to {self.rtg_max:.2f}")

    def __len__(self) -> int:
        return max(1000, self.num_episodes * 100)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx = np.random.randint(0, self.num_episodes)
        episode = self.episodes[ep_idx]
        ep_len = self.lengths[ep_idx]

        start_t = np.random.randint(0, max(1, ep_len))
        end_t = min(start_t + self.context_len, ep_len)
        seq_len = end_t - start_t

        images = [np.asarray(img, dtype=np.float32) for img in episode["images"][start_t:end_t]]
        actions = np.asarray(episode["actions"][start_t:end_t], dtype=np.int64)
        rtg = np.asarray(episode["returns_to_go"][start_t:end_t], dtype=np.float32)

        if self.rtg_max > self.rtg_min:
            rtg = (rtg - self.rtg_min) / (self.rtg_max - self.rtg_min + 1e-5)

        timesteps = np.arange(start_t, end_t, dtype=np.int64)

        state_batch = []
        for img in images:
            state_batch.append(_prepare_image_tensor(img, self.image_size))
        if len(state_batch) == 0:
            state_batch = [_prepare_image_tensor(np.zeros((84, 84, 3), dtype=np.uint8), self.image_size)]

        states = np.stack(state_batch, axis=0).astype(np.float32)

        pad_len = self.context_len - seq_len
        if pad_len > 0:
            states = np.concatenate([
                np.zeros((pad_len, 3, self.image_size[1], self.image_size[0]), dtype=np.float32),
                states
            ], axis=0)
            actions = np.concatenate([np.zeros(pad_len, dtype=np.int64), actions], axis=0)
            rtg = np.concatenate([np.zeros(pad_len, dtype=np.float32), rtg], axis=0)
            timesteps = np.concatenate([np.zeros(pad_len, dtype=np.int64), timesteps], axis=0)

        attention_mask = np.concatenate([
            np.zeros(pad_len, dtype=np.float32),
            np.ones(seq_len, dtype=np.float32)
        ], axis=0)

        return {
            "states": torch.tensor(states),
            "actions": torch.tensor(actions, dtype=torch.long),
            "returns_to_go": torch.tensor(rtg, dtype=torch.float32).unsqueeze(-1),
            "timesteps": torch.tensor(timesteps, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
        }


def _prepare_image_tensor(image: Any, image_size: Tuple[int, int]) -> np.ndarray:
    """観測を Decision Transformer へ入力できる形 (3, H, W) に整形する。"""
    if image is None:
        return np.zeros((3, image_size[1], image_size[0]), dtype=np.float32)

    if isinstance(image, (str, os.PathLike)):
        try:
            img = Image.open(image).convert("RGB")
            img = img.resize(image_size, Image.BILINEAR)
            return np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        except Exception:
            return np.zeros((3, image_size[1], image_size[0]), dtype=np.float32)

    arr = np.asarray(image)
    if arr.ndim == 0:
        return np.zeros((3, image_size[1], image_size[0]), dtype=np.float32)

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)

    if arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[1] > 5 and arr.shape[2] > 5:
            img = arr
        elif arr.shape[-1] in (1, 3, 4) and arr.shape[0] > 5 and arr.shape[1] > 5:
            img = np.transpose(arr, (2, 0, 1))
        else:
            img = arr

        if img.shape[0] != 3:
            if img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)
            else:
                img = img[:3]

        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)

        if img.shape[0] == 3 and img.shape[1] != image_size[1] and img.shape[2] != image_size[0]:
            pil_image = Image.fromarray(np.uint8(np.clip(img.transpose(1, 2, 0), 0, 255)))
            pil_image = pil_image.resize(image_size, Image.BILINEAR)
            img = np.array(pil_image, dtype=np.float32).transpose(2, 0, 1) / 255.0
            return img

        if img.shape[0] == 3 and img.shape[1] > 5 and img.shape[2] > 5:
            if img.shape[1:] != (image_size[1], image_size[0]):
                pil_image = Image.fromarray(np.uint8(np.clip(img.transpose(1, 2, 0), 0, 255)))
                pil_image = pil_image.resize(image_size, Image.BILINEAR)
                img = np.array(pil_image, dtype=np.float32).transpose(2, 0, 1) / 255.0
                return img
            return img.astype(np.float32) / 255.0 if img.max() > 1.0 else img.astype(np.float32)

    return np.zeros((3, image_size[1], image_size[0]), dtype=np.float32)


def _load_ppo_log(log_path: Path) -> List[Dict[str, Any]]:
    if not log_path.exists():
        raise FileNotFoundError(f"PPO log not found: {log_path}")

    suffix = log_path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with log_path.open("rb") as f:
            data = pickle.load(f)
    elif suffix == ".json":
        with log_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif suffix == ".jsonl":
        episodes = []
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
        return episodes
    else:
        raise ValueError(f"Unsupported PPO log format: {log_path}")

    if isinstance(data, dict):
        if "episodes" in data:
            return list(data["episodes"])
        if "trajectories" in data:
            return list(data["trajectories"])
        return [data]
    if isinstance(data, list):
        return data
    return [data]


def _normalize_episode(episode: Dict[str, Any], image_size: Tuple[int, int], gamma: float = 0.99) -> Dict[str, Any]:
    actions = list(episode.get("actions", episode.get("a", [])))
    rewards = list(episode.get("rewards", episode.get("r", [])))
    observations = episode.get("observations", episode.get("images", episode.get("states", episode.get("frames", []))))
    if observations is None:
        observations = []
    if not isinstance(observations, list):
        observations = list(observations)

    if len(observations) == 0 and "image_paths" in episode:
        observations = list(episode.get("image_paths", []))

    if len(observations) == 0 and len(actions) > 0:
        observations = [None] * len(actions)

    if len(actions) == 0 and len(rewards) > 0:
        actions = [0] * len(rewards)

    if len(actions) != len(rewards):
        n = min(len(actions), len(rewards), len(observations))
        actions = actions[:n]
        rewards = rewards[:n]
        observations = observations[:n]

    images = [_prepare_image_tensor(obs, image_size) for obs in observations]

    rewards_array = np.asarray(rewards, dtype=np.float32)
    rtg = np.zeros_like(rewards_array)
    running = 0.0
    for t in range(len(rewards_array) - 1, -1, -1):
        running = rewards_array[t] + gamma * running
        rtg[t] = running

    return {
        "images": images,
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": rewards_array,
        "returns_to_go": rtg.astype(np.float32),
    }


def make_dt_dataset_from_ppo_logs(
    log_path: str,
    output_path: Optional[str] = None,
    context_len: int = 30,
    image_size: Tuple[int, int] = (84, 84),
    gamma: float = 0.99,
) -> Tuple[PPORolloutDataset, List[Dict[str, Any]]]:
    """PPO 推論ログから Decision Transformer 用の学習データセットを作る。

    Parameters
    ----------
    log_path : str
        PPO ログファイル (.pkl / .json / .jsonl) またはディレクトリ。
    output_path : Optional[str]
        変換後の pickle を保存するパス。省略時は保存しない。
    context_len : int
        1サンプルに含めるコンテキスト長。
    image_size : Tuple[int, int]
        画像サイズ (H, W)。
    gamma : float
        RTG 計算に用いる割引率。
    """
    log_root = Path(log_path)
    if log_root.is_dir():
        log_files = sorted(log_root.glob("*"))
        log_files = [p for p in log_files if p.is_file() and p.suffix.lower() in {".pkl", ".pickle", ".json", ".jsonl"}]
        if not log_files:
            raise FileNotFoundError(f"No PPO log files found in directory: {log_root}")
    else:
        log_files = [log_root]

    episodes: List[Dict[str, Any]] = []
    for path in log_files:
        raw_episodes = _load_ppo_log(path)
        for episode in raw_episodes:
            if isinstance(episode, dict):
                episodes.append(_normalize_episode(episode, image_size=image_size, gamma=gamma))

    if not episodes:
        raise ValueError("No episode data could be loaded from the PPO log")

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            pickle.dump(episodes, f)
        print(f"Saved DT dataset to {out_path}")

    dataset = PPORolloutDataset(episodes, context_len=context_len, image_size=image_size, gamma=gamma)
    return dataset, episodes


def train_dt_from_ppo_logs(
    log_path: str,
    output_path: Optional[str] = None,
    batch_size: int = 32,
    epochs: int = 10,
    context_len: int = 30,
    image_size: Tuple[int, int] = (84, 84),
    gamma: float = 0.99,
    learning_rate: float = 1e-4,
):
    """PPO ログから DT データセットを作り、簡易的にモデル学習を開始する。"""
    if DecisionTransformer is None:
        raise ImportError("make_DT.py の DecisionTransformer を読み込めませんでした")

    dataset, episodes = make_dt_dataset_from_ppo_logs(
        log_path=log_path,
        output_path=output_path,
        context_len=context_len,
        image_size=image_size,
        gamma=gamma,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = DecisionTransformer(action_vocab_size=256, hidden_size=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rtg = batch["returns_to_go"].to(device)
            timesteps = batch["timesteps"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            action_logits = model(states, actions, rtg, timesteps, attention_mask=attention_mask)
            logits_flat = action_logits.view(-1, 256)
            actions_flat = actions.view(-1)
            mask_flat = attention_mask.view(-1)

            loss = torch.nn.functional.cross_entropy(logits_flat, actions_flat, reduction="none")
            loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            total_loss += float(loss.item())

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / max(1, len(dataloader)):.4f}")

    return model, dataset, episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO 推論ログから Decision Transformer 用データセットを作成")
    parser.add_argument("--log-path", type=str, required=True, help="PPO ログファイル (.pkl/.json/.jsonl) またはディレクトリ")
    parser.add_argument("--output", type=str, default=None, help="出力 pickle パス")
    parser.add_argument("--context-len", type=int, default=30, help="コンテキスト長")
    parser.add_argument("--image-size", type=int, nargs=2, default=[84, 84], help="画像サイズ (H W)")
    parser.add_argument("--gamma", type=float, default=0.99, help="RTG 割引率")
    parser.add_argument("--train", action="store_true", help="作成後に DT 学習まで実行")
    args = parser.parse_args()

    if args.train:
        train_dt_from_ppo_logs(
            log_path=args.log_path,
            output_path=args.output,
            context_len=args.context_len,
            image_size=tuple(args.image_size),
            gamma=args.gamma,
        )
    else:
        make_dt_dataset_from_ppo_logs(
            log_path=args.log_path,
            output_path=args.output,
            context_len=args.context_len,
            image_size=tuple(args.image_size),
            gamma=args.gamma,
        )


if __name__ == "__main__":
    main()
