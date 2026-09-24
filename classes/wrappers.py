"""
PPO / DT で共通して使う観測ラッパー群。

train_ppo.py / infer_ppo.py / collect_dt_dataset.py は必ずここの
`make_mario_env()` を使うこと。学習時と推論時で観測の作り方が
1ピクセルでもずれると性能が出ないため、定義は1箇所に集約する。

観測: (3 * N_STACK, 84, 84) uint8
  - SkipFrame(4)  : 同じ行動を4ゲームフレーム繰り返す
  - 84x84 にリサイズ (RGB)
  - 直近 4 フレームをチャンネル方向に連結（速度・落下方向を推定可能にする）
"""

from collections import deque

import cv2
import numpy as np
import gymnasium as gym

# --- 共通ハイパーパラメータ（学習・推論・データ収集で必ず同じ値を使う） ---
FRAME_SIZE = (84, 84)   # cv2.resize は (width, height)
N_STACK = 4             # フレームスタック数
SKIP = 4                # フレームスキップ数


class SkipFrame(gym.Wrapper):
    """指定フレーム数だけ同じアクションを繰り返す。"""

    def __init__(self, env, skip=SKIP):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None

        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class MarioImageWrapper(gym.ObservationWrapper):
    """Dict 観測から画像を取り出し (3, 84, 84) uint8 に変換する。"""

    def __init__(self, env, size=FRAME_SIZE):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(3, size[1], size[0]),
            dtype=np.uint8,
        )

    def observation(self, obs):
        img = obs["image"] if isinstance(obs, dict) else obs
        # (C, H, W) -> (H, W, C)
        img_hwc = np.transpose(img, (1, 2, 0))
        img_resized = cv2.resize(img_hwc, self.size, interpolation=cv2.INTER_AREA)
        # (H, W, C) -> (C, H, W)
        return np.ascontiguousarray(
            np.transpose(img_resized, (2, 0, 1)), dtype=np.uint8
        )


class FrameStack(gym.Wrapper):
    """直近 n_stack フレームをチャンネル方向に連結する。

    単一フレームだけだと「速度」「落下中か上昇中か」「敵がどちらに動いているか」
    が観測から判定できず、穴や敵への対処が原理的に学習できない。
    """

    def __init__(self, env, n_stack=N_STACK):
        super().__init__(env)
        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)

        c, h, w = env.observation_space.shape
        self.frame_channels = c
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(c * n_stack, h, w),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_stack):
            self.frames.append(obs)
        return self._stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._stacked(), reward, terminated, truncated, info

    def _stacked(self):
        return np.concatenate(list(self.frames), axis=0)


class StallGuard:
    """決定論的推論(argmax)がその場で完全に固着するのを防ぐ保険機構。

    問題: パイプ等の障害物の直前で "前進のないR+Dash" のような行動が
    argmax として選ばれ続けると、観測(直近4フレームスタック)が完全に
    同一のまま変化しなくなり、同じ行動→同じ観測→同じ行動…という
    自己ループに入って抜け出せなくなる（学習時は確率的サンプリングで
    偶然ジャンプして回避できるが、deterministic=True の推論には
    脱出手段が無い）。

    対策: 直近 patience 回の意思決定で mario_x の自己最高値が
    更新されなければ、次の行動を強制的にジャンプ系の脱出行動に
    すり替える。ポリシー自体は変更しない、推論側だけの安全弁。
    """

    def __init__(self, patience=15, escape_actions=(8, 5, 3)):
        self.patience = patience
        self.escape_actions = escape_actions
        self.best_x = -1
        self.stall_count = 0
        self._escape_idx = 0

    def reset(self):
        self.best_x = -1
        self.stall_count = 0
        self._escape_idx = 0

    def choose(self, proposed_action, current_x):
        """proposed_action: モデルが選んだ行動。current_x: 直前ステップ後の mario_x。

        直前の mario_x を見て停滞を検知し、必要なら行動を上書きして返す。
        """
        if current_x > self.best_x:
            self.best_x = current_x
            self.stall_count = 0
        else:
            self.stall_count += 1

        if self.stall_count >= self.patience:
            escape = self.escape_actions[self._escape_idx % len(self.escape_actions)]
            self._escape_idx += 1
            return escape

        return proposed_action


def latest_frame(stacked_obs, channels=3):
    """スタック済み観測から最新の1フレーム (3, 84, 84) を取り出す。

    DT 用データセットの画像保存で、従来どおり 3ch PNG を残すために使う。
    """
    return stacked_obs[-channels:]


def make_mario_env(
    level=None,
    max_episode_steps=8000,
    render_mode=None,
    random_level=False,
    skip=SKIP,
    n_stack=N_STACK,
    levels=None,
):
    """学習・推論・データ収集で共通の Mario 環境を作る。

    Parameters
    ----------
    level : str or None
        レベル名。None かつ random_level=True のときのみランダム選択。
    max_episode_steps : int
        **ゲーム内フレーム数**での上限。SkipFrame の内側で数えられるので、
        エージェントの意思決定回数は max_episode_steps / skip 回になる。
        人間が 4000 フレームでクリアするステージなら 8000 程度を確保すること。
    """
    from classes.MarioGymEnv import MarioEnv

    env = MarioEnv(
        level=level,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        random_level=random_level,
        levels=levels,
    )
    env = SkipFrame(env, skip=skip)
    env = MarioImageWrapper(env)
    env = FrameStack(env, n_stack=n_stack)
    return env
