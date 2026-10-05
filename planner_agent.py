"""
シミュレータで先読みして行動を決めるエージェント（難易度検証の「審判」用）。

段階3 v3 で、目的関数（DT + ルール）で細かく制御した難しさが、独立な審判（PPO パネル）には
一部しか伝わらないことが分かった。「目的関数の評価器の種類を増やすと伝わり方が上がるか」を
確かめるには、どの目的関数にも含まれない種類の審判が要る。

このエージェントは学習もルールの手書きもせず、ゲームの状態を複製して
「この操作をしたら数手先でどうなるか」を実際に試して行動を選ぶ。DT（PPO を模倣した学習モデル）、
ルールベース、PPO のいずれとも仕組みが違う。

  - 候補の行動それぞれについて、状態を複製し、最初の1手をその行動、残りを
    「走り続ける」「走りながら跳び続ける」の2通りで horizon 手先まで進め、良い方で評価する
    （続け方を「走る」だけにすると、パイプなどの壁の前ではどの候補も「進めない」で同点になり、
     壁に張りついたまま動けなくなった）
  - 最も良かった候補を実行する
腕前は先読みの深さ (horizon) と操作ミスの確率 (noise) で変える。

ゲームはシードを固定すれば完全に決定的で、複製どうし・本体とも独立に進むことを確認済み。
先読み中は観測画像の作成を省く（物理は描画処理の中で進むので描画自体は省けない）。
これで1フレーム 3.4ms → 0.6ms になり、進み方は変わらない。
"""

import copy

import numpy as np
import pygame

# 画像・時計は複製せず共有する（描画用で、物理には関係しない。複製を 17ms 程度に抑える）
# 型はクラスそのものから取る。以前は type(pygame.time.Clock()) とインスタンスを作っており、
# モジュールを読み込んだだけで親プロセスの SDL タイマーが初期化されていた疑いがある。
# Linux で fork した子プロセスがそれを引き継ぐと固まり、研究室PCで評価が進まなかった。
for _T in (pygame.Surface, pygame.time.Clock):
    copy._deepcopy_dispatch[_T] = lambda x, memo: x

SKIP = 4
NOP, LEFT, RIGHT, RIGHT_JUMP, RIGHT_DASH, RIGHT_DASH_JUMP = 0, 1, 2, 5, 7, 8
CANDIDATES = (RIGHT_DASH, RIGHT_DASH_JUMP, RIGHT, RIGHT_JUMP, NOP, LEFT)
N_ACTIONS = 10
DEATH_PENALTY = 1e6

# 強い → 弱い。学習用ステージ7個 × 2本での平均クリア率（planner_panel.py --calibrate）:
#   h8 100% / h6 79% / h4 64% / h3 64% / h2 36%（到達率 99/88/83/73/56%）
# 審判には強・中・弱の h8, h4, h2 を使う（run_v4.sh の PLANNER_AGENTS）
PRESETS = {
    "plan_h8": dict(horizon=8, noise=0.00),
    "plan_h6": dict(horizon=6, noise=0.05),
    "plan_h4": dict(horizon=4, noise=0.10),
    "plan_h3": dict(horizon=3, noise=0.20),
    "plan_h2": dict(horizon=2, noise=0.30),
}


def _sim_step(sim, action):
    """複製した環境を1手（4フレーム）進める。(死んだか, ゴールしたか) を返す"""
    for _ in range(SKIP):
        _, _, term, _, info = sim.step(action)
        if term:
            return info.get("reason") != "level_complete", info.get("reason") == "level_complete"
    return False, False


class PlannerAgent:
    def __init__(self, horizon=8, noise=0.0, candidates=CANDIDATES,
                 continuations=(RIGHT_DASH, RIGHT_DASH_JUMP), seed=0):
        self.horizon, self.noise = horizon, noise
        self.candidates, self.continuations = candidates, continuations
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def _score(self, game, first):
        return max(self._rollout(game, first, cont) for cont in self.continuations)

    def _rollout(self, game, first, cont):
        sim = copy.deepcopy(game)
        sim._get_observation = lambda: None   # 先読みでは観測画像は要らない
        x0 = sim.mario.rect.x
        for k in range(self.horizon):
            dead, goal = _sim_step(sim, first if k == 0 else cont)
            if goal:
                return 1e5 - k                 # 早くゴールするほど良い
            if dead:
                return -DEATH_PENALTY + k      # 死ぬなら、できるだけ遅い方がまし
        return float(sim.mario.rect.x - x0)

    def act(self, env):
        if self.noise > 0 and self.rng.random() < self.noise:
            return int(self.rng.integers(N_ACTIONS))  # 操作ミス
        game = env.unwrapped
        scores = [self._score(game, a) for a in self.candidates]
        return int(self.candidates[int(np.argmax(scores))])   # 同点なら候補の先頭（走る）を優先


def run_episode(job):
    """1エピソードを最後まで遊ばせて結果を返す（planner_panel.py が子プロセスで呼ぶ）"""
    from classes.MarioGymEnv import MarioEnv
    from classes.wrappers import SkipFrame
    env = SkipFrame(MarioEnv(level=job["level"], render_mode=None,
                             max_episode_steps=job["max_steps"] * SKIP), skip=SKIP)
    env.reset(seed=job["env_seed"])
    agent = PlannerAgent(**PRESETS[job["agent_name"]], seed=job["sample_seed"])
    total, steps, reason, info = 0.0, 0, "timeout", {}
    for t in range(job["max_steps"]):
        _, reward, term, trunc, info = env.step(agent.act(env))
        total += reward
        steps = t + 1
        if term or trunc:
            reason = info.get("reason", "timeout" if trunc else None) or "timeout"
            break
    env.close()
    return dict(job, cleared=reason == "level_complete", reason=reason,
                total_reward=total, steps=steps, progress=float(info.get("progress", 0.0)))
