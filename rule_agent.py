"""
手書きルールでマリオを操作するエージェント（難易度検証パネル用）。

PPO パネル (panel_difficulty.py) には2つの限界があった:
  1. 腕前がそろいすぎていた（6体のクリア率が 42〜54% に固まっていた）
  2. PPO は DT の教師と系統を共有しており、「共通の癖」まで一致とみなしてしまう恐れがあった
学習を一切しないルールベースのエージェントは PPO / DT と完全に独立しており、
パラメータで腕前を連続的に変えられるので、この2つを同時に解消できる。

判断材料はゲーム内部の状態（画像ではない）:
  - 前方の穴   : 足元の高さ以下に床が1枚も無い列
  - 前方の壁   : マリオの体の高さにある固体タイル（パイプ等）
  - 前方の敵   : 生きている Mob（クリボー・ノコノコ）
  - 行き詰まり : しばらく前に進めていない
いずれかがあれば接地中に跳び、無ければ右へ走る。ジャンプの高さは押し続けても変わらない
（traits/jump.py、約3.75マス）ので、判断すべきは「いつ跳ぶか」と「ダッシュするか」だけ。

腕前のパラメータ:
  look      : 穴・壁を何マス先から見るか。小さすぎると踏切りが遅れ、大きすぎると早跳びで穴に落ちる
  enemy_look: 敵を何マス先から見るか
  react     : 反応の遅れ（何ステップ前の知覚で動くか）
  noise     : 各ステップでランダムな行動を取る確率（操作ミス）
  dash      : ダッシュするか。しないと3マス幅の穴を越えられない
"""

from collections import deque

import numpy as np

# 行動番号（classes/AgentInput.py）
NOP, LEFT, RIGHT, RIGHT_JUMP, RIGHT_DASH, RIGHT_DASH_JUMP = 0, 1, 2, 5, 7, 8
N_ACTIONS = 10

# 強い → 弱い の順。操作ミスの確率 (noise) と反応の遅れ (react) で腕前を段階的に落とす。
# ダッシュしない設定は3マス幅の穴を必ず越えられず、全ステージ 0% になって
# 難しさの区別に役立たないので使わない。学習用ステージで調整（rule_panel.py --calibrate）。
PRESETS = {
    "rule_s0": dict(noise=0.00, react=0),
    "rule_s1": dict(noise=0.05, react=0),
    "rule_s2": dict(noise=0.12, react=1),
    "rule_s3": dict(noise=0.20, react=1),
    "rule_s4": dict(noise=0.30, react=2),
    "rule_s5": dict(noise=0.45, react=2),
}


class RuleAgent:
    def __init__(self, look=2, wall_look=4, enemy_look=4, react=0, noise=0.0, dash=True,
                 stall_patience=6, backoff=4, seed=0):
        self.look, self.wall_look, self.enemy_look = look, wall_look, enemy_look
        self.backoff = backoff
        self.react, self.noise, self.dash = react, noise, dash
        self.stall_patience = stall_patience
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.best_x = -1
        self.stall = 0
        self.backing = 0          # 残りの後退ステップ数
        self.percepts = deque(maxlen=self.react + 1)

    def _danger(self, env):
        game = env.unwrapped
        m, lv = game.mario, game.level
        grid = lv.level
        H, W = len(grid), len(grid[0])

        def solid(x, y):
            return 0 <= y < H and 0 <= x < W and grid[y][x].rect is not None

        cx = m.rect.centerx // 32
        feet = m.rect.bottom // 32          # 立っているときに足の下にある段
        body = range(max(m.rect.top // 32, 0), feet)
        # 穴は近くで跳ぶ（早すぎると穴の中に着地する）。壁は早めに跳ぶ
        # （ダッシュ中は跳び上がりきる前に側面に着き、張りついて抜けられなくなる）
        for dx in range(1, max(self.look, self.wall_look) + 1):
            x = cx + dx
            if x >= W:
                break
            if dx <= self.look and not any(solid(x, y) for y in range(feet, H)):
                return True                 # 穴
            if dx <= self.wall_look and any(solid(x, y) for y in body):
                return True                 # 壁（パイプなど）
        for e in lv.entityList:
            if getattr(e, "type", None) == "Mob" and e.alive:
                dx = e.rect.x - m.rect.x
                # 真横や少し後ろの敵とも当たり判定が重なるので、-16px から見る
                if -16 < dx < self.enemy_look * 32 and abs(e.rect.y - m.rect.y) < 64:
                    return True             # 敵
        return False

    def act(self, env):
        m = env.unwrapped.mario
        if m.rect.x > self.best_x:
            self.best_x, self.stall = m.rect.x, 0
        else:
            self.stall += 1

        # 行き詰まったら一度左へ下がって助走をつけ直す
        # （壁に張りついたまま跳ね続けて抜けられなくなるのを防ぐ）
        if self.stall >= self.stall_patience and self.backing == 0:
            self.backing, self.stall = self.backoff, 0
        if self.backing > 0:
            self.backing -= 1
            return LEFT

        # 反応の遅れ: react ステップ前の知覚で判断する
        self.percepts.append(self._danger(env))
        danger = self.percepts[0]

        # 接地を確かめてから跳ぶのではなく、危険があればジャンプボタンを押すだけにする。
        # 立っている間 mario.onGround は1フレームおきに True/False を繰り返し
        # （速度 0.8px が整数座標で丸められて地面に触れないフレームがある）、
        # 4フレームごとに読むと毎回 False 側だけを見てしまい、一度も跳べなかった。
        # ボタンを押しておけば、4フレームのうち接地している瞬間にゲーム側が跳ばせる
        # （PPO や DT も同じ押し方をしている）。
        if danger:
            action = RIGHT_DASH_JUMP if self.dash else RIGHT_JUMP
        else:
            action = RIGHT_DASH if self.dash else RIGHT
        if self.noise > 0 and self.rng.random() < self.noise:
            action = int(self.rng.integers(N_ACTIONS))  # 操作ミス
        return action


SKIP = 4


def run_episode(job):
    """1エピソードを最後まで遊ばせて結果を返す（rule_panel.py が子プロセスで呼ぶ）。

    multiprocessing で子プロセスに渡す関数は、起動スクリプト (__main__) ではなく
    この普通のモジュールに置く。Windows の spawn では子プロセスが __main__ を読み直すが、
    この環境ではそこが VSCode 側の仕組みに横取りされ、関数が見つからずに固まるため。
    """
    from classes.MarioGymEnv import MarioEnv
    from classes.wrappers import SkipFrame
    env = SkipFrame(MarioEnv(level=job["level"], render_mode=None,
                             max_episode_steps=job["max_steps"] * SKIP), skip=SKIP)
    env.reset(seed=job["env_seed"])
    agent = RuleAgent(**PRESETS[job["agent_name"]], seed=job["sample_seed"])
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
