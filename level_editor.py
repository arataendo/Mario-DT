"""
ステージをタイル単位で編集するための表現（段階3: 難易度を指定したステージ生成の探索で使う）。

ステージの JSON を、次の編集しやすい形に分解して持つ:
  ground[x]   : 列 x に地面があるか（無ければ穴）
  pipes{x: h} : 列 x〜x+1 を占める高さ h (1〜3) のパイプ
  goombas / koopas : 敵の x 座標（地面の上、行13）
  その他（コインブロックなど）は元の JSON のまま保持する

編集（mutate）は「穴の追加・削除・幅の変更・移動」「パイプの追加・削除・高さの変更・移動」
「敵の追加・削除・移動・種類の変更」。移動や種類の変更は物の量を変えない編集で、
段階2で分かった「量では分からない難しさ」を探索が動かせるようにするために入れている。

制約（is_valid）は generate_level.validate_level と同じ安全条件に、
  - 障害物（穴・パイプ）の間に助走 MIN_RUNWAY 列
  - パイプは地面の上、敵は地面の上かつパイプの外
を加えたもの。制約を満たさない編集結果は捨てて引き直す。
"""

import copy
import json

from generate_level import (GOAL_SAFE_TILES, GROUND_ROW, MAX_SAFE_GAP_TILES,
                            MAX_SAFE_PIPE_TILES, SPAWN_SAFE_TILES, validate_level)

MIN_RUNWAY = 2          # 障害物どうしの間に最低限必要な地面の列数
MIN_ENEMY_SPACING = 2   # 敵どうしの最小間隔（同じ列や隣の列に重ねない）


class EditableLevel:
    def __init__(self, length, ground, pipes, goombas, koopas, extras):
        self.length = length
        self.ground = list(ground)
        self.pipes = dict(pipes)
        self.goombas = sorted(goombas)
        self.koopas = sorted(koopas)
        self.extras = extras  # コインブロックなど、編集しない要素（JSON の entities の残り）

    # ---- JSON との変換 ----
    @classmethod
    def from_json(cls, data):
        lv = data["level"]
        length = data["length"]
        top = {x for x, y in lv["objects"]["ground"] if y == GROUND_ROW}
        ground = [x in top for x in range(length)]
        pipes = {px: GROUND_ROW - top_y for px, top_y, _ in lv["objects"]["pipe"]}
        ent = lv["entities"]
        extras = {k: v for k, v in ent.items() if k not in ("Goomba", "Koopa")}
        return cls(length, ground, pipes,
                   [x for x, _ in ent.get("Goomba", [])],
                   [x for x, _ in ent.get("Koopa", [])],
                   copy.deepcopy(extras))

    def to_json(self):
        ground = sorted([x, y] for x in range(self.length) if self.ground[x]
                        for y in range(GROUND_ROW, GROUND_ROW + 2))
        entities = dict(copy.deepcopy(self.extras))
        entities["Goomba"] = [[x, GROUND_ROW - 1] for x in sorted(self.goombas)]
        entities["Koopa"] = [[x, GROUND_ROW - 1] for x in sorted(self.koopas)]
        return {
            "id": 1,
            "length": self.length,
            "level": {
                "objects": {
                    "bush": [], "sky": [], "cloud": [],
                    "pipe": [[x, GROUND_ROW - h, h] for x, h in sorted(self.pipes.items())],
                    "ground": ground,
                },
                "layers": {
                    "sky": {"x": [0, self.length], "y": [0, GROUND_ROW]},
                    "ground": {"x": [0, self.length], "y": [GROUND_ROW + 1, GROUND_ROW + 3]},
                },
                "entities": entities,
            },
        }

    def copy(self):
        return EditableLevel(self.length, self.ground, self.pipes,
                             self.goombas, self.koopas, copy.deepcopy(self.extras))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, separators=(",", ":"))

    # ---- 構造の要約（ベースラインや記録用）----
    def gap_runs(self):
        runs, start = [], None
        for x in range(self.length + 1):
            hole = x < self.length and not self.ground[x]
            if hole and start is None:
                start = x
            elif not hole and start is not None:
                runs.append((start, x - start))
                start = None
        return runs

    def summary(self):
        runs = self.gap_runs()
        return dict(enemies=len(self.goombas) + len(self.koopas), goombas=len(self.goombas),
                    koopas=len(self.koopas), gaps=len(runs), gap_tiles=sum(w for _, w in runs),
                    pipes=len(self.pipes), max_pipe=max(self.pipes.values(), default=0))

    # ---- 制約 ----
    def hazards(self):
        """(開始列, 終了列(含まない)) の一覧。穴とパイプ"""
        hz = [(s, s + w) for s, w in self.gap_runs()]
        hz += [(x, x + 2) for x in self.pipes]
        return sorted(hz)

    def is_valid(self):
        L = self.length
        lo, hi = SPAWN_SAFE_TILES, L - GOAL_SAFE_TILES   # 障害物・敵を置いてよい範囲 [lo, hi)
        if not all(self.ground[x] for x in range(0, lo)):
            return False
        if not all(self.ground[x] for x in range(hi, L)):
            return False
        for s, w in self.gap_runs():
            if w > MAX_SAFE_GAP_TILES:
                return False
        pipe_cols = set()
        for x, h in self.pipes.items():
            if not (1 <= h <= MAX_SAFE_PIPE_TILES) or x < lo or x + 2 > hi:
                return False
            if not (self.ground[x] and self.ground[x + 1]):
                return False
            pipe_cols |= {x, x + 1}
        hz = self.hazards()
        for (s1, e1), (s2, e2) in zip(hz, hz[1:]):
            if s2 - e1 < MIN_RUNWAY:            # 重なり・助走不足
                return False
        enemies = sorted(self.goombas + self.koopas)
        for x in enemies:
            if x < lo or x >= hi or not self.ground[x] or x in pipe_cols:
                return False
        for a, b in zip(enemies, enemies[1:]):
            if b - a < MIN_ENEMY_SPACING:
                return False
        try:
            validate_level(self.to_json())
        except AssertionError:
            return False
        return True

    def repair(self):
        """置き方に意味のない敵を取り除く（探索の初期集団にする生成器のステージ用）。

        生成器は敵を2体ペアで置くとき、2体目が穴の上やパイプの中に来る場合を考慮しておらず、
        コーパスの約半数にそうした敵がいた（穴の上の敵はすぐ落ちて消え、パイプの中の敵は
        埋まったまま残る）。探索がこうした不具合じみた状態を悪用しないよう、禁止して取り除く。
        """
        lo, hi = SPAWN_SAFE_TILES, self.length - GOAL_SAFE_TILES
        pipe_cols = {c for x in self.pipes for c in (x, x + 1)}
        ok = lambda x: lo <= x < hi and self.ground[x] and x not in pipe_cols
        self.goombas = [x for x in self.goombas if ok(x)]
        self.koopas = [x for x in self.koopas if ok(x)]
        kept, last = set(), -10
        for x in sorted(self.goombas + self.koopas):
            if x - last >= MIN_ENEMY_SPACING:
                kept.add(x)
                last = x
        self.goombas = [x for x in self.goombas if x in kept]
        self.koopas = [x for x in self.koopas if x in kept and x not in self.goombas]
        return self

    # ---- 編集 ----
    def _free_x(self, rng):
        return int(rng.integers(SPAWN_SAFE_TILES, self.length - GOAL_SAFE_TILES))

    def mutate(self, rng, n_ops=None, max_tries=200):
        """1〜3個の編集を施した、制約を満たす新しいステージを返す"""
        n_ops = n_ops or int(rng.integers(1, 4))
        for _ in range(max_tries):
            child = self.copy()
            for _ in range(n_ops):
                child._random_op(rng)
            if child.is_valid() and child.to_json() != self.to_json():
                return child
        return self.copy()   # 有効な編集が見つからなければ親をそのまま返す

    def _random_op(self, rng):
        ops = [self._add_gap, self._remove_gap, self._resize_gap, self._move_gap,
               self._add_pipe, self._remove_pipe, self._reheight_pipe, self._move_pipe,
               self._add_enemy, self._remove_enemy, self._move_enemy, self._swap_enemy]
        ops[int(rng.integers(len(ops)))](rng)

    # 穴
    def _add_gap(self, rng):
        x, w = self._free_x(rng), int(rng.integers(1, MAX_SAFE_GAP_TILES + 1))
        for c in range(x, min(x + w, self.length)):
            self.ground[c] = False

    def _remove_gap(self, rng):
        runs = self.gap_runs()
        if runs:
            s, w = runs[int(rng.integers(len(runs)))]
            for c in range(s, s + w):
                self.ground[c] = True

    def _resize_gap(self, rng):
        runs = self.gap_runs()
        if runs:
            s, w = runs[int(rng.integers(len(runs)))]
            if rng.random() < 0.5 and w > 1:
                self.ground[s + w - 1] = True
            elif s + w < self.length:
                self.ground[s + w] = False

    def _move_gap(self, rng):
        runs = self.gap_runs()
        if runs:
            s, w = runs[int(rng.integers(len(runs)))]
            d = int(rng.integers(-3, 4)) or 1
            for c in range(s, s + w):
                self.ground[c] = True
            for c in range(s + d, s + d + w):
                if 0 <= c < self.length:
                    self.ground[c] = False

    # パイプ
    def _add_pipe(self, rng):
        self.pipes[self._free_x(rng)] = int(rng.integers(1, MAX_SAFE_PIPE_TILES + 1))

    def _remove_pipe(self, rng):
        if self.pipes:
            self.pipes.pop(list(self.pipes)[int(rng.integers(len(self.pipes)))])

    def _reheight_pipe(self, rng):
        if self.pipes:
            x = list(self.pipes)[int(rng.integers(len(self.pipes)))]
            self.pipes[x] = min(MAX_SAFE_PIPE_TILES, max(1, self.pipes[x] + (1 if rng.random() < 0.5 else -1)))

    def _move_pipe(self, rng):
        if self.pipes:
            x = list(self.pipes)[int(rng.integers(len(self.pipes)))]
            h = self.pipes.pop(x)
            self.pipes[x + (int(rng.integers(-3, 4)) or 1)] = h

    # 敵
    def _enemy_list(self, rng):
        return self.koopas if (self.koopas and (not self.goombas or rng.random() < 0.3)) else self.goombas

    def _add_enemy(self, rng):
        (self.koopas if rng.random() < 0.25 else self.goombas).append(self._free_x(rng))

    def _remove_enemy(self, rng):
        lst = self._enemy_list(rng)
        if lst:
            lst.pop(int(rng.integers(len(lst))))

    def _move_enemy(self, rng):
        lst = self._enemy_list(rng)
        if lst:
            i = int(rng.integers(len(lst)))
            lst[i] += int(rng.integers(-4, 5)) or 1

    def _swap_enemy(self, rng):
        src = self._enemy_list(rng)
        if src:
            x = src.pop(int(rng.integers(len(src))))
            (self.goombas if src is self.koopas else self.koopas).append(x)
