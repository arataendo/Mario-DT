"""
難易度パラメータを指定してマリオのステージ (levels/*.json) を自動生成する
レベルエディター用スクリプト。

Mario の物理パラメータ（traits/jump.py, traits/go.py, entities/EntityBase.py）を
シミュレーションした結果、助走ありのジャンプで安全に届く距離は概ね以下の通り:
    - 同じ高さへの着地: 約 3.5 タイル
    - 上昇できる高さ:   約 3.75 タイル
これを超えるギャップ幅・パイプの高さを生成すると、エージェントの行動に関係なく
クリア不可能な区間ができてしまう（Level1-2.json で実際に発生していた不具合と同種）。
このスクリプトは難易度が上がっても必ずこの安全範囲内に収まるようにする。

使用方法:
    python generate_level.py --difficulty easy --output Level_easy_01
    python generate_level.py --difficulty 0.7 --length 200 --seed 42 --output Level_hard_01
    python generate_level.py --difficulty medium --count 5 --output-prefix Level_medium
"""

import argparse
import json
import os
import random


# --- ジャンプ物理のシミュレーション結果に基づく安全マージン付きの上限 ---
MAX_SAFE_GAP_TILES = 3       # 実測上限 約3.5タイル
MAX_SAFE_PIPE_TILES = 3      # 実測上限（上昇高さ）約3.75タイル
MIN_RUNWAY_TILES = 3         # 障害物の手前に必要な助走距離（加速に必要な距離）

SPAWN_SAFE_TILES = 6         # スポーン地点 (x=0) から先、必ず平らな地面にする
GOAL_SAFE_TILES = 4          # レベル終端も必ず平らな地面にする（詰み防止）

GROUND_ROW = 14              # 地面の最上段の行（既存レベルに合わせる）
GROUND_DEPTH = 2             # 地面の厚み（行数）


DIFFICULTY_PRESETS = {
    "easy":   0.15,
    "medium": 0.5,
    "hard":   0.85,
}


def resolve_difficulty(value: str) -> float:
    """'easy'/'medium'/'hard' か 0.0-1.0 の文字列を float に変換"""
    if value in DIFFICULTY_PRESETS:
        return DIFFICULTY_PRESETS[value]
    f = float(value)
    if not (0.0 <= f <= 1.0):
        raise ValueError("difficulty は 0.0〜1.0 の範囲、または easy/medium/hard で指定してください")
    return f


def lerp_int(lo: int, hi: int, t: float) -> int:
    return round(lo + (hi - lo) * t)


class LevelGenerator:
    def __init__(self, difficulty: float, length: int, rng: random.Random):
        self.difficulty = difficulty
        self.length = length
        self.rng = rng

        # 難易度に応じたパラメータレンジ
        self.max_gap = max(1, lerp_int(1, MAX_SAFE_GAP_TILES, difficulty))
        self.max_pipe = max(0, lerp_int(0, MAX_SAFE_PIPE_TILES, difficulty))
        self.hazard_prob = 0.15 + 0.45 * difficulty       # 1区間ごとに障害物を置く確率
        self.enemy_prob = 0.20 + 0.55 * difficulty        # 1区間ごとに敵を置く確率
        self.enemy_pair_prob = 0.15 + 0.4 * difficulty    # 敵を2体ペアで置く確率
        self.min_runway = MIN_RUNWAY_TILES + (0 if difficulty > 0.5 else 1)

        self.ground = set()      # (x, y) の集合。物理的な床タイル
        self.pipes = []          # [x, top_y, length]
        self.goombas = []
        self.koopas = []
        self.coin_bricks = []
        self.coin_boxes = []
        self.random_boxes = []
        self.gap_tiles = set()   # ギャップ（穴）になっている x 座標
        self.last_enemy_x = -999  # 直近に置いた敵の x（区間をまたいだ密集を防ぐ）

    def _add_flat_ground(self, x_start: int, x_end: int):
        """[x_start, x_end) に地面タイルを敷く"""
        for x in range(x_start, x_end):
            for d in range(GROUND_DEPTH):
                self.ground.add((x, GROUND_ROW + d))

    def generate(self):
        x = 0
        # --- スポーン直後は必ず安全な平地にする ---
        self._add_flat_ground(0, SPAWN_SAFE_TILES)
        x = SPAWN_SAFE_TILES

        goal_start = self.length - GOAL_SAFE_TILES

        last_hazard_end = x
        while x < goal_start:
            # 障害物の手前に最低限の助走距離を確保
            runway = self.rng.randint(self.min_runway, self.min_runway + 4)
            seg_end = min(x + runway, goal_start)
            self._add_flat_ground(x, seg_end)

            # 敵の配置（助走区間の後半、障害物の直前は避ける）
            if seg_end - x >= 2 and self.rng.random() < self.enemy_prob:
                enemy_x = self.rng.randint(x + 1, seg_end - 1)
                self._place_enemy(enemy_x)

            x = seg_end
            if x >= goal_start:
                break

            # 障害物（ギャップ or パイプ）を置くかどうか
            if self.rng.random() < self.hazard_prob and x - last_hazard_end >= self.min_runway:
                if self.max_pipe > 0 and self.rng.random() < 0.4:
                    x = self._place_pipe(x, goal_start)
                else:
                    x = self._place_gap(x, goal_start)
                last_hazard_end = x

        # --- ゴール手前は必ず平地 ---
        self._add_flat_ground(max(x, goal_start), self.length)

        return self._to_json()

    def _place_gap(self, x: int, goal_start: int) -> int:
        width = self.rng.randint(1, self.max_gap)
        width = min(width, max(1, goal_start - x - 1))
        for gx in range(x, x + width):
            self.gap_tiles.add(gx)
        return x + width

    def _place_pipe(self, x: int, goal_start: int) -> int:
        height = self.rng.randint(1, self.max_pipe)
        if x + 2 > goal_start:
            return x
        top_y = GROUND_ROW - height
        self.pipes.append([x, top_y, height])
        # パイプの土台部分は地面として扱う（左右は通行可能、上には乗れない前提で無視）
        self._add_flat_ground(x, x + 2)
        return x + 2

    MIN_ENEMY_SPACING = 4  # 区間をまたいだ敵同士の最小間隔（ペア配置は例外）

    def _place_enemy(self, x: int):
        # 直前に置いた敵群から近すぎる場合はスキップ（密集による理不尽な難易度を防ぐ）
        if x - self.last_enemy_x < self.MIN_ENEMY_SPACING:
            return

        pair = self.rng.random() < self.enemy_pair_prob
        use_koopa = self.difficulty > 0.6 and self.rng.random() < 0.25
        target = self.koopas if use_koopa else self.goombas
        target.append([x, GROUND_ROW - 1])
        self.last_enemy_x = x
        if pair:
            target.append([x + 2, GROUND_ROW - 1])
            self.last_enemy_x = x + 2
        # たまにコイン等を近くに配置
        if self.rng.random() < 0.3:
            self.coin_bricks.append([x, GROUND_ROW - 5])

    def _to_json(self):
        ground_list = sorted([list(t) for t in self.ground])
        return {
            "id": 1,
            "length": self.length,
            "level": {
                "objects": {
                    "bush": [],
                    "sky": [],
                    "cloud": [],
                    "pipe": self.pipes,
                    "ground": ground_list,
                },
                "layers": {
                    "sky": {"x": [0, self.length], "y": [0, GROUND_ROW]},
                    "ground": {"x": [0, self.length], "y": [GROUND_ROW + 1, GROUND_ROW + 3]},
                },
                "entities": {
                    "CoinBox": self.coin_boxes,
                    "coinBrick": self.coin_bricks,
                    "coin": [],
                    "Goomba": self.goombas,
                    "Koopa": self.koopas,
                    "RandomBox": self.random_boxes,
                    "Firebar": [],
                },
            },
        }


def generate_level(difficulty_str: str, length: int, seed: int):
    difficulty = resolve_difficulty(difficulty_str)
    rng = random.Random(seed)
    gen = LevelGenerator(difficulty, length, rng)
    return gen.generate()


def validate_level(data: dict):
    """生成結果が最低限の安全条件を満たしているか検証する"""
    length = data["length"]
    ground_xs = {}
    for x, y in data["level"]["objects"]["ground"]:
        ground_xs.setdefault(x, []).append(y)

    def has_ground(x):
        return x in ground_xs and GROUND_ROW in ground_xs[x]

    # スポーン地点の安全確認
    for x in range(SPAWN_SAFE_TILES):
        assert has_ground(x), f"スポーン地点 x={x} に地面がありません"

    # ゴール手前の安全確認
    for x in range(length - GOAL_SAFE_TILES, length):
        assert has_ground(x), f"ゴール手前 x={x} に地面がありません"

    # ギャップ幅の確認（連続して地面が無い区間が安全範囲内か）
    gap_run = 0
    for x in range(length):
        if has_ground(x):
            gap_run = 0
        else:
            gap_run += 1
            assert gap_run <= MAX_SAFE_GAP_TILES, f"x={x} 付近のギャップが安全範囲({MAX_SAFE_GAP_TILES}タイル)を超えています"

    # パイプの高さ確認
    for px, top_y, plen in data["level"]["objects"]["pipe"]:
        height = GROUND_ROW - top_y
        assert height <= MAX_SAFE_PIPE_TILES, f"x={px} のパイプが安全範囲({MAX_SAFE_PIPE_TILES}タイル)を超えています"

    return True


def main():
    parser = argparse.ArgumentParser(
        description="難易度指定でマリオのステージ JSON を自動生成する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python generate_level.py --difficulty easy --output Level_easy_01
  python generate_level.py --difficulty 0.7 --seed 42 --output Level_hard_01
  python generate_level.py --difficulty medium --count 5 --output-prefix Level_medium
        """,
    )
    parser.add_argument("--difficulty", type=str, default="medium",
                         help="easy/medium/hard または 0.0〜1.0 (デフォルト: medium)")
    parser.add_argument("--length", type=int, default=194, help="レベルの長さ（タイル数）")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード")
    parser.add_argument("--output", type=str, default=None,
                         help="出力ファイル名（拡張子・levels/ プレフィックスなし）")
    parser.add_argument("--output-prefix", type=str, default="Level_generated",
                         help="--count 指定時のファイル名プレフィックス")
    parser.add_argument("--count", type=int, default=1, help="生成するレベル数")
    parser.add_argument("--levels-dir", type=str, default="./levels")

    args = parser.parse_args()
    os.makedirs(args.levels_dir, exist_ok=True)

    for i in range(args.count):
        seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
        if args.count > 1 or args.seed is None:
            seed = seed + i
        data = generate_level(args.difficulty, args.length, seed)
        validate_level(data)

        if args.output and args.count == 1:
            name = args.output
        else:
            name = f"{args.output_prefix}_{i+1:02d}"

        out_path = os.path.join(args.levels_dir, f"{name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        n_gaps = len(data["level"]["objects"]["pipe"])
        n_enemies = len(data["level"]["entities"]["Goomba"]) + len(data["level"]["entities"]["Koopa"])
        print(f"✅ 生成完了: {out_path} (difficulty={args.difficulty}, seed={seed}, "
              f"pipes={n_gaps}, enemies={n_enemies})")


if __name__ == "__main__":
    main()
