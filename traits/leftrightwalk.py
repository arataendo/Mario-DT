import random

from classes.Collider import Collider


class LeftRightWalkTrait:
    def __init__(self, entity, level):
        # 初期の向きは MarioEnv が Level に持たせた環境ごとの乱数で決める。
        # 以前はグローバルな random を使っていたため、同じプロセスで複数の環境を
        # 動かすと互いの敵の向きが変わり、同じステージでも評価の順番次第で
        # 結果が変わっていた。rng が無い場合（ゲーム単体で遊ぶ時など）は従来どおり。
        rng = getattr(level, "rng", None) or random
        self.direction = rng.choice([-1, 1])
        self.entity = entity
        self.collDetection = Collider(self.entity, level)
        self.speed = 1
        self.entity.vel.x = self.speed * self.direction

    def update(self):
        if self.entity.vel.x == 0:
            self.direction *= -1
        self.entity.vel.x = self.speed * self.direction
        self.moveEntity()

    def moveEntity(self):
        self.entity.rect.y += self.entity.vel.y
        self.collDetection.checkY()
        self.entity.rect.x += self.entity.vel.x
        self.collDetection.checkX()
