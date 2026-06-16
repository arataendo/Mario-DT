import math
import pygame
from entities.EntityBase import EntityBase

class Fireball(EntityBase):
    def __init__(self, pixel_x, pixel_y, screen, image):
        # 親クラスの初期化では座標が32倍されてしまうため、一旦0,0で初期化します
        super(Fireball, self).__init__(0, 0, 0)
        # ピクセル座標で上書きし、サイズを16x16とします
        self.rect = pygame.Rect(pixel_x, pixel_y, 16, 16)
        self.screen = screen
        self.image = image
        # Mario.py側でダメージ判定を行うためのタイプ名を設定します
        self.type = "obstacle"
        self.obeyGravity = False

    def update(self, camera):
        # 描画のみ行います（座標の更新は親のFirebarが行うため）
        if self.image and self.screen is not None:
            self.screen.blit(self.image, (self.rect.x + camera.x, self.rect.y))

class Firebar(EntityBase):
    def __init__(self, screen, spriteColl, x, y, level, length=6, speed=2.0):
        # x, y はタイルのグリッド座標です
        super(Firebar, self).__init__(x, y, 0)
        self.screen = screen
        self.levelObj = level
        self.length = length    # ファイアボールの数
        self.speed = speed      # 回転スピード
        self.angle = 0.0
        self.obeyGravity = False

        # 中心座標をブロックの中心に設定します (32x32のタイルの中心)
        self.center_x = x * 32 + 8
        self.center_y = y * 32 + 8

        # スプライトから画像を取得します（用意されていない場合は一時的な赤い四角形を使用）
        try:
            self.fireball_image = spriteColl.get("fireball").image
        except AttributeError:
            self.fireball_image = pygame.Surface((16, 16))
            self.fireball_image.fill((255, 50, 50))

        self.fireballs = []
        for i in range(self.length):
            fb = Fireball(self.center_x, self.center_y, screen, self.fireball_image)
            self.fireballs.append(fb)
            # LevelのentityListに追加することで、自動的に描画とMarioとの当たり判定が行われます
            level.entityList.append(fb)

    def update(self, camera):
        # 角度の更新
        self.angle += self.speed
        if self.angle >= 360:
            self.angle -= 360

        rad = math.radians(self.angle)

        # 各ファイアボールの座標を円運動の計算に基づいて更新します
        for i, fb in enumerate(self.fireballs):
            # 中心からの距離 (1つあたり16ピクセル間隔)
            distance = (i + 1) * 16
            fb.rect.x = self.center_x + math.cos(rad) * distance
            fb.rect.y = self.center_y + math.sin(rad) * distance