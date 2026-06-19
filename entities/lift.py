import pygame
from entities.EntityBase import EntityBase

class Lift(EntityBase):
    def __init__(self, screen, spriteColl, x, y, level, width_tiles=3, move_x=0, move_y=4, speed=1.0):
        # x, y はタイルのグリッド座標です
        super(Lift, self).__init__(x, y, 0)
        self.screen = screen
        self.levelObj = level
        self.type = "Lift"          # マリオ側で判定するためのタイプ名
        self.obeyGravity = False    # 重力の影響を受けないようにする

        # リフトのサイズ（幅は引数で指定されたタイル数×32、高さは16ピクセルなど薄めに設定）
        self.rect = pygame.Rect(x * 32, y * 32, width_tiles * 32, 16)

        # スプライト画像の取得（用意されていない場合は一時的な青いブロックを描画）
        try:
            self.image = spriteColl.get("lift").image
        except AttributeError:
            self.image = pygame.Surface((width_tiles * 32, 16))
            self.image.fill((0, 150, 255)) # リフトの色（青）

        # 移動の基準点と、移動範囲（ピクセル単位）を計算
        self.start_x = self.rect.x
        self.start_y = self.rect.y
        self.move_range_x = move_x * 32
        self.move_range_y = move_y * 32
        
        # 速度を設定
        self.vel.x = speed if move_x != 0 else 0
        self.vel.y = speed if move_y != 0 else 0

    def update(self, camera):
        # リフトの座標を更新
        self.rect.x += self.vel.x
        self.rect.y += self.vel.y
        
        # X方向の移動範囲を超えたら反転
        if self.move_range_x != 0:
            if abs(self.rect.x - self.start_x) >= abs(self.move_range_x):
                self.vel.x *= -1
                
        # Y方向の移動範囲を超えたら反転
        if self.move_range_y != 0:
            if abs(self.rect.y - self.start_y) >= abs(self.move_range_y):
                self.vel.y *= -1

        self.drawLift(camera)

    def drawLift(self, camera):
        if self.screen is not None:
            self.screen.blit(self.image, (self.rect.x + camera.x, self.rect.y))