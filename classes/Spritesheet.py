import os
import pygame
from PIL import Image  # ★追加: Pillowを使って画像を読み込む

class Spritesheet(object):
    def __init__(self, filename):
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            abs_path = os.path.normpath(os.path.join(base_dir, filename)).replace('\\', '/')
            
            # ---------------------------------------------------------
            # ★修正: Pygameの壊れた読み込み機能を捨て、Pillowで読み込む
            # ---------------------------------------------------------
            # 1. Pillowで画像を開き、RGBA(透過あり)形式に変換
            pil_image = Image.open(abs_path).convert("RGBA")
            
            # 2. 画像データをバイナリ文字列に変換
            image_data = pil_image.tobytes()
            
            # 3. バイナリデータからPygameのSurface（画像）を直接生成する
            self.sheet = pygame.image.frombytes(image_data, pil_image.size, "RGBA")
            # ---------------------------------------------------------

            if not self.sheet.get_alpha():
                self.sheet.set_colorkey((0, 0, 0))
                
        except Exception as e:
            print("画像読み込みエラー:", filename)
            print("詳細:", e)
            raise SystemExit

    def image_at(self, x, y, scalingfactor, colorkey=None, ignoreTileSize=False, xTileSize=16, yTileSize=16):
        # （ここから下の行は一切変更なし）
        if ignoreTileSize:
            rect = pygame.Rect((x, y, xTileSize, yTileSize))
        else:
            rect = pygame.Rect((x * xTileSize, y * yTileSize, xTileSize, yTileSize))
        image = pygame.Surface(rect.size, pygame.SRCALPHA) # 背景透過を維持
        image.blit(self.sheet, (0, 0), rect)
        
        if colorkey is not None:
            if colorkey == -1:
                colorkey = image.get_at((0, 0))
            image.set_colorkey(colorkey, pygame.RLEACCEL)
            
        return pygame.transform.scale(
            image, (xTileSize * scalingfactor, yTileSize * scalingfactor)
        )