import cv2
import numpy as np
import os

# 基本設定
ORIGINAL_TILE_SIZE = 16  # スプライトシート上の1マスのサイズ（16x16）

# スクリーンショット上の1マスのサイズ（長方形に対応）
GRID_W = 56  # 横幅
GRID_H = 48  # 縦幅

# スプライトシートと座標の定義
# フォーマット: '出力文字': ('画像ファイル名', スプライトシート上のX座標(マス目), Y座標(マス目))
SPRITE_DEFS = {
    'G': ('img/tiles.png', 0, 0),   # 地面
    'b': ('img/tiles.png', 1, 0),   # レンガブロック
    'B': ('img/tiles.png', 11, 11),   #bush 
    'h': ('img/tiles.png', 0, 1),   # hardblock
    '?': ('img/tiles.png', 24, 0),  # ハテナブロック
    'c': ('img/Items.png', 0, 0),   # コイン
    'g': ('img/koopas.png', 0, 1),  # クリボー
    'p': ('img/tiles.png', 0, 11),  # パイプ
    # 必要に応じて追加してください
}

def extract_and_resize_sprite(sheet_image, grid_x, grid_y, tile_size, target_w, target_h):
    """
    スプライトシートから指定座標の画像を切り出し、
    スクリーンショットのブロックサイズ（今回は長方形）に合わせて引き伸ばす関数
    """
    x_start = grid_x * tile_size
    y_start = grid_y * tile_size
    
    # 1マス分(16x16)を切り出す
    sprite = sheet_image[y_start:y_start + tile_size, x_start:x_start + tile_size]
    
    if sprite.shape[0] != tile_size or sprite.shape[1] != tile_size:
        return None

    # 用意した長方形のサイズ（53x48）にリサイズ
    scaled_sprite = cv2.resize(sprite, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return scaled_sprite

def calculate_difference(img1, img2):
    """2つの画像の差を計算。数値が小さいほど似ている"""
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    return err

def image_to_text(screenshot_path, output_txt_path):
    img = cv2.imread(screenshot_path)
    if img is None:
        print(f"❌ エラー: スクリーンショット '{screenshot_path}' が読み込めません。")
        return

    height, width, _ = img.shape
    cols = width // GRID_W
    rows = height // GRID_H

    # スプライトシートから見本画像を生成してキャッシュ
    loaded_sheets = {}
    templates = {}

    for char, (filename, x, y) in SPRITE_DEFS.items():
        if filename not in loaded_sheets:
            if os.path.exists(filename):
                loaded_sheets[filename] = cv2.imread(filename)
            else:
                print(f"⚠️ 警告: スプライトシート '{filename}' が見つかりません。")
                continue
        
        sheet_img = loaded_sheets[filename]
        # ここで長方形のサイズ（GRID_W, GRID_H）を指定して引き伸ばす
        sprite = extract_and_resize_sprite(sheet_img, x, y, ORIGINAL_TILE_SIZE, GRID_W, GRID_H)
        
        if sprite is not None:
            templates[char] = sprite
        else:
            print(f"⚠️ 警告: '{char}' の切り出し座標が画像サイズを超えています。")

    output_lines = []

    # 画像をマス目ごとに走査
    for y in range(rows):
        row_chars = []
        for x in range(cols):
            x_start = x * GRID_W
            y_start = y * GRID_H
            x_end = x_start + GRID_W
            y_end = y_start + GRID_H
            
            # 長方形（53x48）で切り取る
            cell = img[y_start:y_end, x_start:x_end]
            
            best_char = '-'
            best_score = float('inf')

            for char, tmpl_img in templates.items():
                score = calculate_difference(cell, tmpl_img)
                if score < best_score:
                    best_score = score
                    best_char = char

            # 判定の閾値（JPEG特有のモヤモヤがあるので、少し緩めに設定）
            THRESHOLD = 38000 
            if best_score > THRESHOLD:
                best_char = '-'

            row_chars.append(best_char)
        
        output_lines.append("".join(row_chars))

    with open(output_txt_path, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')
            
    print(f"✅ 画像を解析し、{output_txt_path} を生成しました！")

if __name__ == "__main__":
    image_to_text("stage_screenshot.png", "level_wide.txt")