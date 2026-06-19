import random

class LevelGenerator:
    def __init__(self, length=100):
        self.length = length
        self.height = 15 # Level.pyの描画範囲やLayers設定に基づく高さ

    def generate(self, difficulty: float, output_filename="generated_level.txt"):
        """
        difficulty: 0.0 (簡単) 〜 1.0 (激ムズ)
        """
        # 難易度に基づく確率設定
        gap_prob = 0.02 + (difficulty * 0.15)       # 穴の確率
        enemy_prob = 0.05 + (difficulty * 0.15)     # 敵の確率（地面がある場合）
        pipe_prob = 0.02 + (difficulty * 0.05)      # 土管の確率
        block_prob = 0.15 - (difficulty * 0.08)     # ハテナブロックやレンガの確率（難しいほど減る）

        # 15行 × length列 の空マップを初期化
        # text_to_json.py では '-' が空白として扱われます
        level_map = [['-' for _ in range(self.length)] for _ in range(self.height)]

        for x in range(self.length):
            # 最初の5マスと最後の5マスは安全地帯として穴を開けない
            is_safe_zone = (x < 5) or (x > self.length - 5)

            # 1. 地面と穴の生成
            if is_safe_zone or random.random() > gap_prob:
                # text_to_json.py では 'G' が地面
                level_map[13][x] = 'G'
                level_map[14][x] = 'G'
                
                # 地面がある場合のみ障害物や敵を配置可能にする
                if not is_safe_zone:
                    # 2. 土管の生成 (p)
                    if random.random() < pipe_prob and x < self.length - 10:
                        level_map[12][x] = 'p'
                        level_map[11][x] = 'p'
                    
                    # 3. 敵の生成 (g:クリボー, k:ノコノコ)
                    elif random.random() < enemy_prob:
                        enemy_type = 'k' if random.random() < difficulty else 'g'
                        level_map[12][x] = enemy_type
            else:
                # 連続した穴が長すぎないようにする制限などのロジックを入れるとより良くなります
                pass

            # 4. ブロックとアイテムの生成 (高さ y=8 または 9付近)
            if not is_safe_zone and random.random() < block_prob:
                block_type = random.choice(['?', 'b', 'b', 'M', 'h']) 
                level_map[9][x] = block_type

        # テキストファイルとして出力
        with open(output_filename, 'w') as f:
            for row in level_map:
                f.write("".join(row) + "\n")
        
        print(f"難易度 {difficulty:.1f} のステージを {output_filename} に生成しました。")

# --- 実行用コード ---
if __name__ == "__main__":
    generator = LevelGenerator(length=150)
    
    # 簡単なステージ (難易度0.1)
    generator.generate(difficulty=0.1, output_filename="level_easy.txt")
    
    # 難しいステージ (難易度0.9)
    generator.generate(difficulty=0.9, output_filename="level_hard.txt")