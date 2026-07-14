import json

def convert_text_to_json(txt_filepath, json_filepath):
    with open(txt_filepath, 'r') as f:
        lines = [line.strip('\n') for line in f.readlines()]

    height = len(lines)
    length = max(len(line) for line in lines) if lines else 0

    # JSONのベースとなるデータ構造
    data = {
        "id": 1,
        "length": length,
        "level": {
            "objects": {
                "bush": [],
                "sky": [],
                "cloud": [],
                "pipe": [],
                "ground": []
            },
            "layers": {
                "sky": {
                    "x": [0, length],
                    "y": [0, height - 2] # 変更: height(15) - 2 = 13 (空を13マス分作る)
                },
                "ground": {
                    "x": [0, length],
                    "y": [height - 1, height + 1] # 変更: 14から16未満 (地面を2マス分作る)
                }
            },
            "entities": {
                "CoinBox": [],
                "coinBrick": [],
                "coin": [],
                "hardblock": [],
                "Goomba": [],
                "Koopa": [],
                "RandomBox": [],
                "Firebar": []
            }
        }
    }

    # テキストを1文字ずつ走査して座標を取得
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char == '-':
                continue # 空白は無視
            elif char == 'G':
                data["level"]["objects"]["ground"].append([x, y])
            elif char == 'B':
                data["level"]["objects"]["bush"].append([x, y])
            elif char == 'C':
                data["level"]["objects"]["cloud"].append([x, y])
            elif char == 'P':
                # 左隣や上が 'P' なら無視する（左上の1箇所だけで綺麗な土管を1つ生成するため）
                is_top_left = True
                if x > 0 and lines[y][x-1] == 'P':
                    is_top_left = False
                if y > 0 and lines[y-1][x] == 'P':
                    is_top_left = False
                
                if is_top_left:
                    # 下に何ブロック連続しているか数えて、土管の正しい高さを測る
                    pipe_height = 0
                    ty = y
                    while ty < height and len(lines[ty]) > x and lines[ty][x] == 'P':
                        pipe_height += 1
                        ty += 1
                    data["level"]["objects"]["pipe"].append([x, y, pipe_height])
            elif char == '?':
                data["level"]["entities"]["CoinBox"].append([x, y])
            elif char == 'b':
                data["level"]["entities"]["coinBrick"].append([x, y])
            elif char == 'h':
                data["level"]["entities"]["hardblock"].append([x, y])
            elif char == 'c':
                data["level"]["entities"]["coin"].append([x, y])
            elif char == 'g':
                data["level"]["entities"]["Goomba"].append([x, y])
            elif char == 'k':
                data["level"]["entities"]["Koopa"].append([x, y])
            elif char == 'M':
                # 元の仕様に合わせて [x, y, item名] の形式にする
                data["level"]["entities"]["RandomBox"].append([x, y, "RedMushroom"])
            elif char == 's':
                data["level"]["entities"]["RandomBox"].append([x, y, "SuperStar"])
            elif char == 'F':
                data["level"]["entities"]["Firebar"].append([x, y])

    # JSONファイルとして出力
    with open(json_filepath, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"✅ {txt_filepath} を {json_filepath} に変換しました！")

# 実行
if __name__ == "__main__":
    # 読み込むテキストファイル名と、出力するJSONファイル名を指定
    convert_text_to_json("level.txt", "levels/Level_custom2.json")