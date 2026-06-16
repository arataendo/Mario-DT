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
                    "y": [0, height - 3] # 下から3行目より上を空と仮定
                },
                "ground": {
                    "x": [0, length],
                    "y": [height - 2, height] # 下から2行を地面と仮定
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
            elif char == 'p':
                # 元の仕様に合わせて [x, y, length] の形式にする（ここでは長さを2で固定）
                # 上のブロックを優先して登録し、下のpは無視するなどの工夫も可能ですが、簡易的に登録します。
                data["level"]["objects"]["pipe"].append([x, y, 2])
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
    convert_text_to_json("level.txt", "levels/Level_custom.json")