import json

from classes.Animation import Animation
from classes.Sprite import Sprite
from classes.Spritesheet import Spritesheet


class Sprites:
    # プロセス内で一度読み込んだスプライト集合をキャッシュする。
    # Level は reset() のたびに Sprites() を新規生成するため、キャッシュが無いと
    # エピソードごとにディスクI/O・画像デコード・Surface生成が繰り返され、
    # 長時間学習で顕著な速度低下（メモリ/GC負荷の蓄積）を引き起こす。
    _cached_collection = None

    def __init__(self):
        if Sprites._cached_collection is None:
            Sprites._cached_collection = self.loadSprites(
                [
                    "./sprites/Mario.json",
                    "./sprites/Goomba.json",
                    "./sprites/Koopa.json",
                    "./sprites/Animations.json",
                    "./sprites/BackgroundSprites.json",
                    "./sprites/ItemAnimations.json",
                    "./sprites/RedMushroom.json"
                ]
            )
        self.spriteCollection = Sprites._cached_collection

    def loadSprites(self, urlList):
        resDict = {}
        for url in urlList:
            with open(url) as jsonData:
                data = json.load(jsonData)
                mySpritesheet = Spritesheet(data["spriteSheetURL"])
                dic = {}
                if data["type"] == "background":
                    for sprite in data["sprites"]:
                        try:
                            colorkey = sprite["colorKey"]
                        except KeyError:
                            colorkey = None
                        dic[sprite["name"]] = Sprite(
                            mySpritesheet.image_at(
                                sprite["x"],
                                sprite["y"],
                                sprite["scalefactor"],
                                colorkey,
                            ),
                            sprite["collision"],
                            None,
                            sprite["redrawBg"],
                        )
                    resDict.update(dic)
                    continue
                elif data["type"] == "animation":
                    for sprite in data["sprites"]:
                        images = []
                        for image in sprite["images"]:
                            images.append(
                                mySpritesheet.image_at(
                                    image["x"],
                                    image["y"],
                                    image["scale"],
                                    colorkey=sprite["colorKey"],
                                )
                            )
                        dic[sprite["name"]] = Sprite(
                            None,
                            None,
                            animation=Animation(images, deltaTime=sprite["deltaTime"]),
                        )
                    resDict.update(dic)
                    continue
                elif data["type"] == "character" or data["type"] == "item":
                    for sprite in data["sprites"]:
                        try:
                            colorkey = sprite["colorKey"]
                        except KeyError:
                            colorkey = None
                        try:
                            xSize = sprite['xsize']
                            ySize = sprite['ysize']
                        except KeyError:
                            xSize, ySize = data['size']
                        dic[sprite["name"]] = Sprite(
                            mySpritesheet.image_at(
                                sprite["x"],
                                sprite["y"],
                                sprite["scalefactor"],
                                colorkey,
                                True,
                                xTileSize=xSize,
                                yTileSize=ySize,
                            ),
                            sprite["collision"],
                        )
                    resDict.update(dic)
                    continue
        return resDict
