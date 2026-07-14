from classes.Maths import Vec2D


class Camera:
    def __init__(self, pos, entity):
        self.pos = Vec2D(pos.x, pos.y)
        self.entity = entity
        self.x = self.pos.x * 32
        self.y = self.pos.y * 32

    def move(self):
        xPosFloat = self.entity.getPosIndexAsFloat().x
        
        # マリオがいるステージの全マス目（長さ）を取得
        level_length = self.entity.levelObj.levelLength
        
        # ステージの長さから10マス引いた場所までカメラが追従するようにする
        if 10 < xPosFloat < (level_length - 10):
            self.pos.x = -xPosFloat + 10
            
        self.x = self.pos.x * 32
        self.y = self.pos.y * 32