"""
Gym エージェント用の入力クラス
Gym のアクションを Mario のトレイト入力に変換する
"""


class AgentInput:
    """
    アクションマッピング:
    0: NOP (何もしない)
    1: Left (左移動)
    2: Right (右移動)
    3: Jump (ジャンプ)
    4: Left + Jump
    5: Right + Jump
    6: Dash (ダッシュ/スプリント)
    7: Right + Dash
    """
    
    def __init__(self, entity):
        self.entity = entity
        self.current_action = 0
    
    def setAction(self, action):
        """Gym エージェントからのアクションを設定"""
        self.current_action = action
        self._apply_action()
    
    def _apply_action(self):
        """アクションを Mario のトレイトに適用"""
        action = self.current_action
        
        # 移動方向とジャンプ、ダッシュを一旦リセット
        self.entity.traits["goTrait"].direction = 0
        self.entity.traits["goTrait"].boost = False
        self.entity.traits['jumpTrait'].jump(False)
        
        # --- データセット仕様(0~255のビットマップ)に基づくボタン判定 ---
        btn_A     = bool(action & 128)  # ジャンプ
        btn_up    = bool(action & 64)   # 上 (今回は使用しない)
        btn_left  = bool(action & 32)   # 左
        btn_B     = bool(action & 16)   # ダッシュ
        btn_start = bool(action & 8)    # スタート
        btn_right = bool(action & 4)    # 右
        btn_down  = bool(action & 2)    # 下 (土管など)
        btn_select= bool(action & 1)    # セレクト
        
        # 1. 左右の移動 (左右同時押しの場合は相殺して動かないようにする)
        if btn_left and not btn_right:
            self.entity.traits["goTrait"].direction = -1
        elif btn_right and not btn_left:
            self.entity.traits["goTrait"].direction = 1
            
        # 2. ダッシュ (Bボタン)
        if btn_B:
            self.entity.traits["goTrait"].boost = True
            
        # 3. ジャンプ (Aボタン)
        if btn_A:
            self.entity.traits['jumpTrait'].jump(True)
            
        # ※ もし自作マリオ側に「しゃがむ」や「土管に入る」機能があれば
        # if btn_down:
        #     ... のように追加可能です
    def checkForInput(self):
        """checkForInput インターフェースの互換性のため（何もしない）"""
        pass
