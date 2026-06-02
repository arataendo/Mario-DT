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
        
        # 移動方向とジャンプ、ダッシュをリセット
        self.entity.traits["goTrait"].direction = 0
        self.entity.traits["goTrait"].boost = False
        self.entity.traits['jumpTrait'].jump(False)
        
        # アクションをデコード
        if action == 0:  # NOP
            pass
        elif action == 1:  # Left
            self.entity.traits["goTrait"].direction = -1
        elif action == 2:  # Right
            self.entity.traits["goTrait"].direction = 1
        elif action == 3:  # Jump
            self.entity.traits['jumpTrait'].jump(True)
        elif action == 4:  # Left + Jump
            self.entity.traits["goTrait"].direction = -1
            self.entity.traits['jumpTrait'].jump(True)
        elif action == 5:  # Right + Jump
            self.entity.traits["goTrait"].direction = 1
            self.entity.traits['jumpTrait'].jump(True)
        elif action == 6:  # Dash
            self.entity.traits["goTrait"].boost = True
        elif action == 7:  # Right + Dash
            self.entity.traits["goTrait"].direction = 1
            self.entity.traits["goTrait"].boost = True
    
    def checkForInput(self):
        """checkForInput インターフェースの互換性のため（何もしない）"""
        pass
