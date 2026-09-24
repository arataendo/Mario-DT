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
    8: Right + Dash + Jump (走りジャンプ。幅の広い穴はこれでないと越えられない)
    9: Left + Dash
    """

    N_ACTIONS = 10
    
    def __init__(self, entity):
        self.entity = entity
        self.current_action = 0
    
    def setAction(self, action):
        """Gym エージェントからのアクションを設定"""
        self.current_action = action
        self._apply_action()
    
    def _apply_action(self):
        """アクションを Mario のトレイトに適用"""
        action = int(self.current_action)

        # 移動方向とジャンプ、ダッシュを一旦リセット
        self.entity.traits["goTrait"].direction = 0
        self.entity.traits["goTrait"].boost = False
        self.entity.traits['jumpTrait'].jump(False)

        # PPO のアクション空間は Discrete(10) なので、0〜9 の整数として解釈する
        if action == 1:
            self.entity.traits["goTrait"].direction = -1
        elif action == 2:
            self.entity.traits["goTrait"].direction = 1
        elif action == 3:
            self.entity.traits['jumpTrait'].jump(True)
        elif action == 4:
            self.entity.traits["goTrait"].direction = -1
            self.entity.traits['jumpTrait'].jump(True)
        elif action == 5:
            self.entity.traits["goTrait"].direction = 1
            self.entity.traits['jumpTrait'].jump(True)
        elif action == 6:
            self.entity.traits["goTrait"].boost = True
        elif action == 7:
            self.entity.traits["goTrait"].direction = 1
            self.entity.traits["goTrait"].boost = True
        elif action == 8:
            # 走りジャンプ: 幅の広い穴を越えるために必須
            self.entity.traits["goTrait"].direction = 1
            self.entity.traits["goTrait"].boost = True
            self.entity.traits['jumpTrait'].jump(True)
        elif action == 9:
            self.entity.traits["goTrait"].direction = -1
            self.entity.traits["goTrait"].boost = True

        # 0: NOP (何もしない)
    def checkForInput(self):
        """checkForInput インターフェースの互換性のため（何もしない）"""
        pass
