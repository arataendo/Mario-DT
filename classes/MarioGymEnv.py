"""
OpenAI Gymnasium 環境 - マリオゲーム
機械学習エージェント向けのラッパー

Note: gym の後継である gymnasium を使用
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

# 音声デバイスの無い Linux サーバーで pygame が ALSA を探して警告・待機しないようにする。
# 効果音は classes/Sound.py で常に無効化しているので、ここで止めても挙動は変わらない。
# （映像側は render で窓を出したい場合があるので、各実行スクリプトで個別に設定している）
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
from classes.Dashboard import Dashboard
from classes.Level import Level
from classes.Sound import Sound
from entities.Mario import Mario
from classes.AgentInput import AgentInput


class MarioEnv(gym.Env):
    """
    OpenAI Gym v0.26+ 互換のマリオゲーム環境
    
    観測空間:
    - Dict({'image': Box(3, 640, 480), 'state': Box(9,)})
    
    アクション空間:
    - Discrete(10): [NOP, Left, Right, Jump, Left+Jump, Right+Jump,
                     Dash, Right+Dash, Right+Dash+Jump, Left+Dash]

    報酬関数（1 ゲームフレームあたり）:
    - 最高到達点(max_x)の更新分のみ: +Δタイル数  ← 往復稼ぎができない形にする
    - 時間ペナルティ: -0.01 / frame        ← 立ち止まりを不利にする
    - コイン: +1.0 / 枚
    - スコア増加(敵撃破など): +0.5 / 100pt
    - ゲームオーバー: -25.0
    - ステージクリア: +100.0

    ステージ全長 194 タイルを 4000 フレームで走破した場合の合計は概ね
    194 + 100 - 40 = +254、途中で死ぬと大きく下回る、という設計。
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", None],
        "render_fps": 60,
    }
    
    def __init__(self, level=None, random_level=True, render_mode=None, max_episode_steps=8000,
                 levels=None):
        """
        Parameters:
        -----------
        level : str or None
            レベルを指定 ('Level1-1', 'Level1-2' など)
            None の場合は random_level に従う
        
        random_level : bool
            level=None の場合、エピソード開始時にランダムにレベルを選択
        
        render_mode : str or None
            - 'human': pygame 画面を表示
            - 'rgb_array': 画像を RGB 配列で返す
            - None: 描画しない（最速訓練用）
        
        max_episode_steps : int
            エピソードの最大**ゲームフレーム**数（timeout で終了）。

            ⚠️ このカウンタは SkipFrame ラッパーの内側で加算されるため、
            SkipFrame(skip=4) を被せると
                エージェントの意思決定回数 = max_episode_steps / 4
            になる。人間が 4000 フレーム必要なステージで 1000 を指定すると
            ステージの 1/4 で必ず打ち切られ、ゴール報酬が学習データに
            一度も現れない（＝原理的にクリアを学習できない）ので注意。
        """
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.level_name = level
        self.random_level = random_level
        # levels: ランダム選択の候補をこの集合に限定する（None なら全レベル）。
        # 特定の難易度だけを重点的に学習させたいときに使う。
        self.allowed_levels = set(levels) if levels else None
        self.available_levels = self._get_available_levels()
        self.current_level_name = level if level else (self.available_levels[0] if self.available_levels else 'Level1-1')
        
        # ゲーム状態
        self.level = None
        self.mario = None
        self.screen = None
        self.dashboard = None
        self.sound = None
        self.clock = None
        
        # エピソード統計
        self.episode_step = 0
        self.episode_reward = 0.0
        self.initial_mario_x = 0
        self.prev_coins = 0
        self.prev_points = 0
        self.prev_enemies_killed = 0
        
        # 観測空間の定義
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255,
                shape=(3, 480, 640),  # CHW format (Gym standard for images)
                dtype=np.uint8
            ),
            'state': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(9,),  # [mario_x, mario_y, vel_x, vel_y, camera_x, progress, coins, enemies, powerup]
                dtype=np.float32
            )
        })
        
        # アクション空間の定義
        # 0: NOP, 1: Left, 2: Right, 3: Jump, 4: Left+Jump, 5: Right+Jump,
        # 6: Dash, 7: Right+Dash, 8: Right+Dash+Jump, 9: Left+Dash
        self.action_space = spaces.Discrete(AgentInput.N_ACTIONS)
        # pygame 初期化
        if self.render_mode is not None or self.render_mode == 'human':
            self._init_pygame()
    
    # Level1-2 は x=0〜39 に地面タイルが存在せず、スポーン地点(0,0)から
    # 行動に関係なく即座に穴に落下してゲームオーバーになる（レベルデータ側の不具合）。
    # レベルを修正するまでランダム選択の対象から除外する。
    _BROKEN_LEVELS = {'Level1-2'}

    # DT の汎化性能を測るための「未知ステージ」。学習で一度も見せてはいけないので
    # ランダム選択の対象から常に外す。評価時は level= で明示指定すれば読み込める。
    _HELDOUT_PREFIX = 'Level_test_'

    @classmethod
    def is_heldout(cls, level_name):
        return level_name.startswith(cls._HELDOUT_PREFIX)

    def _get_available_levels(self):
        """ランダム選択の対象になるレベルのリストを取得"""
        levels_dir = "./levels"
        levels = []
        if os.path.exists(levels_dir):
            for file in os.listdir(levels_dir):
                if file.endswith('.json'):
                    level_name = file.replace('.json', '')
                    if level_name in self._BROKEN_LEVELS:
                        continue
                    if self.is_heldout(level_name):
                        continue
                    # levels= で明示的に絞られている場合はそれ以外を除く
                    if self.allowed_levels is not None and level_name not in self.allowed_levels:
                        continue
                    levels.append(level_name)
        if not levels:
            raise ValueError(
                f"選択可能なレベルがありません (levels={self.allowed_levels})。"
                f"レベル名の綴りと levels/ の中身を確認してください。"
            )
        return sorted(levels)
    
    def _init_pygame(self):
        """pygame の初期化"""
        import pygame
        
        if not pygame.get_init():
            pygame.mixer.pre_init(44100, -16, 2, 4096)
            pygame.init()
        
        if self.render_mode == 'human':
            self.screen = pygame.display.set_mode((640, 480))
            pygame.display.set_caption("Mario Game - Training")
        else:
            # RGB array 用のサーフェス（描画のみ）
            self.screen = pygame.Surface((640, 480))
        
        self.clock = pygame.time.Clock()
    
    def _setup_game(self, level_name):
        """ゲームの初期設定"""
        self.dashboard = Dashboard("./img/font.png", 8, self.screen)
        # headless モードの場合は sound を無効化
        self.sound = Sound(enabled=(self.render_mode == 'human'))
        
        self.level = Level(self.screen, self.sound, self.dashboard)
        self.level.loadLevel(level_name)
        
        # Mario をエージェント入力モードで作成
        self.mario = Mario(0, 0, self.level, self.screen, self.dashboard, self.sound)
        self.mario.input = AgentInput(self.mario)
        self.mario.input_source = 'agent'
        
        self.initial_mario_x = self.mario.rect.x
        self.prev_mario_x = self.mario.rect.x  # ★ここを追加: 前回のX座標を保持
        self.max_mario_x = self.mario.rect.x   # そのエピソードでの最高到達点
        self.prev_coins = self.dashboard.coins
        self.prev_points = self.dashboard.points
        self.prev_enemies_killed = 0
        
    def reset(self, seed=None, options=None):
        """
        環境をリセット
        
        Returns:
        --------
        observation : Dict
        info : Dict
        """
        super().reset(seed=seed)
        
        if self.render_mode is None:
            self._init_pygame()
        
        # レベルを選択
        if self.level_name is not None:
            level_name = self.level_name
        elif self.random_level:
            level_name = self.np_random.choice(self.available_levels)
        else:
            level_name = self.available_levels[0]
        
        # 現在のレベルを記録
        self.current_level_name = level_name
        
        # ゲーム初期化
        self._setup_game(level_name)
        
        # エピソード統計をリセット
        self.episode_step = 0
        self.episode_reward = 0.0
        
        observation = self._get_observation()
        info = {
            'level': self.current_level_name,
            'episode': 0,
        }
        
        return observation, info
    
    def step(self, action):
        # --- 前半はそのまま ---
        self.mario.input.setAction(action)
        self.level.drawLevel(self.mario.camera)
        self.dashboard.update()
        self.mario.update()
        
        if self.render_mode == 'human':
            pygame.display.update()
            self.clock.tick(60)
        
        self.episode_step += 1
        
        # 報酬計算
        reward = self._calculate_reward()
        self.episode_reward += reward
        
        # 終了条件確認
        terminated = self.mario.restart  # ゲームオーバーまたはステージクリア
        truncated = self.episode_step >= self.max_episode_steps  # タイムアウト
        
        observation = self._get_observation()
        info = {
            'level': self.current_level_name,
            'episode_step': self.episode_step,
            'cumulative_reward': self.episode_reward,
            'mario_x': self.mario.rect.x,
            'mario_y': self.mario.rect.y,
            'max_x': self.max_mario_x,
            'progress': self.max_mario_x / max(1, self.level.levelLength * 32),
            'coins': self.dashboard.coins,
            'points': self.dashboard.points,
            'powerup_state': self.mario.powerUpState,
        }
        
        # --- ここから修正 ---
        if terminated:
            # goalReached フラグでクリアかどうかを確実に判定する
            if hasattr(self.mario, 'goalReached') and self.mario.goalReached:
                info['reason'] = 'level_complete'
            else:
                info['reason'] = 'game_over'
        
        if truncated:
            info['reason'] = 'timeout'
        # --------------------
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self):
        """報酬を計算

        設計方針:
          * 前進報酬は「最高到達点(max_x)の更新分」だけに与える。
            毎フレームの delta_x に報酬を与えると左右に往復して稼げてしまい、
            逆に後退へ強いペナルティを課すと穴の前での助走ができなくなる。
          * 位置に依存する定常報酬（旧: progress_ratio * 0.25）は入れない。
            時間ペナルティを上回る定常プラス報酬があると
            「そこそこ進んだ地点でひたすら待つ」のが最適解になってしまう。
          * 高さボーナス（旧: y < 200 で +0.01）も入れない。無意味な
            ジャンプ連打を助長するだけだった。
        """
        reward = 0.0

        current_x = self.mario.rect.x
        current_coins = self.dashboard.coins
        current_points = self.dashboard.points

        # 1. 最高到達点の更新分だけを前進報酬にする（単位: タイル）
        if current_x > self.max_mario_x:
            reward += (current_x - self.max_mario_x) / 32.0
            self.max_mario_x = current_x

        # 2. 時間ペナルティ（立ち止まり・引き返しを不利にする唯一の項）
        reward -= 0.01

        # 3. コイン取得
        coins_delta = current_coins - self.prev_coins
        if coins_delta > 0:
            reward += coins_delta * 1.0

        # 4. スコア増加（敵撃破・アイテム等）
        if current_points > self.prev_points:
            reward += (current_points - self.prev_points) / 100.0 * 0.5

        # 5. エピソード終了時の報酬/ペナルティ
        if self.mario.restart:
            if getattr(self.mario, 'goalReached', False):
                reward += 100.0
            else:
                reward -= 25.0

        # 次ステップ計算のために保存
        self.prev_mario_x = current_x
        self.prev_points = current_points
        self.prev_coins = current_coins

        return reward
    
    def _get_observation(self):
        """現在の観測を取得"""
        # 画像部分
        image = self._get_screen_image()
        
        # 状態ベクトル
        state = np.array([
            self.mario.rect.x / 32.0,  # Mario X 座標 (正規化)
            self.mario.rect.y / 32.0,  # Mario Y 座標
            self.mario.vel.x,          # 速度 X
            self.mario.vel.y,          # 速度 Y
            self.mario.camera.pos.x,   # カメラ位置
            self.mario.rect.x / self.level.levelLength / 32,  # 進行度 (0-1)
            self.dashboard.coins,      # コイン数
            len(self.level.entityList),  # 敵数（近似）
            self.mario.powerUpState,   # パワーアップ状態
        ], dtype=np.float32)
        
        observation = {
            'image': image,
            'state': state
        }
        
        return observation
    
    def _get_screen_image(self):
        """スクリーン画像を numpy 配列 (CHW) で取得"""
        import pygame
        
        if self.screen is None:
            raise RuntimeError("Screen not initialized")
        
        # pygame surface を numpy 配列に変換 (RGB, HWC)
        surf_array = pygame.surfarray.array3d(self.screen)  # shape: (W, H, 3)
        
        # HWC から CHW に変換
        image = np.transpose(surf_array, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        
        # 注: pygame.surfarray は (W, H, C) を返すため、軸を入れ替える必要がある
        image = np.transpose(image, (0, 2, 1))  # (C, W, H) -> (C, H, W)
        
        return image.astype(np.uint8)
    
    def render(self):
        """render_mode に応じて描画"""
        if self.render_mode == 'human':
            pygame.display.update()
            self.clock.tick(60)
        elif self.render_mode == 'rgb_array':
            return self._get_screen_image()
        
        return None
    
    def close(self):
        """環境をクローズ"""
        if self.screen is not None:
            import pygame
            pygame.quit()
    
    def seed(self, seed=None):
        """Random seed を設定"""
        super().reset(seed=seed)
        return [seed]
    
    @property
    def spec(self):
        """Gym spec 情報"""
        return gym.envs.registration.EnvSpec(
            id='Mario-v0',
            entry_point=__name__ + ':MarioEnv',
            max_episode_steps=self.max_episode_steps,
        )
