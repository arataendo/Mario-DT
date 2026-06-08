"""
OpenAI Gymnasium 環境 - マリオゲーム
機械学習エージェント向けのラッパー

Note: gym の後継である gymnasium を使用
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
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
    - Discrete(8): [NOP, Left, Right, Jump, Left+Jump, Right+Jump, Dash, Right+Dash]
    
    報酬関数:
    - 基本: 進行度 * 0.1
    - 時間ペナルティ: -0.01 per step
    - コイン: +1.0
    - 敵撃破: +5.0
    - ゲームオーバー: -10.0
    - ステージクリア: +100.0
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", None],
        "render_fps": 60,
    }
    
    def __init__(self, level=None, random_level=True, render_mode=None, max_episode_steps=10800):
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
            エピソードの最大ステップ数（timeout で終了）
        """
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.level_name = level
        self.random_level = random_level
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
        # 0: NOP, 1: Left, 2: Right, 3: Jump, 4: Left+Jump, 5: Right+Jump, 6: Dash, 7: Right+Dash
        self.action_space = spaces.Discrete(8)
        
        # pygame 初期化
        if self.render_mode is not None or self.render_mode == 'human':
            self._init_pygame()
    
    def _get_available_levels(self):
        """利用可能なレベルのリストを取得"""
        levels_dir = "./levels"
        levels = []
        if os.path.exists(levels_dir):
            for file in os.listdir(levels_dir):
                if file.endswith('.json'):
                    level_name = file.replace('.json', '')
                    levels.append(level_name)
        return sorted(levels) if levels else ['Level1-1']
    
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
        """報酬を計算"""
        reward = 0.0
        
        # ★修正: 進行度報酬 (1ステップでの右への移動距離に基づく)
        delta_x = self.mario.rect.x - getattr(self, 'prev_mario_x', self.initial_mario_x)
        
        # 1ブロック(32ピクセル)進むごとに報酬を与えるよう正規化
        # 左に戻った場合はペナルティ（マイナス報酬）になる
        reward += delta_x / 32.0  
        
        # 次のステップの計算のために現在のX座標を保存
        self.prev_mario_x = self.mario.rect.x
        
        # 時間ペナルティ
        reward -= 0.01
        
        # コイン収集ボーナス
        current_coins = self.dashboard.coins
                
        # ポイント増加ボーナス（敵撃破など）
        current_points = self.dashboard.points
        if current_points > self.prev_points:
            points_delta = current_points - self.prev_points
            reward += points_delta / 100.0  # 100ポイント = 1.0 報酬
            self.prev_points = current_points
        
        # --- ここから修正 ---
        if self.mario.restart:
            # goalReached フラグが True ならゴール到達！
            if hasattr(self.mario, 'goalReached') and self.mario.goalReached:
                reward += 100.0  # ★ ゴール報酬（必要に応じて大きくしてください）
            # それ以外（敵に当たった、穴に落ちた等）はゲームオーバーペナルティ
            else:
                reward -= 10.0
        # --------------------
        
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
