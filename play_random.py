import os
import sys

# 基準フォルダの設定
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)

import pygame
pygame.mixer.pre_init(44100, -16, 2, 4096)
pygame.init()
pygame.mixer.init()

from classes.MarioGymEnv import MarioEnv

def play_random():
    # 画面を表示するモードで環境を起動
    env = MarioEnv(render_mode='human')
    obs, info = env.reset()
    
    print("ランダムエージェントのプレイを開始します！")
    
    done = False
    step = 0
    while not done:
        pygame.event.pump() # フリーズ防止
        
        # 0〜7のアクション（待機、左右、ジャンプなど）をランダムに選択
        action = env.action_space.sample() 
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        step += 1
        if step % 30 == 0:
            print(f"Step {step} | ランダムアクション: {action}")

    print("Game Over!")
    env.close()

if __name__ == "__main__":
    play_random()