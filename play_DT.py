import os
import sys
import torch
import numpy as np
from PIL import Image

# 1. 【先に】マリオのゲーム本体があるフォルダに移動してパスを通す
GAME_DIR = r"C:\Users\arata\esslab\Mario-DT"
if os.path.exists(GAME_DIR):
    os.chdir(GAME_DIR)         
    sys.path.append(GAME_DIR)  
else:
    print(f"エラー: {GAME_DIR} が見つかりません。")

# 2. 【後から】自作モジュールをインポートする
from classes.MarioGymEnv import MarioEnv 
from make_DT import DecisionTransformer

def play_mario(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. モデルの準備
    model = DecisionTransformer(action_vocab_size=256, hidden_size=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. 環境の準備 (描画モードをhumanにして画面を見れるようにする)
    # level 引数にプレイしたいステージ名を文字列で指定します
    env = MarioEnv(level='Level_custom', render_mode='human')
    obs, info = env.reset()
    
    # 3. Decision Transformer用の履歴バッファ
    context_len = 30
    target_return = 300.0  # ★「スコア300(クリア)を目指せ」という指示
    
    states = []
    actions = []
    rtgs = [target_return]
    
    # 画像の前処理関数
    def preprocess_image(img_array):
        img = Image.fromarray(img_array.transpose(1, 2, 0)) # CHW -> HWC
        img = img.resize((84, 84), Image.BILINEAR)
        img_arr = np.array(img, dtype=np.float32) / 255.0
        return img_arr.transpose(2, 0, 1) # HWC -> CHW

    # ========= 前略 (変数の準備など) =========
    print("AIマリオのプレイを開始します！ターミナルの出力を確認してください...")
    
    # ゲームループ
    done = False
    step_count = 0
    while not done:
        # ★追加1: Pygameのウィンドウフリーズを防止する
        import pygame
        pygame.event.pump()
        
        # 現在の画像を処理して履歴に追加
        state = preprocess_image(obs['image'])
        states.append(state)
        
        # 履歴をコンテキスト長(K=30)に切り詰める
        states_input = torch.tensor(np.array(states[-context_len:]), dtype=torch.float32).unsqueeze(0).to(device)
        
        # ====== 修正ポイント ======
        # 未知の巨大な数字でAIがパニックにならないよう、
        # 学習時と同じ「最大値(1.0)」のテンソルを固定で渡し、「常に最高のプレイをしろ」と指示します。
        rtgs_input = torch.ones((1, states_input.shape[1], 1), dtype=torch.float32).to(device)*0.6
        # ==========================
        if len(actions) == 0:
            actions_input = torch.zeros((1, 1), dtype=torch.long).to(device)
        else:
            acts = actions[-(context_len-1):] + [0] 
            actions_input = torch.tensor(np.array(acts), dtype=torch.long).unsqueeze(0).to(device)
            
        timesteps = torch.arange(0, states_input.shape[1], dtype=torch.long).unsqueeze(0).to(device)
        
        # モデルによるアクション予測
        with torch.no_grad():
            action_logits = model(states_input, actions_input, rtgs_input, timesteps)
            action_pred = torch.argmax(action_logits[0, -1]).item()
            
        # ★追加2: AIの思考状況をターミナルに表示
        print(f"Step {step_count} | 予測アクション: {action_pred} | 残り目標スコア: {rtgs[-1]}")
        
        actions.append(action_pred)
        
        # 環境を1ステップ進める
        obs, reward, terminated, truncated, info = env.step(action_pred)
        done = terminated or truncated
        
        rtgs.append(rtgs[-1] - reward)
        step_count += 1
    
    print(f"Game Over! Final RTG left: {rtgs[-1]}")
    env.close()

if __name__ == "__main__":
    # 学習した重みファイルがある場所の絶対パスを指定する
    # （※ 以下のパスは実際に .pth がある場所に合わせて書き換えてください）
    model_weight_path = r"C:\Users\arata\esslab\Mario-DT\mario_dt_epoch_51.pth" 
    
    play_mario(model_weight_path)