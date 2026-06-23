import torch
import numpy as np
from PIL import Image
from classes.MarioGymEnv import MarioEnv # あなたの環境に合わせてインポート

# ※ 先ほどの make_DT.py にある DecisionTransformer クラスの定義がここにも必要です
from make_DT import DecisionTransformer 

def play_mario(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. モデルの準備
    model = DecisionTransformer(action_vocab_size=256, hidden_size=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. 環境の準備 (描画モードをhumanにして画面を見れるようにする)
    env = MarioEnv(render_mode='human')
    obs, info = env.reset()
    
    # 3. Decision Transformer用の履歴バッファ
    context_len = 30
    target_return = 100.0  # ★「スコア100(クリア)を目指せ」という指示
    
    states = []
    actions = []
    rtgs = [target_return]
    
    # 画像の前処理関数
    def preprocess_image(img_array):
        img = Image.fromarray(img_array.transpose(1, 2, 0)) # CHW -> HWC
        img = img.resize((84, 84), Image.BILINEAR)
        img_arr = np.array(img, dtype=np.float32) / 255.0
        return img_arr.transpose(2, 0, 1) # HWC -> CHW

    # ゲームループ
    done = False
    while not done:
        # 現在の画像を処理して履歴に追加
        state = preprocess_image(obs['image'])
        states.append(state)
        
        # 履歴をコンテキスト長(K=30)に切り詰める
        states_input = torch.tensor(np.array(states[-context_len:]), dtype=torch.float32).unsqueeze(0).to(device)
        rtgs_input = torch.tensor(np.array(rtgs[-context_len:]), dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        
        # アクションは「現在選ぼうとしているもの」の分だけ足りないので0埋めパディング
        if len(actions) == 0:
            actions_input = torch.zeros((1, 1), dtype=torch.long).to(device)
        else:
            acts = actions[-(context_len-1):] + [0] # ダミーアクションを追加
            actions_input = torch.tensor(np.array(acts), dtype=torch.long).unsqueeze(0).to(device)
            
        timesteps = torch.arange(0, states_input.shape[1], dtype=torch.long).unsqueeze(0).to(device)
        
        # 4. モデルによるアクション予測
        with torch.no_grad():
            action_logits = model(states_input, actions_input, rtgs_input, timesteps)
            # 最後のステップのアクションを予測
            action_pred = torch.argmax(action_logits[0, -1]).item()
            
        # 予測したアクションを履歴に保存
        actions.append(action_pred)
        
        # 5. 環境を1ステップ進める
        obs, reward, terminated, truncated, info = env.step(action_pred)
        done = terminated or truncated
        
        # 6. 目標スコア(RTG)を消費した分だけ減らす
        # (DTは「残りあとどれくらい稼げばいいか」を入力するため)
        rtgs.append(rtgs[-1] - reward)

    print(f"Game Over! Final RTG left: {rtgs[-1]}")
    env.close()

if __name__ == "__main__":
    # 学習でできた重みファイルを指定して実行
    play_mario("mario_dt_epoch_10.pth")