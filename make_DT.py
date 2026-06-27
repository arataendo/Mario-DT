import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from PIL import Image
from transformers import GPT2Config, GPT2Model

# ==========================================
# 1. データセットの定義
# ==========================================
class MarioDTDataset(Dataset):
    def __init__(self, pkl_path, context_len=30, image_size=(84, 84)):
        """
        context_len: K（Transformerに入力する系列長）
        """
        print(f"Loading metadata from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            self.episodes = pickle.load(f)
            
        self.context_len = context_len
        self.image_size = image_size
        
        # エピソードごとの長さを計算
        self.lengths = [len(ep['rewards']) for ep in self.episodes]
        self.num_episodes = len(self.episodes)
        
        # 状態（RTGなど）の最大・最小値（正規化用）
        all_rtg = np.concatenate([ep['returns_to_go'] for ep in self.episodes])
        self.rtg_max, self.rtg_min = np.max(all_rtg), np.min(all_rtg)
        print(f"Dataset Loaded: {self.num_episodes} episodes. RTG range: {self.rtg_min:.2f} to {self.rtg_max:.2f}")

    def __len__(self):
        # 任意のエピソードをランダムサンプリングするため、仮想的なデータセット長を定義
        return 50000

    def __getitem__(self, idx):
        # ランダムにエピソードを選択
        ep_idx = random.randint(0, self.num_episodes - 1)
        episode = self.episodes[ep_idx]
        ep_len = self.lengths[ep_idx]
        
        # サンプリングする開始ステップをランダムに決定
        start_t = random.randint(0, ep_len - 1)
        end_t = min(start_t + self.context_len, ep_len)
        seq_len = end_t - start_t
        
        # データの切り出し
        image_paths = episode['image_paths'][start_t:end_t]
        actions = episode['actions'][start_t:end_t]
        rtg = episode['returns_to_go'][start_t:end_t]
        # 正規化: 0 ~ 1の範囲にスケール
        rtg = (rtg - self.rtg_min) / (self.rtg_max - self.rtg_min + 1e-5)
        
        timesteps = np.arange(start_t, end_t)

        # 画像の遅延読み込みと前処理 (C, H, W)
        images = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(self.image_size, Image.BILINEAR)
                img_arr = np.array(img, dtype=np.float32) / 255.0
                images.append(img_arr.transpose(2, 0, 1)) # HWC -> CHW
            except Exception as e:
                # 読み込み失敗時はゼロ埋め画像
                images.append(np.zeros((3, self.image_size[1], self.image_size[0]), dtype=np.float32))
        
        images = np.array(images)

        # Padding (系列長が K に満たない場合、先頭を0で埋める)
        pad_len = self.context_len - seq_len
        
        # パディング処理
        images = np.concatenate([np.zeros((pad_len, 3, *self.image_size), dtype=np.float32), images], axis=0)
        actions = np.concatenate([np.zeros(pad_len, dtype=np.int32), actions], axis=0)
        rtg = np.concatenate([np.zeros(pad_len, dtype=np.float32), rtg], axis=0)
        timesteps = np.concatenate([np.zeros(pad_len, dtype=np.int32), timesteps], axis=0)
        
        # アテンションマスク (パディング部分は 0, 実際のデータ部分は 1)
        attention_mask = np.concatenate([np.zeros(pad_len, dtype=np.float32), np.ones(seq_len, dtype=np.float32)], axis=0)

        return {
            'states': torch.tensor(images),
            'actions': torch.tensor(actions, dtype=torch.long), # 離散アクション
            'returns_to_go': torch.tensor(rtg, dtype=torch.float32).unsqueeze(-1),
            'timesteps': torch.tensor(timesteps, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float32)
        }


# ==========================================
# 2. モデルの定義 (Decision Transformer)
# ==========================================
class DecisionTransformer(nn.Module):
    def __init__(self, action_vocab_size=256, hidden_size=128, max_ep_len=10000):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_len = 30 # K
        
        # GPT2 の設定 (軽量化設定)
        config = GPT2Config(
            vocab_size=1,  # 使用しないが必須
            n_embd=hidden_size,
            n_layer=3,
            n_head=4,
            n_inner=4 * hidden_size,
            activation_function='relu',
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )
        self.transformer = GPT2Model(config)
        
        # 画像エンコーダ (Nature CNN)
        # 入力: (B, 3, 84, 84) -> 出力: (B, hidden_size)
        self.state_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, hidden_size),
            nn.Tanh()
        )
        
        # 各モダリティのエンベディング
        self.embed_rtg = nn.Linear(1, hidden_size)
        self.embed_action = nn.Embedding(action_vocab_size, hidden_size)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        # アクション予測用ヘッド (離散アクションクラス分類)
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, action_vocab_size)
        )

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        
        # 画像状態のエンコード (B*T, C, H, W) -> (B, T, hidden_size)
        states = states.view(-1, 3, 84, 84)
        state_embeddings = self.state_encoder(states).view(batch_size, seq_length, self.hidden_size)
        
        # 他のエンベディング
        action_embeddings = self.embed_action(actions)
        rtg_embeddings = self.embed_rtg(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)
        
        # タイムステップの情報を加算
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        rtg_embeddings = rtg_embeddings + time_embeddings
        
        # (RTG, State, Action) の順に系列を並べる
        # 形状: (B, T, 3, hidden_size)
        stacked_inputs = torch.stack(
            (rtg_embeddings, state_embeddings, action_embeddings), dim=2
        )
        # 形状: (B, T*3, hidden_size)
        inputs_embeds = stacked_inputs.view(batch_size, seq_length * 3, self.hidden_size)
        inputs_embeds = self.embed_ln(inputs_embeds)
        
        # Attention Maskの拡張 (RTG, State, Action 全てに適用するため3倍にする)
        if attention_mask is not None:
            stacked_mask = torch.stack(
                (attention_mask, attention_mask, attention_mask), dim=2
            ).view(batch_size, seq_length * 3)
        else:
            stacked_mask = None

        # Transformer の順伝播
        transformer_outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=stacked_mask,
        )
        x = transformer_outputs['last_hidden_state']
        
        # x の形状: (B, T*3, hidden_size)
        # 予測ヘッドの入力は State エンベディングの位置の出力（Stateの次に来るActionを予測）
        # RTG_0, State_0, Action_0, RTG_1, State_1, Action_1 ...
        # Actionの予測に使いたいのは State_t の表現なので、インデックスは 1, 4, 7...
        states_preds_idx = torch.arange(1, seq_length * 3, 3, device=x.device)
        state_representations = x[:, states_preds_idx, :]
        
        action_logits = self.predict_action(state_representations)
        
        return action_logits


# ==========================================
# 3. 学習ループ
# ==========================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ハイパーパラメータ
    batch_size = 64
    epochs = 10
    learning_rate = 1e-4
    context_len = 30
    
    # データローダー
    dataset = MarioDTDataset("smbdataset-main/dt_mario_dataset_metadata.pkl", context_len=context_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # モデルの初期化 (Action空間は 0~255 の 256クラス)
    model = DecisionTransformer(action_vocab_size=256, hidden_size=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 学習ループ
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        # 1エポックあたり 1000ステップ回す（データセットサイズが仮想的なため）
        for step, batch in enumerate(dataloader):
            if step >= 1000:
                break
                
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            rtg = batch['returns_to_go'].to(device)
            timesteps = batch['timesteps'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # フォワードパス
            action_logits = model(states, actions, rtg, timesteps, attention_mask=attention_mask)
            
            # ロスの計算 (CrossEntropyLoss)
            # action_logits: (B, T, vocab_size)
            # actions: (B, T)
            # マスクされている部分（パディング）はロスに含めない
            logits_flat = action_logits.view(-1, 256)
            actions_flat = actions.view(-1)
            mask_flat = attention_mask.view(-1)
            
            loss = F.cross_entropy(logits_flat, actions_flat, reduction='none')
            loss = (loss * mask_flat).sum() / mask_flat.sum()
            
            # バックプロパゲーション
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            
            total_loss += loss.item()
            
            if step % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {step} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / 1000
        print(f"=== Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f} ===")
        
        # モデルの保存
        torch.save(model.state_dict(), f"mario_dt_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()