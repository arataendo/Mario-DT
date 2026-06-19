import os
import glob
import struct
import pickle
import numpy as np

def parse_png_metadata(filepath):
    """
    PNGからカスタムメタデータ (RAM, BP1, OUTCOME) を抽出
    ※データセットの仕様上、IENDチャンクの後ろにメタデータが追記されているため、
     最後まで読み切るようにしています。
    """
    metadata = {}
    with open(filepath, 'rb') as f:
        data = f.read()
        
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        return metadata
        
    offset = 8
    while offset < len(data):
        # バッファオーバーラン防止
        if offset + 4 > len(data): break
        length = struct.unpack('>I', data[offset:offset+4])[0]
        offset += 4
        
        if offset + 4 > len(data): break
        chunk_type = data[offset:offset+4]
        offset += 4
        
        if offset + length > len(data): break
        chunk_data = data[offset:offset+length]
        offset += length
        
        offset += 4 # Fake CRC分をスキップ
        
        if chunk_type == b'tEXt':
            null_idx = chunk_data.find(b'\x00')
            if null_idx != -1:
                key = chunk_data[:null_idx].decode('ascii', errors='ignore')
                value = chunk_data[null_idx+1:]
                metadata[key] = value
                
        # 修正ポイント: ここにあった `if chunk_type == b'IEND': break` を削除しました
        
    return metadata
def get_mario_x(ram):
    if len(ram) <= 0x0086: return 0
    return ram[0x006D] * 256 + ram[0x0086]

def get_mario_score(ram):
    if len(ram) <= 0x07E2: return 0
    score_str = "".join(str(ram[i]) for i in range(0x07DD, 0x07DD + 6))
    try:
        return int(score_str) * 10
    except ValueError:
        return 0

def extract_frame_num(filename):
    parts = os.path.basename(filename).split('_')
    for p in parts:
        if p.startswith('f') and p[1:].isdigit():
            return int(p[1:])
    return 0

def process_episode_for_dt(folder_path):
    """
    1エピソードを処理し、Decision Transformer向けの辞書を作成する
    """
    png_files = glob.glob(os.path.join(folder_path, "*.png"))
    png_files.sort(key=extract_frame_num)
    
    if not png_files:
        return None
        
    prev_x = None
    prev_score = None
    
    # DT用に記録するリスト
    image_paths = []
    actions = []
    rewards = []
    states_scalar = [] # RAMから抽出したスカラー状態（Pygame版のstateベクトル相当）
    dones = []
    
    for i, filepath in enumerate(png_files):
        metadata = parse_png_metadata(filepath)
        ram = metadata.get('RAM', b'')
        
        if len(ram) < 2048:
            continue
            
        current_x = get_mario_x(ram)
        current_score = get_mario_score(ram)
        action = metadata.get('BP1', b'\x00')[0]
        reward = 0.0
        done = False
        
        if prev_x is not None:
            # 1. 進行度報酬
            reward += (current_x - prev_x) / 16.0
            # 2. 時間ペナルティ
            reward -= 0.01
            # 3. スコア増分ボーナス
            if current_score > prev_score:
                reward += (current_score - prev_score) / 100.0
                
        # 終了フレーム判定
        if i == len(png_files) - 1:
            done = True
            outcome = metadata.get('OUTCOME', b'\x00')[0]
            if filepath.endswith('.win.png') or outcome == 2:
                reward += 100.0  # クリア
            elif filepath.endswith('.fail.png') or outcome == 1:
                reward -= 10.0   # ゲームオーバー
                
        image_paths.append(filepath)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        
        # Pygame版の 'state' に似たスカラー情報を構築（必要に応じて調整）
        # [mario_x, score, progress...]
        # ※ DTで画像のみを使う場合は不要ですが、状態ベクトルも入力に使う場合に役立ちます
        states_scalar.append([current_x, current_score])
        
        prev_x = current_x
        prev_score = current_score

    # ====== Return-to-go (RTG) の計算 ======
    # 後ろから累積して、各ステップでの「これ以降に得られる総報酬」を計算
    returns_to_go = np.zeros_like(rewards, dtype=np.float32)
    current_rtg = 0.0
    for t in reversed(range(len(rewards))):
        current_rtg = rewards[t] + current_rtg
        returns_to_go[t] = current_rtg

    # エピソードデータを辞書としてまとめる
    episode_data = {
        'image_paths': image_paths,
        'actions': np.array(actions, dtype=np.int32),
        'rewards': np.array(rewards, dtype=np.float32),
        'returns_to_go': returns_to_go,
        'states_scalar': np.array(states_scalar, dtype=np.float32),
        'dones': np.array(dones, dtype=bool)
    }
    
    return episode_data

if __name__ == "__main__":
    dataset_root = "data-smb" # 実際のフォルダパスに合わせてください
    all_episodes_data = []
    
    # フォルダごとに処理
    for folder_name in os.listdir(dataset_root):
        folder_path = os.path.join(dataset_root, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing {folder_name}...", end=" ")
            ep_data = process_episode_for_dt(folder_path)
            
            if ep_data is not None and len(ep_data['rewards']) > 0:
                print(f"-> OK ({len(ep_data['rewards'])} frames)")
                all_episodes_data.append(ep_data)
            else:
                print("-> Skipped (No valid frames)")
                
    # メタデータをPickleで保存
    output_file = "dt_mario_dataset_metadata.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(all_episodes_data, f)
        
    print(f"Saved metadata for {len(all_episodes_data)} episodes to {output_file}.")