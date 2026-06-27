import os
import glob
import struct
import pickle
import logging
import time
import numpy as np

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("dt_preprocessing.log", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def parse_png_metadata(filepath):
    """PNGからカスタムメタデータ (RAM, BP1, OUTCOME) を抽出"""
    metadata = {}
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
    except Exception as e:
        logging.error(f"Failed to read file {filepath}: {e}")
        return metadata
        
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        return metadata
        
    offset = 8
    while offset < len(data):
        if offset + 4 > len(data): break
        length = struct.unpack('>I', data[offset:offset+4])[0]
        offset += 4
        
        if offset + 4 > len(data): break
        chunk_type = data[offset:offset+4]
        offset += 4
        
        if offset + length > len(data): break
        chunk_data = data[offset:offset+length]
        offset += length
        
        offset += 4  # Fake CRCをスキップ
        
        if chunk_type == b'tEXt':
            null_idx = chunk_data.find(b'\x00')
            if null_idx != -1:
                key = chunk_data[:null_idx].decode('ascii', errors='ignore')
                value = chunk_data[null_idx+1:]
                metadata[key] = value
                
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
    png_files = glob.glob(os.path.join(folder_path, "*.png"))
    png_files.sort(key=extract_frame_num)
    
    if not png_files:
        return None
        
    prev_x = None
    prev_score = None
    
    image_paths = []
    actions = []
    rewards = []
    states_scalar = []
    dones = []
    
    for i, filepath in enumerate(png_files):
        metadata = parse_png_metadata(filepath)
        ram = metadata.get('RAM', b'')
        
        # ====== GitHub Issue #4 バグ回避処理 ======
        if len(ram) > 2048:
            # Windowsのテキストモード追記による改行コード(\n -> \r\n)の増殖を元に戻す
            ram = ram.replace(b'\r\n', b'\n')
            if len(ram) > 2048:
                ram = ram.replace(b'\r\n', b'\r')
        # ==========================================
        
        if len(ram) < 2048:
            continue
            
        current_x = get_mario_x(ram)
        current_score = get_mario_score(ram)
        action = metadata.get('BP1', b'\x00')[0]
        reward = 0.0
        done = False
        
        if prev_x is not None:
            # 進行度報酬 (NES版は1ブロック16px)
            reward += (current_x - prev_x) / 16.0
            # 時間ペナルティ
            reward -= 0.01
            # スコアボーナス
            if current_score > prev_score:
                reward += (current_score - prev_score) / 100.0
                
        if i == len(png_files) - 1:
            done = True
            outcome = metadata.get('OUTCOME', b'\x00')[0]
            if filepath.endswith('.win.png') or outcome == 2:
                reward += 100.0
            elif filepath.endswith('.fail.png') or outcome == 1:
                reward -= 10.0
                
        image_paths.append(filepath)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        states_scalar.append([current_x, current_score])
        
        prev_x = current_x
        prev_score = current_score

    if not rewards:
        return None

    returns_to_go = np.zeros_like(rewards, dtype=np.float32)
    current_rtg = 0.0
    for t in reversed(range(len(rewards))):
        current_rtg = rewards[t] + current_rtg
        returns_to_go[t] = current_rtg

    episode_data = {
        'image_paths': image_paths,
        'actions': np.array(actions, dtype=np.int32),
        'rewards': np.array(rewards, dtype=np.float32),
        'returns_to_go': returns_to_go,
        'states_scalar': np.array(states_scalar, dtype=np.float32),
        'dones': np.array(dones, dtype=bool)
    }
    
    return episode_data

def main():
    dataset_root = "data"  # 環境に応じて書き換えてください
    all_episodes_data = []
    
    logging.info(f"Starting DT dataset preprocessing. Root directory: {dataset_root}")
    start_time = time.time()
    
    if not os.path.exists(dataset_root):
        logging.error(f"Root directory '{dataset_root}' does not exist.")
        return

    folders = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
    total_folders = len(folders)
    logging.info(f"Found {total_folders} episode folders to process.")

    for idx, folder_name in enumerate(folders, 1):
        folder_path = os.path.join(dataset_root, folder_name)
        ep_data = process_episode_for_dt(folder_path)
        
        if ep_data and len(ep_data['rewards']) > 0:
            # ★ 追加: そのエピソードの合計報酬を計算
            total_reward = sum(ep_data['rewards'])
            
            # ログに合計報酬(Total Reward)を含めて出力
            logging.info(f"[{idx}/{total_folders}] {folder_name} -> OK ({len(ep_data['rewards'])} frames) | Total Reward: {total_reward:.2f}")
            all_episodes_data.append(ep_data)
        else:
            logging.warning(f"[{idx}/{total_folders}] {folder_name} -> Skipped (No valid frames)")
            
    output_file = "dt_mario_dataset_metadata.pkl"
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(all_episodes_data, f)
        elapsed_time = time.time() - start_time
        logging.info(f"Successfully saved metadata for {len(all_episodes_data)} episodes to {output_file}.")
        logging.info(f"Preprocessing completed in {elapsed_time:.2f} seconds.")
    except Exception as e:
        logging.error(f"Failed to save pickle file: {e}")

if __name__ == "__main__":
    main()