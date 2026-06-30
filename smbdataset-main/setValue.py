import os
import glob
import pickle
import logging
import time
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("dt_preprocessing.log", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def parse_png_metadata_raw(filepath):
    """
    PNGの構造を無視し、バイナリ文字列検索でメタデータを強引かつ安全に抽出するパーサー
    """
    metadata = {}
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
    except Exception as e:
        return metadata

    # 各チャンクの開始位置を検索
    ram_start = content.find(b'tEXtRAM\x00')
    bp1_start = content.find(b'tEXtBP1\x00')
    bp2_start = content.find(b'tEXtBP2\x00')
    outcome_start = content.find(b'tEXtOUTCOME\x00')

    # RAMデータの抽出と修復
    if ram_start != -1:
        # 次のチャンクまでの間がRAMデータ（前後のヘッダー等8バイトを除外）
        ends = [idx for idx in [bp1_start, bp2_start, outcome_start, len(content)] if idx != -1 and idx > ram_start]
        ram_end = min(ends) - 8 if ends else len(content)
        
        ram_raw = content[ram_start+8 : ram_end]
        
        # Windowsテキストモードの改行バグを修復
        if len(ram_raw) == 2048:
            ram = ram_raw
        else:
            ram = ram_raw.replace(b'\r\n', b'\n')
            if len(ram) != 2048:
                ram = ram_raw.replace(b'\r\n', b'\r')
                
        # 厳密に2048バイトに復元できた場合のみ採用
        if len(ram) == 2048:
            metadata['RAM'] = ram

    # アクションデータの抽出
    if bp1_start != -1:
        val = content[bp1_start+8]
        if val == 13 and content[bp1_start+9] == 10:  # 改行バグの巻き添え対応
            val = 10
        metadata['BP1'] = bytes([val])

    # アウトカムデータの抽出
    if outcome_start != -1:
        val = content[outcome_start+12]  # tEXtOUTCOME\x00 は12バイト
        if val == 13 and content[outcome_start+13] == 10:
            val = 10
        metadata['OUTCOME'] = bytes([val])

    return metadata

def get_mario_x(ram):
    """X座標（安全装置付き）"""
    page = ram[0x006D]
    x_pos = ram[0x0086]
    # バグでページ数が異常な値になった場合は無効化
    if page > 20: 
        return 0
    return page * 256 + x_pos

def get_mario_score(ram):
    """スコア計算（超巨大文字列バグ防止付き）"""
    score_str = ""
    for i in range(0x07DD, 0x07DD + 6):
        val = ram[i]
        # ファミコンのスコアは0~9のBCD形式。それ以外のゴミデータは0として扱う
        if 0 <= val <= 9:
            score_str += str(val)
        else:
            score_str += "0"
            
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
    
    if not png_files: return None
        
    prev_x = None
    prev_score = None
    
    image_paths, actions, rewards, states_scalar, dones = [], [], [], [], []
    
    for i, filepath in enumerate(png_files):
        metadata = parse_png_metadata_raw(filepath)
        ram = metadata.get('RAM')
        
        # 復元に失敗した（長さが2048じゃない）フレームはスキップ
        if not ram: continue
            
        current_x = get_mario_x(ram)
        current_score = get_mario_score(ram)
        action = metadata.get('BP1', b'\x00')[0]
        reward = 0.0
        done = False
        
        if prev_x is not None:
            # 1. 進行度報酬 (異常なジャンプは除外)
            delta_x = current_x - prev_x
            if -50 < delta_x < 50: 
                reward += delta_x / 16.0
                
            # 2. 時間ペナルティ
            reward -= 0.01
            
            # 3. スコアボーナス
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

    if not rewards: return None

    returns_to_go = np.zeros_like(rewards, dtype=np.float32)
    current_rtg = 0.0
    for t in reversed(range(len(rewards))):
        current_rtg = rewards[t] + current_rtg
        returns_to_go[t] = current_rtg

    return {
        'image_paths': image_paths,
        'actions': np.array(actions, dtype=np.int32),
        'rewards': np.array(rewards, dtype=np.float32),
        'returns_to_go': returns_to_go,
        'states_scalar': np.array(states_scalar, dtype=np.float32),
        'dones': np.array(dones, dtype=bool)
    }

def main():
    dataset_root = "data-smb"
    all_episodes_data = []
    
    folders = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
    total_folders = len(folders)
    
    for idx, folder_name in enumerate(folders, 1):
        folder_path = os.path.join(dataset_root, folder_name)
        ep_data = process_episode_for_dt(folder_path)
        
        if ep_data and len(ep_data['rewards']) > 0:
            total_reward = sum(ep_data['rewards'])
            logging.info(f"[{idx}/{total_folders}] {folder_name} -> OK ({len(ep_data['rewards'])} frames) | Total Reward: {total_reward:.2f}")
            all_episodes_data.append(ep_data)
        else:
            logging.warning(f"[{idx}/{total_folders}] {folder_name} -> Skipped (No valid frames)")
            
    try:
        with open("dt_mario_dataset_metadata.pkl", 'wb') as f:
            pickle.dump(all_episodes_data, f)
        logging.info("Preprocessing completed successfully.")
    except Exception as e:
        logging.error(f"Failed to save pickle file: {e}")

if __name__ == "__main__":
    main()
