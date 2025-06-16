import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis, entropy, iqr

Type = "train"

info_df = pd.read_csv(f'../dataset/39_Training_Dataset/{Type}_info.csv')

# # # Original Figure
# base_output_dir = 'plots'
# os.makedirs(base_output_dir, exist_ok=True)

# for _, row in info_df.iterrows():
#     unique_id = row['unique_id']
#     mode = row['mode']
#     data_path = f'../dataset/39_Training_Dataset/{Type}_data/{unique_id}.txt'

#     # 要輸出的檔案路徑
#     mode_folder = os.path.join(base_output_dir, f'mode_{mode}')
#     os.makedirs(mode_folder, exist_ok=True)
#     output_path = os.path.join(mode_folder, f'{unique_id}.png')

#     # 如果檔案已存在就跳過
#     if os.path.exists(output_path):
#         print(f"⏩ Skipping ID {unique_id} (already exists)")
#         continue
#     # print(unique_id)
#     # 嘗試讀取 .txt 檔案
#     try:
#         data = np.loadtxt(data_path)
#     except Exception as e:
#         print(f"⚠️ Error loading {data_path}: {e}")
#         continue

#     # 檢查是否有6欄
#     if data.shape[1] != 6:
#         print(f"⚠️ Unexpected shape in {data_path}: {data.shape}")
#         continue

#     # 繪圖
#     plt.figure(figsize=(12, 6))
#     labels = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
#     for i in range(6):
#         plt.plot(data[:, i], label=labels[i])
    
#     plt.title(f'ID {unique_id} - Mode {mode}')
#     plt.xlabel('Time Step')
#     plt.ylabel('Sensor Value')
#     plt.legend(loc="best")
#     plt.grid(True)

#     # 儲存圖片
#     plt.savefig(output_path)
#     plt.close()


# Cycle Figure

# # 輸出資料夾
# output_dir = 'cycle_plots'
# os.makedirs(output_dir, exist_ok=True)

# 輸入資料，直接使用原始資料
period = []
for _, row in info_df.iterrows():
    unique_id = row['unique_id']
    mode = row['mode']
    data_path = f'../dataset/39_Training_Dataset/{Type}_data/{unique_id}.txt'
    
    # 載入揮拍資料
    try:
        data = np.loadtxt(data_path)
    except Exception as e:
        print(f"⚠️ Failed loading {data_path}: {e}")
        continue

    if data.shape[1] != 6:
        print(f"⚠️ Unexpected shape: {data.shape}")
        continue

    trimmed = data

    # cut_point 處理：字串轉陣列
    cut_points = np.fromstring(row['cut_point'].strip("[]"), sep=' ').astype(int)
    
    # 只保留在 trimmed 範圍內的切割點
    cut_points = cut_points[(cut_points >= 0) & (cut_points < len(trimmed))]
    
    # 確保有有效的切割點
    if len(cut_points) < 2:
        print(f"⚠️ No valid cut points for {unique_id}, skipping.")
        continue

    # 切割每個週期段
    segments = []
    for i in range(len(cut_points) - 1):
        seg = trimmed[cut_points[i]:cut_points[i+1]]
        if seg.shape[0] > 0:
            segments.append(seg)

    # 確保有足夠的有效區段
    if len(segments) < 1:
        print(f"⚠️ {unique_id} has no valid segments after cutting, skipping.")
        continue
    # print(len(segments))
    
    segments = segments[5:-5]
    # 對齊週期長度（補齊或裁剪成統一長度）
    min_len = min([len(s) for s in segments])
    period.append(min_len)

    aligned = np.array([s[:min_len] for s in segments])  # (num_cycles, min_len, 6)

    # 平均每個時間點的數值
    mean_cycle = aligned.mean(axis=0)  # (min_len, 6)

    
    
#     # 儲存圖片
#     mode_dir = os.path.join(output_dir, f'mode_{mode}')
#     # if os.path.exists(os.path.join(mode_dir, f'{unique_id}_cycle.png')):
#     #     # print(f"⏩ Skipping ID {unique_id} (already exists)")
#     #     continue
#     # print(unique_id)
#     # 繪圖
#     plt.figure(figsize=(12, 6))
#     labels = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
#     for i in range(6):
#         plt.plot(mean_cycle[:, i], label=labels[i])

#     plt.title(f'ID {unique_id} - Avg Cycle (Mode {mode})')
#     plt.xlabel('Time Step (normalized)')
#     plt.ylabel('Sensor Value')
#     plt.legend()
#     plt.grid(True)
    

#     os.makedirs(mode_dir, exist_ok=True)
#     plt.savefig(os.path.join(mode_dir, f'{unique_id}_cycle.png'))
#     # print(os.path.join(mode_dir, f'{unique_id}_cycle.png'))
#     plt.close()

# print("✅ 完成平均週期繪圖")


training_df = info_df[['unique_id', 'mode', 'gender', 'hold racket handed', 'play years', 'level']].copy()
training_df['period'] = period
training_df.to_csv(f"{Type}_data.csv", index = None)

# ----- 特徵提取函式 -----
def extract_features(signal):
    features = {}
    features['mean'] = np.mean(signal)
    features['median'] = np.median(signal)
    features['min'] = np.min(signal)
    features['max'] = np.max(signal)
    features['std'] = np.std(signal)
    features['var'] = np.var(signal)
    features['range'] = np.max(signal) - np.min(signal)
    features['iqr'] = iqr(signal)
    features['skew'] = skew(signal)
    features['kurtosis'] = kurtosis(signal)
    features['energy'] = np.sum(signal ** 2)
    features['rms'] = np.sqrt(np.mean(signal**2))
    features['zcr'] = ((signal[:-1] * signal[1:]) < 0).sum()
    features['sma'] = np.sum(np.abs(signal)) / len(signal)

    # Entropy
    hist, _ = np.histogram(signal, bins=20, density=True)
    hist += 1e-12  # avoid log(0)
    features['entropy'] = -np.sum(hist * np.log2(hist))

    features['ptp'] = np.ptp(signal)
    return features

# ----- 資料處理 -----
all_rows = []  # 用來儲存每筆資料的完整特徵（包含基本欄位與 6 條感測器的統計特徵）

for _, row in info_df.iterrows():
    unique_id = row['unique_id']
    mode = row['mode']
    data_path = f'../dataset/39_Training_Dataset/{Type}_data/{unique_id}.txt'


    try:
        data = np.loadtxt(data_path)
    except Exception as e:
        print(f"⚠️ Failed loading {data_path}: {e}")
        continue

    if data.shape[1] != 6:
        print(f"⚠️ Unexpected shape: {data.shape}")
        continue

    trimmed = data

    cut_points = np.fromstring(row['cut_point'].strip("[]"), sep=' ').astype(int)
    cut_points = cut_points[(cut_points >= 0) & (cut_points < len(trimmed))]
    if len(cut_points) < 2:
        print(f"⚠️ No valid cut points for {unique_id}, skipping.")
        continue

    segments = []
    for i in range(len(cut_points) - 1):
        seg = trimmed[cut_points[i]:cut_points[i+1]]
        if seg.shape[0] > 0:
            segments.append(seg)

    if len(segments) < 1:
        print(f"⚠️ {unique_id} has no valid segments after cutting, skipping.")
        continue

    segments = segments[3:-3]
    min_len = min([len(s) for s in segments])
    aligned = np.array([s[:min_len] for s in segments])  # (num_cycles, min_len, 6)
    mean_cycle = aligned.mean(axis=0)  # (min_len, 6)

    # 統計特徵提取
    labels = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    all_features = {}
    for i in range(6):
        signal = mean_cycle[:, i]
        feats = extract_features(signal)
        for k, v in feats.items():
            all_features[f"{labels[i]}_{k}"] = v

    # 加入 meta 資訊
    record = {
        'unique_id': unique_id,
        'mode': row['mode'],
        'gender': row['gender'],
        'hold racket handed': row['hold racket handed'],
        'play years': row['play years'],
        'level': row['level'],
        'period': min_len
    }
    record.update(all_features)
    all_rows.append(record)

# ----- 建立 DataFrame 並存成 CSV -----
training_df = pd.DataFrame(all_rows)
training_df.to_csv("train_data_v2.csv", index=False)



target_len = 173
output_txt_dir = 'avg_cycles_txt'
os.makedirs(output_txt_dir, exist_ok=True)

def pad_cycle_head_repeat(data, target_len=173):
    current_len = data.shape[0]
    if current_len == target_len:
        return data
    elif current_len > target_len:
        return data[:target_len]
    else:
        pad_len = target_len - current_len
        # 複製開頭資料來補
        repeat_times = int(np.ceil(pad_len / current_len))
        head_repeated = np.tile(data[:current_len], (repeat_times, 1))[:pad_len]
        padded = np.vstack([data, head_repeated])
        return padded

for _, row in info_df.iterrows():
    unique_id = row['unique_id']
    data_path = f'../dataset/39_Training_Dataset/{Type}_data/{unique_id}.txt'

    try:
        data = np.loadtxt(data_path)
    except Exception as e:
        print(f"⚠️ Failed loading {data_path}: {e}")
        continue

    if data.shape[1] != 6:
        print(f"⚠️ Unexpected shape: {data.shape}")
        continue

    trimmed = data

    cut_points = np.fromstring(row['cut_point'].strip("[]"), sep=' ').astype(int)
    cut_points = cut_points[(cut_points >= 0) & (cut_points < len(trimmed))]

    if len(cut_points) < 2:
        print(f"⚠️ No valid cut points for {unique_id}, skipping.")
        continue

    # 丟棄前三後三
    if len(cut_points) < 8:
        print(f"⚠️ Not enough segments after removing head/tail for {unique_id}, skipping.")
        continue

    segments = []
    for i in range(3, len(cut_points) - 4):  # 去掉前三後三
        seg = trimmed[cut_points[i]:cut_points[i+1]]
        if seg.shape[0] > 0:
            segments.append(seg)

    if len(segments) < 1:
        print(f"⚠️ {unique_id} has no valid segments after cut, skipping.")
        continue
    print(unique_id)
    min_len = min([len(s) for s in segments])
    aligned = np.array([s[:min_len] for s in segments])  # (n_segments, min_len, 6)
    mean_cycle = aligned.mean(axis=0)  # shape: (min_len, 6)

    padded_cycle = pad_cycle_head_repeat(mean_cycle, target_len=173)

    # 儲存為 txt
    out_path = os.path.join(output_txt_dir, f"{unique_id}.txt")
    np.savetxt(out_path, padded_cycle, fmt="%.6f")

print("✅ 所有平均週期已補齊至 173 並儲存完畢")
