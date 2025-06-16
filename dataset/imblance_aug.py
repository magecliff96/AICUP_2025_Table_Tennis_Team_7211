import pandas as pd
import numpy as np
from pathlib import Path

def augment_manual_combos(
    info_csv='39_Training_Dataset/train_info.csv',
    data_dir='39_Training_Dataset/train_data',
    output_csv='train_info_balanced.csv',
    output_dir='train_data_balanced',
    seed=42,
):
    np.random.seed(seed)

    def jitter(data, sigma_accel=182.47, sigma_gyro=935.88):
        noise = np.zeros_like(data, dtype=np.float32)
        noise[:, :3] = np.random.normal(0, sigma_accel, size=(data.shape[0], 3))
        noise[:, 3:] = np.random.normal(0, sigma_gyro, size=(data.shape[0], 3))
        return data + noise

    def scale(data, sigma_accel=0.15, sigma_gyro=0.07):
        scale_factors = np.ones((data.shape[0], 6))
        scale_factors[:, :3] = np.random.normal(1.0, sigma_accel, size=(data.shape[0], 3))
        scale_factors[:, 3:] = np.random.normal(1.0, sigma_gyro, size=(data.shape[0], 3))
        scale_factors = np.clip(scale_factors, 0.8, 1.2)
        return data * scale_factors

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    info = pd.read_csv(info_csv)
    new_rows = []

    rare_combo_dict = {
            (2, 1, 2, 2): 4,
            (1, 1, 1, 3): 2,
            (1, 1, 0, 3): 2,
            (1, 2, 1, 4): 4,
            (1, 1, 0, 4): 4,
            (2, 1, 1, 5): 4,
            (1, 2, 0, 3): 4,
            (2, 2, 2, 2): 4,
            (1, 2, 2, 2): 4,
        }

    combo_cols = ['gender', 'hold racket handed', 'play years', 'level']

    for _, row in info.iterrows():
        uid = row['unique_id']
        combo = tuple(row[col] for col in combo_cols)

        try:
            data = np.loadtxt(Path(data_dir) / f"{uid}.txt")
            # Always save original
            np.savetxt(output_path / f"{uid}.txt", data.astype(int), fmt='%d')
            new_rows.append(row)

            if combo in rare_combo_dict:
                aug_num = np.clip(rare_combo_dict[combo], 1, 10)  # restrict to [2, 5]

                for i in range(aug_num):
                    aug = jitter(data)
                    aug = scale(aug)
                    aug = np.round(aug).astype(int)
                    new_uid = f"{uid}_aug{i+1}"
                    new_path = output_path / f"{new_uid}.txt"
                    np.savetxt(new_path, aug, fmt='%d')
                    new_row = row.copy()
                    new_row['unique_id'] = new_uid
                    new_rows.append(new_row)

        except Exception as e:
            print(f"⚠️ Error processing {uid}: {e}")
            continue

    pd.DataFrame(new_rows).to_csv(output_csv, index=False)
    print(f"✅ Saved augmented data to: {output_dir}")
    print(f"✅ Saved new info CSV to: {output_csv}")
    
if __name__ == "__main__":
    augment_manual_combos()