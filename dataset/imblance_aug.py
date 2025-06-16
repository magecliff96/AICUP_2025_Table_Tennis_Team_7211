import pandas as pd
import numpy as np
from pathlib import Path

# def augment_balanced(
#     info_csv='39_Training_Dataset/train_info.csv',
#     data_dir='39_Training_Dataset/train_data',
#     output_csv='train_info_balanced.csv',
#     output_dir='train_data_balanced',
#     max_aug=5,
#     min_aug=1,
#     seed=42,
# ):
#     np.random.seed(seed)

#     def jitter(data, sigma_accel=182.47, sigma_gyro=935.88):
#         noise = np.zeros_like(data, dtype=np.float32)
#         noise[:, :3] = np.random.normal(0, sigma_accel, size=(data.shape[0], 3))
#         noise[:, 3:] = np.random.normal(0, sigma_gyro, size=(data.shape[0], 3))
#         return data + noise

#     def scale(data, sigma_accel=0.15, sigma_gyro=0.07):
#         scale_factors = np.ones((data.shape[0], 6))
#         scale_factors[:, :3] = np.random.normal(1.0, sigma_accel, size=(data.shape[0], 3))
#         scale_factors[:, 3:] = np.random.normal(1.0, sigma_gyro, size=(data.shape[0], 3))
#         scale_factors = np.clip(scale_factors, 0.8, 1.2)
#         return data * scale_factors

#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)

#     info = pd.read_csv(info_csv)
#     new_rows = []

#     # Compute label frequencies
#     freq_gender = info['gender'].value_counts(normalize=True).to_dict()
#     freq_handed = info['hold racket handed'].value_counts(normalize=True).to_dict()
#     freq_years = info['play years'].value_counts(normalize=True).to_dict()
#     freq_level = info['level'].value_counts(normalize=True).to_dict()

#     for _, row in info.iterrows():
#         uid = row['unique_id']
#         try:
#             data = np.loadtxt(Path(data_dir) / f"{uid}.txt")

#             # Scarcity score: 1 / min frequency across all labels
#             label_freqs = [
#                 freq_gender.get(row['gender'], 1.0),
#                 freq_handed.get(row['hold racket handed'], 1.0),
#                 freq_years.get(row['play years'], 1.0),
#                 freq_level.get(row['level'], 1.0)
#             ]
#             min_freq = min(label_freqs)
#             rarity_score = 1 / min_freq

#             # Normalize to [min_aug, max_aug]
#             rarity_score = np.clip(rarity_score / 10, 0, 1)
#             aug_num = int(min_aug + rarity_score * (max_aug - min_aug))

#             # Save original
#             np.savetxt(output_path / f"{uid}.txt", data.astype(int), fmt='%d')
#             new_rows.append(row)

#             for i in range(aug_num):
#                 aug = jitter(data)
#                 aug = scale(aug)
#                 aug = np.round(aug).astype(int)
#                 new_uid = f"{uid}_aug{i+1}"
#                 new_path = output_path / f"{new_uid}.txt"
#                 np.savetxt(new_path, aug, fmt='%d')
#                 new_row = row.copy()
#                 new_row['unique_id'] = new_uid
#                 new_rows.append(new_row)

#         except Exception as e:
#             print(f"⚠️ Error processing {uid}: {e}")
#             continue

#     # Save updated info CSV
#     pd.DataFrame(new_rows).to_csv(output_csv, index=False)
#     print(f"✅ Saved augmented data to: {output_dir}")
#     print(f"✅ Saved new info CSV to: {output_csv}")

# Run it
# augment_balanced()


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

    # 👇 Define rare combinations manually with requested augmentation times
    # Format: (gender, hand, years, level): times
    # rare_combo_dict = {
    #     (1, 1, 0, 3): 1,
    #     (1, 2, 1, 4): 3,
    #     (1, 1, 0, 4): 2,
    #     (1, 2, 0, 3): 5,
    #     (2, 1, 1, 5): 3,
    #     (2, 2, 2, 2): 10,
    # }
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
    
augment_manual_combos()


# import pandas as pd
# from collections import defaultdict

# def simulate_augmented_label_distribution(df, rare_combo_dict):
#     """
#     Simulate new label distributions after applying augmentation based on rare_combo_dict.

#     Parameters:
#     - df: pd.DataFrame with columns ['gender', 'hand', 'year', 'level', 'count']
#     - rare_combo_dict: dict with keys as (gender, hand, year, level) and values as augmentation times

#     Returns:
#     - dict of simulated label distributions for each column
#     """
#     # Start with original counts
#     simulated_counts = {
#         'gender': defaultdict(int),
#         'hold racket handed' : defaultdict(int),
#         'play years': defaultdict(int),
#         'level': defaultdict(int),
#     }

#     # Add original counts
#     for _, row in df.iterrows():
#         c = row['count']
#         simulated_counts['gender'][row['gender']] += c
#         simulated_counts['hold racket handed'][row['hold racket handed']] += c
#         simulated_counts['play years'][row['play years']] += c
#         simulated_counts['level'][row['level']] += c

#     # Add augmented counts
#     for (g, h, y, l), times in rare_combo_dict.items():
#         # Get the original count for that combination
#         match = df[(df['gender'] == g) & (df['hold racket handed'] == h) & (df['play years'] == y) & (df['level'] == l)]
#         if not match.empty:
#             c = match['count'].iloc[0]
#             added = c * times
#             simulated_counts['gender'][g] += added
#             simulated_counts['hold racket handed'][h] += added
#             simulated_counts['play years'][y] += added
#             simulated_counts['level'][l] += added

#     # Convert defaultdicts to sorted DataFrames for readability
#     final_distributions = {}
#     for key in simulated_counts:
#         final_distributions[key] = pd.Series(dict(simulated_counts[key])).sort_index()

#     return final_distributions

# # example df
# df = pd.read_csv("39_Training_Dataset/train_info.csv")

# # Group by label combinations
# group_counts = df.groupby(['gender', 'hold racket handed', 'play years', 'level']).size().reset_index(name='count')

# # Sort by count (optional)
# df = group_counts.sort_values(by='count', ascending=False).reset_index().drop(columns='index')

# # rare_combo_dict = {
# #     (1, 1, 0, 3): 1,
# #     (1, 2, 1, 4): 3,
# #     (1, 1, 0, 4): 2,
# #     (1, 2, 0, 3): 5,
# #     (2, 2, 2, 2): 10
# # }

# rare_combo_dict = {
#         (2, 1, 2, 2): 4,
#         (1, 1, 1, 3): 2,
#         (1, 1, 0, 3): 2,
#         (1, 2, 1, 4): 4,
#         (1, 1, 0, 4): 4,
#         (2, 1, 1, 5): 4,
#         (1, 2, 0, 3): 4,
#         (2, 2, 2, 2): 4,
#         (1, 2, 2, 2): 4,
#     }

# simulated = simulate_augmented_label_distribution(df, rare_combo_dict)

# for label, dist in simulated.items():
#     print(f"\nSimulated distribution for '{label}':")
#     print(dist)
