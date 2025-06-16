import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

# === Parameters ===
random_seed = 1001
info_path = './39_Training_Dataset/train_info.csv'
feature_dir = './39_Training_Dataset/tabular_data_train'
target_columns = ['gender', 'hold racket handed', 'play years', 'level']

# === Load player metadata ===
info = pd.read_csv(info_path)
unique_players = info['player_id'].unique()
train_players, test_players = train_test_split(unique_players, test_size=0.2, random_state=random_seed)

# === Initialize DataFrames ===
x_train, y_train = pd.DataFrame(), pd.DataFrame(columns=target_columns)
x_test, y_test = pd.DataFrame(), pd.DataFrame(columns=target_columns)

# === Process all feature files ===
feature_files = list(Path(feature_dir).glob('**/*.csv'))

for file in feature_files:
    unique_id = int(Path(file).stem)
    row = info[info['unique_id'] == unique_id]
    if row.empty:
        continue

    player_id = row['player_id'].iloc[0]
    features = pd.read_csv(file)
    if 'weight' in features.columns:
        features = features.drop(columns=['weight'])

    target_row = row[target_columns]
    target_repeated = pd.concat([target_row] * len(features), ignore_index=True)

    if player_id in train_players:
        x_train = pd.concat([x_train, features], ignore_index=True)
        y_train = pd.concat([y_train, target_repeated], ignore_index=True)
    else:
        x_test = pd.concat([x_test, features], ignore_index=True)
        y_test = pd.concat([y_test, target_repeated], ignore_index=True)

# === Class distribution function ===
def print_class_distribution(y_df, name):
    print(f"\n=== Class Distribution in {name} ===")
    for col in y_df.columns:
        counts = Counter(y_df[col])
        print(f"\n[{col}]")
        total = sum(counts.values())
        for k, v in sorted(counts.items()):
            print(f"  Class {k}: {v} ({v / total:.2%})")

# === Print results ===
print_class_distribution(y_train, "Train Set")
print_class_distribution(y_test, "Test Set")
