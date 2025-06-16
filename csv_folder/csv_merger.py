import pandas as pd

# === File paths ===
csv_gender = "saved\gender95_handed99.csv"
csv_handed = "saved\gender93_handed99.csv"
csv_years = "saved\submission_yr.csv"
csv_level = "saved\submission_lv.csv"
output_csv = "combined_labels.csv"

# === Define which columns to use from each CSV ===
gender_cols = ["unique_id", "gender"]
handed_cols = ["unique_id", "hold racket handed"]
years_cols = ["unique_id", "play years_0", "play years_1", "play years_2"]
level_cols = ["unique_id", "level_2", "level_3", "level_4", "level_5"]

# === Read only the necessary columns ===
df_gender = pd.read_csv(csv_gender, usecols=gender_cols)
df_handed = pd.read_csv(csv_handed, usecols=handed_cols)
df_years = pd.read_csv(csv_years, usecols=years_cols)
df_level = pd.read_csv(csv_level, usecols=level_cols)

# === Merge all on 'unique_id' ===
df_merged = df_gender.merge(df_handed, on="unique_id") \
                     .merge(df_years, on="unique_id") \
                     .merge(df_level, on="unique_id")

# === Save final output ===
df_merged.to_csv(output_csv, index=False)
print(f"Merged CSV saved to: {output_csv}")
