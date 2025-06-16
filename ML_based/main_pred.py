import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from tqdm import tqdm
import xgboost as xgb
from collections import Counter
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer

def multiclass_roc_auc_micro(y_true, y_pred_proba):
    """Micro-average ROC AUC scorer for multiclass"""
    return roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='micro')

# 轉成 sklearn 用的 scorer
micro_auc_scorer = make_scorer(multiclass_roc_auc_micro, needs_proba=True)

data_dir = 'avg_cycles_txt'
sensor_names = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']

# Step 1: 先收集每個感測器的所有資料
sensor_data = {name: [] for name in sensor_names}
file_ids = []

for fname in tqdm(os.listdir(data_dir)):
    if not fname.endswith('.txt'):
        continue
    uid = fname.replace('.txt', '')
    file_ids.append(uid)
    
    data = np.loadtxt(os.path.join(data_dir, fname))  # shape: (173, 6)
    for i, name in enumerate(sensor_names):
        sensor_data[name].append(data[:, i])  # shape: (173,)

n_components = 3

# Step 2: 對每個感測器做 PCA（fit 所有 sample）
sensor_pca = {}
for name in sensor_names:
    X = np.array(sensor_data[name])  # shape: (N_samples, 173)
    pca = PCA(n_components=n_components)
    pca.fit(X)
    sensor_pca[name] = pca

# Step 3: 對每筆資料做 transform → 得到 18 維特徵
features = []

for idx, uid in enumerate(file_ids):
    feature_row = []
    for i, name in enumerate(sensor_names):
        vec = sensor_data[name][idx].reshape(1, -1)  # shape: (1, 173)
        reduced = sensor_pca[name].transform(vec)[0]  # shape: (3,)
        feature_row.extend(reduced)
    features.append([uid] + feature_row)

# Step 4: 放進 DataFrame，合併回 training_df
columns = ['unique_id'] + [f'{name}_pca_{i}' for name in sensor_names for i in range(n_components)]
feature_df = pd.DataFrame(features, columns=columns)
feature_df.to_csv("pca_feature.csv", index = None)

# feature_df = pd.read_csv("pca_feature.csv")
feature_df['unique_id'] = feature_df['unique_id'].astype("int")

train_df = pd.read_csv("train_data_v2.csv")
test_df = pd.read_csv("test_data_v2.csv")

# 合併特徵進去
train_df = train_df.merge(feature_df, on='unique_id', how='left')
test_df = test_df.merge(feature_df, on='unique_id', how='left')

train_df.to_csv("train_data_v2.csv", index = None)
test_df.to_csv("test_data_v2.csv", index = None)





# 特徵欄位（排除目標與 id）
exclude_cols = ['unique_id', 'gender', 'hold racket handed', 'play years', 'level']
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

# 設定目標欄位與對應類別數
targets = {
    'gender': 2,
    'hold racket handed': 2,
    'play years': 3,
    'level': 4,
}

# 儲存預測結果
predictions = pd.DataFrame({'unique_id': test_df['unique_id']})

param_grid = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# 對每個目標做預測
for target, n_classes in targets.items():
    print(f"\n🔍 Target: {target}")
    y_train = train_df[target].copy()
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    # 類別編碼調整
    if n_classes == 2:
        y_train = y_train - 1
    

    if target == 'level':
        y_train = y_train - 2
        

    class_counts = Counter(y_train)

    # 固定參數
    base_params = {
        'objective': 'multi:softprob' if n_classes > 2 else 'binary:logistic',
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
    }
    if n_classes > 2:
        base_params['num_class'] = n_classes
        scoring = micro_auc_scorer
    else:
        # 自動加 scale_pos_weight（僅 binary）  
        scoring = 'roc_auc'
        neg_class = min(class_counts, key=class_counts.get)
        pos_class = max(class_counts, key=class_counts.get)
        base_params['scale_pos_weight'] = class_counts[neg_class] / class_counts[pos_class]
        print(f"⚖️  scale_pos_weight = {base_params['scale_pos_weight']:.2f}")

    # 建立 base model
    base_model = xgb.XGBClassifier(**base_params)

    # Grid Search
    # 隨機搜尋超參數
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=50,
        scoring=scoring,
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train, y_train)

    print(f"[{target}] Best Params: {search.best_params_}")
    print(f"[{target}] Best CV Accuracy: {search.best_score_:.4f}")

    # 預測
    best_model = search.best_estimator_
    if n_classes == 2:
        prob = best_model.predict_proba(X_test)[:, 1]
        predictions[target] = prob
    else:
        prob = best_model.predict_proba(X_test)
        for i in range(n_classes):
            class_index = i + (2 if target == 'level' else 0)
            predictions[f'{target}_{class_index}'] = prob[:, i]


predictions['gender'] = 1 - predictions['gender']
predictions['hold racket handed'] = 1 - predictions['hold racket handed']

# result_df = predictions
# 四捨五入所有非 unique_id 的欄位到小數點 4 位
for col in predictions.columns:
    if col != 'unique_id':
        predictions[col] = predictions[col].round(3)

predictions.to_csv("../csv_folder/handed99_predictions.csv", index=False)
