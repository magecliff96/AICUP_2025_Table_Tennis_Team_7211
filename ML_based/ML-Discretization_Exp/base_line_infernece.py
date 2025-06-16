import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pathlib import Path

#boost models
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

#stack model
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def main():
    group_size = 27
    seed = 1001
    # Load info and feature paths
    train_info = pd.read_csv('./39_Training_Dataset/train_info.csv')
    test_info = pd.read_csv('./39_Test_Dataset/test_info.csv')
    train_datapath = './39_Training_Dataset/tabular_data_train2'
    test_datapath = './39_Test_Dataset/tabular_data_test2'
    train_datalist = list(Path(train_datapath).glob('**/*.csv'))
    test_datalist = list(Path(test_datapath).glob('**/*.csv'))
    target_mask = ['gender', 'hold racket handed', 'play years', 'level']

    # Prepare training features and labels
    x_train = pd.DataFrame()
    y_train = pd.DataFrame(columns=target_mask)

    # === Initialize data holders ===
    x_train = pd.DataFrame()
    y_train = pd.DataFrame(columns=target_mask)
    train_weights = []

    x_test = pd.DataFrame()
    test_weights = []

    for file in train_datalist:
        unique_id = int(Path(file).stem)
        row = train_info[train_info['unique_id'] == unique_id]
        if row.empty:
            continue

        data = pd.read_csv(file)

        if 'weight' in data.columns:
            train_w = data['weight'].values
        else:
            train_w = np.ones(len(data))  # fallback

        # Add play_mode column
        play_mode = row['mode'].iloc[0]
        data['play_mode'] = play_mode

        target = row[target_mask]
        target_repeated = pd.concat([target] * len(data), ignore_index=True)

        x_train = pd.concat([x_train, data], ignore_index=True)
        y_train = pd.concat([y_train, target_repeated], ignore_index=True)
        train_weights.extend(train_w)



    # Prepare test features and test IDs

    x_test = pd.DataFrame()
    test_weights = []
    test_ids = []

    for file in test_datalist:
        unique_id = int(Path(file).stem)
        test_ids.append(unique_id)

        row = test_info[test_info['unique_id'] == unique_id]
        if row.empty:
            continue

        data = pd.read_csv(file)

        # Add play_mode column
        play_mode = row['mode'].iloc[0]
        data['play_mode'] = play_mode

        if 'weight' in data.columns:
            test_w = data['weight'].values
        else:
            test_w = np.ones(len(data))

        x_test = pd.concat([x_test, data], ignore_index=True)
        test_weights.extend(test_w)

    # === Normalize sample weights ===
    train_weights = np.array(train_weights)
    train_weights = train_weights / np.mean(train_weights)  # Optional: normalize

    # Normalize
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)



    # Label encoding
    le_gender = LabelEncoder()
    le_hold = LabelEncoder()
    le_years = LabelEncoder()
    le_level = LabelEncoder()

    y_gender = le_gender.fit_transform(y_train['gender'])
    y_hold = le_hold.fit_transform(y_train['hold racket handed'])
    y_years = le_years.fit_transform(y_train['play years'])
    y_level = le_level.fit_transform(y_train['level'])

    # Binary prediction helper
    def predict_gender(X_train, y_train, X_test, weights, seed):
        base_learners = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=seed)),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=seed)),
            ('lgb', LGBMClassifier(objective='binary', random_state=seed)),
            ('cat', CatBoostClassifier(loss_function='Logloss', verbose=0, random_state=seed))
        ]

        meta_model = LogisticRegression(max_iter=300, multi_class='auto', solver='lbfgs')

        clf = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1,
            passthrough=True  # Optional, gives meta-model access to original features
        )

        clf.fit(X_train, y_train, sample_weight=weights)
        proba = clf.predict_proba(X_test)
        pos_probs = [p[1] for p in proba]
        group_preds = [max(pos_probs[i*group_size:(i+1)*group_size]) for i in range(len(pos_probs)//group_size)]
        return group_preds
    
    def predict_hand(X_train, y_train, X_test, weights, seed):
        clf = CatBoostClassifier(
            loss_function='MultiClass',
            random_state=seed,
            verbose=0
          )  # Suppress training output        
        clf.fit(X_train, y_train, sample_weight=weights)
        proba = clf.predict_proba(X_test)
        pos_probs = [p[1] for p in proba]
        group_preds = [max(pos_probs[i*group_size:(i+1)*group_size]) for i in range(len(pos_probs)//group_size)]
        
        return group_preds

    # Multi-class prediction helper
    def predict_year(X_train, y_train, X_test, num_class, weights, seed):
        clf = RandomForestClassifier(random_state=seed)
        clf.fit(X_train, y_train, sample_weight=weights)
        proba = clf.predict_proba(X_test)
        pred_list = []
        for i in range(len(proba) // group_size):
            group = proba[i*group_size:(i+1)*group_size]
            class_sums = [sum([g[j] for g in group]) for j in range(num_class)]
            chosen_class = np.argmax(class_sums)
            best_idx = np.argmax([g[chosen_class] for g in group])
            pred_list.append(group[best_idx])  # softmax-like
        return pred_list

    def predict_level(X_train, y_train, X_test, num_class, weights, seed):
        clf = CatBoostClassifier(
            loss_function='MultiClass',
            random_state=seed,
            verbose=0  # Suppress training output
            )
        clf.fit(X_train, y_train, sample_weight=weights)
        proba = clf.predict_proba(X_test)
        pred_list = []
        for i in range(len(proba) // group_size):
            group = proba[i*group_size:(i+1)*group_size]
            class_sums = [sum([g[j] for g in group]) for j in range(num_class)]
            chosen_class = np.argmax(class_sums)
            best_idx = np.argmax([g[chosen_class] for g in group])
            pred_list.append(group[best_idx])  # softmax-like
        return pred_list

    # Generate predictions
    pred_gender = predict_gender(X_train_scaled, y_gender, X_test_scaled, train_weights, seed)
    pred_hold = predict_hand(X_train_scaled, y_hold, X_test_scaled, train_weights, seed)
    pred_years = predict_year(X_train_scaled, y_years, X_test_scaled, len(le_years.classes_), train_weights, seed)
    pred_level = predict_level(X_train_scaled, y_level, X_test_scaled, len(le_level.classes_), train_weights, seed)

    # Submission formatting
    year_cols = [f'play years_{i}' for i in range(len(le_years.classes_))]
    level_cols = [f'level_{i+2}' for i in range(len(le_level.classes_))]

    result_df = pd.DataFrame()
    result_df['unique_id'] = test_ids
    result_df['gender'] = pred_gender
    result_df['hold racket handed'] = pred_hold

    # One-hot-ish probabilities for multi-class
    for i, probs in enumerate(pred_years):
        for j, col in enumerate(year_cols):
            if col not in result_df:
                result_df[col] = 0.0
            result_df.loc[i, col] = probs[j]

    for i, probs in enumerate(pred_level):
        for j, col in enumerate(level_cols):
            if col not in result_df:
                result_df[col] = 0.0
            result_df.loc[i, col] = probs[j]

    # Final column ordering
    result_df = result_df[['unique_id', 'gender', 'hold racket handed'] + year_cols + level_cols]
    result_df = result_df.astype('float64').round(4)
    result_df['unique_id'] = np.array(test_ids).astype(int)

    # Invert specified columns
    result_df["gender"] = 1 - result_df["gender"]
    result_df["hold racket handed"] = 1 - result_df["hold racket handed"]

    # Round all numerical values to 4 decimal places
    result_df = result_df.round(4)


    print("🔍 Column Summary:")
    for col in result_df.columns:
        print(f"→ {col}:")
        print(f"   dtype: {result_df[col].dtype}")
        print(f"   sample values: {result_df[col].head(3).tolist()}")

    result_df.to_csv('submission.csv', index=False)
    print("✅ submission.csv generated!")

if __name__ == '__main__':
    main()