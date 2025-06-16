from pathlib import Path
import numpy as np
import pandas as pd
import math
import csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import os
import ast
#boost models
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

#stack model
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder


def main():
    random_seed = 1001
    
    # === Load player info & split train/test ===
    info = pd.read_csv('./39_Training_Dataset/train_info.csv')
    unique_players = info['player_id'].unique()
    train_players, test_players = train_test_split(unique_players, test_size=0.2, random_state=random_seed)
    
    # === Load tabular feature CSVs ===
    datapath = './39_Training_Dataset/tabular_data_train'
    datalist = list(Path(datapath).glob('**/*.csv'))
    target_mask = ['gender', 'hold racket handed', 'play years', 'level']

    # === Initialize data holders ===
    x_train = pd.DataFrame()
    y_train = pd.DataFrame(columns=target_mask)
    train_weights = []

    x_test = pd.DataFrame()
    y_test = pd.DataFrame(columns=target_mask)
    test_weights = []

    # === Load data and weights ===
    for file in datalist:
        unique_id = int(Path(file).stem)
        row = info[info['unique_id'] == unique_id]
        if row.empty:
            continue

        player_id = row['player_id'].iloc[0]
        data = pd.read_csv(file)

        if 'weight' in data.columns:
            file_weights = data['weight'].values
            data = data.drop(columns=['weight'])
        else:
            file_weights = np.ones(len(data))

        # Add play_mode column
        play_mode = row['mode'].iloc[0]
        data['play_mode'] = play_mode

        target = row[target_mask]
        target_repeated = pd.concat([target] * len(data), ignore_index=True)

        if player_id in train_players:
            x_train = pd.concat([x_train, data], ignore_index=True)
            y_train = pd.concat([y_train, target_repeated], ignore_index=True)
            train_weights.extend(file_weights)
        elif player_id in test_players:
            x_test = pd.concat([x_test, data], ignore_index=True)
            y_test = pd.concat([y_test, target_repeated], ignore_index=True)
            test_weights.extend(file_weights)

    # === Normalize sample weights ===
    train_weights = np.array(train_weights)
    train_weights = train_weights / np.mean(train_weights)  # Optional: normalize

    # === Standardize features ===
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)

    group_size = 27

    def model_gender(X_train, y_train, X_test, y_test, weights, random_seed):
        clf = CatBoostClassifier(
            loss_function='MultiClass',
            random_state=random_seed,
            verbose=0  # Suppress training output
            )

        clf.fit(X_train, y_train, sample_weight=weights)
        
        predicted = clf.predict_proba(X_test,)
        # 取出正類（index 1）的概率
        predicted = [predicted[i][1] for i in range(len(predicted))]

        
        num_groups = len(predicted) // group_size 
        if sum(predicted[:group_size]) / group_size > 0.5:
            y_pred = [max(predicted[i*group_size: (i+1)*group_size]) for i in range(num_groups)]
        else:
            y_pred = [min(predicted[i*group_size: (i+1)*group_size]) for i in range(num_groups)]
        
        
        y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
        
        auc_score = roc_auc_score(y_test_agg, y_pred, average='micro')
        print('Gender AUC:', auc_score)

    def model_hand(X_train, y_train, X_test, y_test, weights, random_seed):
        clf = CatBoostClassifier(
            loss_function='MultiClass',
            random_state=random_seed,
            verbose=0  # Suppress training output
            )
        clf.fit(X_train, y_train, sample_weight=weights)
        
        predicted = clf.predict_proba(X_test)
        # 取出正類（index 1）的概率
        predicted = [predicted[i][1] for i in range(len(predicted))]

        
        num_groups = len(predicted) // group_size 
        if sum(predicted[:group_size]) / group_size > 0.5:
            y_pred = [max(predicted[i*group_size: (i+1)*group_size]) for i in range(num_groups)]
        else:
            y_pred = [min(predicted[i*group_size: (i+1)*group_size]) for i in range(num_groups)]
        
        
        y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
        
        auc_score = roc_auc_score(y_test_agg, y_pred, average='micro')
        print('Play Hand AUC:', auc_score)

    # 定義多類別分類評分函數 (例如 play years、level)
    def model_year(X_train, y_train, X_test, y_test, weights, random_seed):
        clf = CatBoostClassifier(
            loss_function='MultiClass',
            random_state=random_seed,
            verbose=0  # Suppress training output
            )


        clf.fit(X_train, y_train, sample_weight=weights)
        predicted = clf.predict_proba(X_test)
        num_groups = len(predicted) // group_size
        y_pred = []
        for i in range(num_groups):
            group_pred = predicted[i*group_size: (i+1)*group_size]
            num_classes = len(np.unique(y_train))
            # 對每個類別計算該組內的總機率
            class_sums = [sum([group_pred[k][j] for k in range(group_size)]) for j in range(num_classes)]
            chosen_class = np.argmax(class_sums)
            candidate_probs = [group_pred[k][chosen_class] for k in range(group_size)]
            best_instance = np.argmax(candidate_probs)
            y_pred.append(group_pred[best_instance])
        
        y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
        auc_score = roc_auc_score(y_test_agg, y_pred, average='micro', multi_class='ovr')
        print('Play Year AUC:', auc_score)

    def model_level(X_train, y_train, X_test, y_test, weights, random_seed):
        clf = CatBoostClassifier(
            loss_function='MultiClass',
            random_state=random_seed,
            verbose=0  # Suppress training output
            )
        clf.fit(X_train, y_train, sample_weight=weights)
        predicted = clf.predict_proba(X_test)
        num_groups = len(predicted) // group_size
        y_pred = []
        for i in range(num_groups):
            group_pred = predicted[i*group_size: (i+1)*group_size]
            num_classes = len(np.unique(y_train))
            # 對每個類別計算該組內的總機率
            class_sums = [sum([group_pred[k][j] for k in range(group_size)]) for j in range(num_classes)]
            chosen_class = np.argmax(class_sums)
            candidate_probs = [group_pred[k][chosen_class] for k in range(group_size)]
            best_instance = np.argmax(candidate_probs)
            y_pred.append(group_pred[best_instance])
        
        y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
        auc_score = roc_auc_score(y_test_agg, y_pred, average='micro', multi_class='ovr')
        print('Level AUC:', auc_score)

    # 評分：針對各目標進行模型訓練與評分
    le = LabelEncoder()
    y_train_le_gender = le.fit_transform(y_train['gender'])
    y_test_le_gender = le.transform(y_test['gender'])
    model_gender(X_train_scaled, y_train_le_gender, X_test_scaled, y_test_le_gender, train_weights, random_seed)
    
    y_train_le_hold = le.fit_transform(y_train['hold racket handed'])
    y_test_le_hold = le.transform(y_test['hold racket handed'])
    model_hand(X_train_scaled, y_train_le_hold, X_test_scaled, y_test_le_hold, train_weights, random_seed)
    
    y_train_le_years = le.fit_transform(y_train['play years'])
    y_test_le_years = le.transform(y_test['play years'])
    model_year(X_train_scaled, y_train_le_years, X_test_scaled, y_test_le_years, train_weights, random_seed)
    
    y_train_le_level = le.fit_transform(y_train['level'])
    y_test_le_level = le.transform(y_test['level'])
    model_level(X_train_scaled, y_train_le_level, X_test_scaled, y_test_le_level, train_weights, random_seed)

    # AUC SCORE: 0.792(gender) + 0.998(hold) + 0.660(years) + 0.822(levels)

if __name__ == '__main__':
    #data_generate()
    main()



    ### MODEL BANK

        # clf = RandomForestClassifier(random_state=random_seed)

        # clf = CatBoostClassifier(
        #     loss_function='MultiClass',
        #     random_state=random_seed,
        #     verbose=0  # Suppress training output
        #     )

        # clf = lgb.LGBMClassifier(
        #     objective='multiclass',
        #     num_class=len(np.unique(y_train)),
        #     random_state=random_seed
        # )

        # clf = XGBClassifier(
        #     objective='multi:softprob',  # Enables multiclass probability output
        #     num_class=len(np.unique(y_train)),  # Specify number of classes
        #     eval_metric='mlogloss',  # Standard for multiclass
        #     use_label_encoder=False,
        #     random_state=random_seed
        # )

        # base_learners = [
        #     ('lr', LogisticRegression(max_iter=300)),
        #     ('mlp', MLPClassifier(hidden_layer_sizes=(64,))),
        #     ('knn', KNeighborsClassifier(n_neighbors=5)),
        #     ('rf', RandomForestClassifier(n_estimators=100)),
        #     ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
        # ]
        # meta_model = LogisticRegression()
        # clf = StackingClassifier(estimators=base_learners, final_estimator=meta_model, cv=5)

        # clf = KNeighborsClassifier(n_neighbors=5)

