from pathlib import Path
import numpy as np
import pandas as pd
import math
import csv

from scipy.stats import kurtosis, skew, iqr
from sklearn.decomposition import PCA

import os
from scipy.signal import find_peaks


from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import entropy as scipy_entropy
from catboost import CatBoostClassifier, Pool

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing")


# === Outlier Detection on DataFrame ===
#Strategies: ["quantile", "kmeans", "uniform"]
from catboost import CatBoostClassifier, Pool

def prune_features_with_catboost(df, threshold=0.1, random_seed=42):
    if 'weight' in df.columns:
        X = df.drop(columns=['weight'])
    else:
        X = df.copy()

    y_dummy = np.random.randint(0, 2, len(X))  # Dummy target for feature importance
    
    model = CatBoostClassifier(
        iterations=50,
        learning_rate=0.1,
        depth=6,
        verbose=0,
        random_seed=random_seed
    )
    model.fit(X, y_dummy)

    importances = model.get_feature_importance(Pool(X, y_dummy))
    importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
    importance_df.sort_values(by='importance', inplace=True)

    low_importance_features = importance_df[importance_df['importance'] < threshold]['feature'].tolist()

    print(f"[CatBoost Prune] Dropping {len(low_importance_features)} features: {low_importance_features}")
    df_pruned = df.drop(columns=low_importance_features)
    
    return df_pruned



def try_discretize(col_values, strategy="quantile", n_bins=5):
    try:
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        discretizer.fit(col_values)
        n_bins_used = len(discretizer.bin_edges_[0]) - 1
        if n_bins_used > 1:
            return discretizer.transform(col_values).astype(int)
    except Exception as e:
        pass  # You can log or print(e) here if desired
    return None  # Strategy failed


def pca_on_continuous_features(full_df, pca_threshold=0.9):
    assert 'weight' in full_df.columns, "full_df must include 'weight' column"
    cont_cols = full_df.columns.difference(['weight'])
    df_cont = full_df[cont_cols]

    pca = PCA(n_components=pca_threshold)
    pca_components = pca.fit_transform(df_cont)
    pca_df = pd.DataFrame(pca_components, columns=[f'cont_pca_{i}' for i in range(pca.n_components_)])
    
    print(f"[PCA:Continuous] Retained {pca.n_components_} components for >{pca_threshold*100:.1f}% variance")

    df_combined = pd.concat([full_df.reset_index(drop=True), pca_df], axis=1)
    return df_combined, pca


def pca_on_discretized_features(full_df, strategies=["quantile", "kmeans", "uniform"], bin_list=[5, 3], pca_threshold=0.9):
    assert 'weight' in full_df.columns, "full_df must include 'weight' column"
    cont_cols = full_df.columns.difference(['weight'])

    # === Discretization ===
    discretized_cols = {}
    for col in cont_cols:
        col_values = full_df[[col]].values
        if len(np.unique(col_values)) < 2:
            continue
        for strategy in strategies:
            for n_bin in bin_list:
                result = try_discretize(col_values, strategy=strategy, n_bins=n_bin)
                if result is not None:
                    key = f"{col}_bin_{strategy}_{n_bin}"
                    discretized_cols[key] = result.flatten().astype(int)

    df_discrete = pd.DataFrame(discretized_cols)

    if df_discrete.empty:
        print("[PCA:Discretized] No valid discretized features.")
        return full_df, None

    # === PCA on Discretized Features ===
    pca = PCA(n_components=pca_threshold)
    pca_components = pca.fit_transform(df_discrete)
    pca_df = pd.DataFrame(pca_components, columns=[f'disc_pca_{i}' for i in range(pca.n_components_)])

    print(f"[PCA:Discretized] Retained {pca.n_components_} components for >{pca_threshold*100:.1f}% variance")

    df_combined = pd.concat([full_df.reset_index(drop=True), pca_df], axis=1)
    return df_combined, pca


def align_and_save(train_dict, test_dict, out_train_dir, out_test_dir):
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)

    for key in train_dict:
        if key not in test_dict:
            print(f"Missing {key} in test set. Skipping.")
            continue

        df_train = train_dict[key]
        df_test = test_dict[key]

        # Keep only shared bin columns + all non-bin columns
        bin_cols_train = [c for c in df_train.columns if c.endswith('_bin')]
        bin_cols_test = [c for c in df_test.columns if c.endswith('_bin')]

        shared_bins = list(set(bin_cols_train) & set(bin_cols_test))

        base_cols_train = [c for c in df_train.columns if not c.endswith('_bin')]
        base_cols_test = [c for c in df_test.columns if not c.endswith('_bin')]
        shared_base = list(set(base_cols_train) & set(base_cols_test))

        final_cols = shared_base + shared_bins
        df_train = df_train[final_cols]
        df_test = df_test[final_cols]

        df_train.to_csv(os.path.join(out_train_dir, f"{key}.csv"), index=False)
        df_test.to_csv(os.path.join(out_test_dir, f"{key}.csv"), index=False)


def detect_outliers(df, random_seed, method='zscore', threshold=3.0):
    if method == 'zscore':
        z_scores = np.abs(zscore(df.select_dtypes(include=[np.number])))
        outlier_mask = (z_scores > threshold).any(axis=1)
        return outlier_mask.tolist()
    elif method == 'iqr':
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
        return outlier_mask.tolist()
    elif method == 'isolationforest':
        clf = IsolationForest(contamination=0.05, random_state=random_seed)
        is_outlier = clf.fit_predict(df.select_dtypes(include=[np.number]))
        return (is_outlier == -1).tolist()
    else:
        raise ValueError("Unsupported method. Choose from 'zscore', 'iqr', or 'isolationforest'.")


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

# === Subsegment Filtering ===
def FFT(xreal, ximag):    
    n = 2
    while(n*2 <= len(xreal)):
        n *= 2
    
    p = int(math.log(n, 2))
    
    for i in range(0, n):
        a = i
        b = 0
        for j in range(0, p):
            b = int(b*2 + a%2)
            a = a/2
        if(b > i):
            xreal[i], xreal[b] = xreal[b], xreal[i]
            ximag[i], ximag[b] = ximag[b], ximag[i]
            
    wreal = []
    wimag = []
        
    arg = float(-2 * math.pi / n)
    treal = float(math.cos(arg))
    timag = float(math.sin(arg))
    
    wreal.append(float(1.0))
    wimag.append(float(0.0))
    
    for j in range(1, int(n/2)):
        wreal.append(wreal[-1] * treal - wimag[-1] * timag)
        wimag.append(wreal[-1] * timag + wimag[-1] * treal)
        
    m = 2
    while(m < n + 1):
        for k in range(0, n, m):
            for j in range(0, int(m/2), 1):
                index1 = k + j
                index2 = int(index1 + m / 2)
                t = int(n * j / m)
                treal = wreal[t] * xreal[index2] - wimag[t] * ximag[index2]
                timag = wreal[t] * ximag[index2] + wimag[t] * xreal[index2]
                ureal = xreal[index1]
                uimag = ximag[index1]
                xreal[index1] = ureal + treal
                ximag[index1] = uimag + timag
                xreal[index2] = ureal - treal
                ximag[index2] = uimag - timag
        m *= 2
        
    return n, xreal, ximag   
    
def FFT_data(input_data, swinging_times):   
    txtlength = swinging_times[-1] - swinging_times[0]
    a_mean = [0] * txtlength
    g_mean = [0] * txtlength
       
    for num in range(len(swinging_times)-1):
        a = []
        g = []
        for swing in range(swinging_times[num], swinging_times[num+1]):
            a.append(math.sqrt(input_data[swing][0]**2 + input_data[swing][1]**2 + input_data[swing][2]**2)) # correct
            g.append(math.sqrt(input_data[swing][3]**2 + input_data[swing][4]**2 + input_data[swing][5]**2)) # correct
            # a.append(math.sqrt(math.pow((input_data[swing][0] + input_data[swing][1] + input_data[swing][2]), 2))) # error
            # g.append(math.sqrt(math.pow((input_data[swing][3] + input_data[swing][4] + input_data[swing][5]), 2))) # error

        a_mean[num] = (sum(a) / len(a))
        g_mean[num] = (sum(g) / len(g)) #fixed
    
    return a_mean, g_mean



def feature(input_data, swinging_now, swinging_times, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag):
    eps = 1e-8
    allsum = [0] * 6
    var = [0] * 6
    rms = [0] * 6
    a = []
    g = []
    signals = [[] for _ in range(6)]  # ax, ay, az, gx, gy, gz

    for row in input_data:
        ax, ay, az, gx, gy, gz = row
        a_mag = math.sqrt(ax**2 + ay**2 + az**2)
        g_mag = math.sqrt(gx**2 + gy**2 + gz**2)
        a.append(a_mag)
        g.append(g_mag)
        for i in range(6):
            value = max(min(row[i], 1e6), -1e6)
            signals[i].append(value)
            allsum[i] += value
            var[i] += value**2
            rms[i] += min(value**2, 1e12)

    mean = [s / len(input_data) for s in allsum]
    var = [math.sqrt((v / len(input_data)) - (m ** 2)) for v, m in zip(var, mean)]
    rms = [math.sqrt(r / len(input_data)) for r in rms]

    a_max = [max(a)]
    a_min = [min(a)]
    a_mean = [sum(a) / len(a) if len(a) > 0 else 0]
    g_max = [max(g)]
    g_min = [min(g)]
    g_mean = [sum(g) / len(g) if len(g) > 0 else 0]

    a_kurtosis, g_kurtosis = [0], [0]
    a_skewness, g_skewness = [0], [0]

    a_var_mag = [np.var(a)]
    g_var_mag = [np.var(g)]

    if len(input_data) > 0:
        a_mean_val = a_mean[0]
        g_mean_val = g_mean[0]
        a_diff = [ai - a_mean_val for ai in a]
        g_diff = [gi - g_mean_val for gi in g]

        a_s2 = sum(x**2 for x in a_diff) / len(input_data)
        g_s2 = sum(x**2 for x in g_diff) / len(input_data)

        if a_s2 > eps:
            a_s4 = sum(x**4 for x in a_diff) / len(input_data)
            a_s3 = sum(x**3 for x in a_diff) / len(input_data)
            a_kurtosis = [a_s4 / (a_s2**2)]
            a_skewness = [a_s3 / (a_s2**1.5 + eps)]

        if g_s2 > eps:
            g_s4 = sum(x**4 for x in g_diff) / len(input_data)
            g_s3 = sum(x**3 for x in g_diff) / len(input_data)
            g_kurtosis = [g_s4 / (g_s2**2)]
            g_skewness = [g_s3 / (g_s2**1.5 + eps)]

    cut = int(n_fft / swinging_times)
    start = cut * swinging_now
    end = cut * (swinging_now + 1)

    a_fft_seg = a_fft[start:end]
    g_fft_seg = g_fft[start:end]
    a_fft_imag_seg = a_fft_imag[start:end]
    g_fft_imag_seg = g_fft_imag[start:end]

    a_fft_mean = np.mean(a_fft_seg)
    g_fft_mean = np.mean(g_fft_seg)

    a_psd = [re**2 + im**2 for re, im in zip(a_fft_seg, a_fft_imag_seg)]
    g_psd = [re**2 + im**2 for re, im in zip(g_fft_seg, g_fft_imag_seg)]

    a_psd_mean = np.mean(a_psd)
    g_psd_mean = np.mean(g_psd)

    a_energy = [math.sqrt(max(p, 0)) for p in a_psd]
    g_energy = [math.sqrt(max(p, 0)) for p in g_psd]

    a_total_energy = sum(a_energy) + eps
    g_total_energy = sum(g_energy) + eps

    entropy_a = [-max(e / a_total_energy, eps) * math.log(max(e / a_total_energy, eps)) for e in a_energy]
    entropy_g = [-max(e / g_total_energy, eps) * math.log(max(e / g_total_energy, eps)) for e in g_energy]

    a_entropy_mean = sum(entropy_a) / len(entropy_a) if entropy_a else 0
    g_entropy_mean = sum(entropy_g) / len(entropy_g) if entropy_g else 0

    gx_saturation = sum(1 for row in input_data if row[3] == 32768 or row[3] == -32768)
    gy_saturation = sum(1 for row in input_data if row[4] == 32768 or row[4] == -32768)
    gz_saturation = sum(1 for row in input_data if row[5] == 32768 or row[5] == -32768)

    speed = np.cumsum(a).tolist()
    speed_mean = [sum(speed) / len(speed)]
    speed_max = [max(speed)]
    speed_min = [min(speed)]

    # my feature
    jerk = np.diff(a)
    jerk_mean = [np.mean(np.abs(jerk))]
    jerk_std = [np.std(jerk)]

    dominant_freq_a = [np.argmax(np.abs(a_fft_seg))]
    dominant_freq_g = [np.argmax(np.abs(g_fft_seg))]

    asymmetry_ax = [np.abs(np.mean(signals[0][:len(signals[0]) // 2]) - np.mean(signals[0][len(signals[0]) // 2:]))]
    asymmetry_gx = [np.abs(np.mean(signals[3][:len(signals[3]) // 2]) - np.mean(signals[3][len(signals[3]) // 2:]))]

    autocorr_ax = [np.correlate(signals[0], signals[0], mode='full')[len(signals[0])//2 + 1]]
    autocorr_gx = [np.correlate(signals[3], signals[3], mode='full')[len(signals[3])//2 + 1]]

    peak_count_ax = [len(find_peaks(signals[0])[0])]
    peak_count_gx = [len(find_peaks(signals[3])[0])]

    fft_range_a = [np.ptp(a_fft_seg)]
    fft_range_g = [np.ptp(g_fft_seg)]

    #feature part 2
    swing_duration = [len(input_data)]
    jerk_entropy = [scipy_entropy(np.abs(jerk) + eps)]
    jerk_zcr = [((jerk[:-1] * jerk[1:]) < 0).sum()]
    snap = np.diff(jerk)
    snap_zcr = [((snap[:-1] * snap[1:]) < 0).sum()]
    a_high_energy = sum(a_psd[len(a_psd)//2:]) + eps
    a_low_energy = sum(a_psd[:len(a_psd)//2]) + eps
    a_energy_ratio = [a_high_energy / a_low_energy]

    # ===== New Features from code 2 =====
    extended_features = []
    for signal in signals:
        signal = np.array(signal)
        hist, _ = np.histogram(signal, bins=20, density=True)
        hist += 1e-12  # prevent log(0)
        entropy = -np.sum(hist * np.log2(hist))
        if np.std(signal) < 1e-8:
            skew_val = 0
            kurt_val = 0
        else:
            skew_val = skew(signal)
            kurt_val = kurtosis(signal)
        
        feats = [
            np.median(signal),
            np.max(signal) - np.min(signal),  # range
            iqr(signal),
            skew_val,
            kurt_val,
            np.sum(signal ** 2),
            np.sqrt(np.mean(signal ** 2)),
            ((signal[:-1] * signal[1:]) < 0).sum(),  # ZCR
            np.sum(np.abs(signal)) / len(signal),
            entropy,
            np.ptp(signal)
        ]
        extended_features.extend(feats)

    return (
        mean + var + rms + a_max + a_mean + a_min +
        g_max + g_mean + g_min + [a_fft_mean] + [g_fft_mean] +
        [a_psd_mean] + [g_psd_mean] +
        a_var_mag + g_var_mag +
        a_kurtosis + g_kurtosis + a_skewness + g_skewness +
        [a_entropy_mean] + [g_entropy_mean] +
        [gx_saturation, gy_saturation, gz_saturation] +
        speed_max + speed_mean + speed_min +

        #my feature pt1
        jerk_mean + jerk_std +
        dominant_freq_a + dominant_freq_g +
        asymmetry_ax + asymmetry_gx +
        autocorr_ax + autocorr_gx +
        peak_count_ax + peak_count_gx +
        fft_range_a + fft_range_g +

        #my feature pt2
        swing_duration + jerk_entropy + jerk_zcr + snap_zcr + a_energy_ratio +
        
        #kuan feature
        extended_features
    )



def data_generate(random_seed, discretize=True):
    modes = ["train", "test"]
    all_data = []  # List of tuples: (mode, filename, df_feat)

    for mode in modes:
        root = f'./39_Training_Dataset/' if mode == "train" else f'./39_Test_Dataset/'
        datapath = os.path.join(root, f'{mode}_data')
        pathlist_txt = list(Path(datapath).glob('**/*.txt'))

        for file in pathlist_txt:
            with open(file) as f:
                All_data = []
                for i, line in enumerate(f):
                    if line.strip() == '' or i == 0:
                        continue
                    num = line.split(' ')
                    if len(num) >= 6:
                        All_data.append([int(num[j]) for j in range(6)])

            swing_index = np.linspace(0, len(All_data), 28, dtype=int)
            try:
                a_fft, g_fft = FFT_data(All_data, swing_index)
                a_fft_imag = [0] * len(a_fft)
                g_fft_imag = [0] * len(g_fft)
                n_fft, a_fft, a_fft_imag = FFT(a_fft, a_fft_imag)
                n_fft, g_fft, g_fft_imag = FFT(g_fft, g_fft_imag)

                feature_rows = []
                for i in range(1, len(swing_index)):
                    segment = All_data[swing_index[i - 1]: swing_index[i]]
                    row_feat = feature(segment, i - 1, len(swing_index) - 1, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag)
                    feature_rows.append(row_feat)
            except Exception as e:
                print(f"Exception in file {Path(file).stem}: {e}")
                continue

            if not feature_rows:
                continue

            headerList = [
                'ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean',
                'ax_std', 'ay_std', 'az_std', 'gx_std', 'gy_std', 'gz_std',
                'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms',
                'a_max', 'a_mean', 'a_min',
                'g_max', 'g_mean', 'g_min',
                'a_fft_mean', 'g_fft_mean',
                'a_psd_mean', 'g_psd_mean',
                'a_var_mag', 'g_var_mag',
                'a_kurtosis', 'g_kurtosis', 'a_skewness', 'g_skewness',
                'a_entropy', 'g_entropy',
                'gx_saturation_count', 'gy_saturation_count', 'gz_saturation_count',
                'speed_max', 'speed_mean', 'speed_min',

                'jerk_mean', 'jerk_std',
                'a_dominant_freq', 'g_dominant_freq',
                'ax_asymmetry', 'gx_asymmetry',
                'ax_autocorr1', 'gx_autocorr1',
                'ax_peak_count', 'gx_peak_count',
                'a_fft_range', 'g_fft_range',

                'swing_duration',
                'jerk_entropy',
                'jerk_zcr',
                'snap_zcr',
                'a_energy_ratio',
            ]
            labels = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
            code2_stats = [
                'median', 'range', 'iqr', 'skew', 'kurtosis', 'energy',
                'rms2', 'zcr', 'sma', 'entropy_hist', 'ptp'
            ]
            for label in labels:
                for stat in code2_stats:
                    headerList.append(f"{label}_{stat}")

            df_feat = pd.DataFrame(feature_rows, columns=headerList)
            outlier_mask = detect_outliers(df_feat, random_seed=random_seed, method='isolationforest')
            df_feat['weight'] = [20.0 if is_outlier else 1.0 for is_outlier in outlier_mask]
            all_data.append((mode, Path(file).stem, df_feat))

    full_df = pd.concat([df for _, _, df in all_data], axis=0, ignore_index=True)
    full_df = prune_features_with_catboost(full_df, threshold=0.01, random_seed=random_seed)

    # === PCA on continuous features ===
    #full_df, _ = pca_on_continuous_features(full_df)

    # === PCA on discretized version ===
    if discretize:
        full_df, _ = pca_on_discretized_features(full_df)

    # === Split and save ===
    offset = 0
    for mode, filename, df_feat in all_data:
        rows = len(df_feat)
        df_final = full_df.iloc[offset:offset + rows].copy()
        offset += rows

        tar_dir = f'./39_Training_Dataset/tabular_data_train' if mode == "train" else f'./39_Test_Dataset/tabular_data_test'
        os.makedirs(tar_dir, exist_ok=True)
        df_final.to_csv(os.path.join(tar_dir, f'{filename}.csv'), index=False)





random_seed = 1001
data_generate(random_seed, discretize=True)

