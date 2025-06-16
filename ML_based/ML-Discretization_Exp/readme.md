# 🧠 Machine Learning Based Prediction Pipeline

This module implements a classic machine learning approach built upon handcrafted statistical features extracted from IMU swing cycle data. The pipeline transforms variable-length time-series sensor data into structured feature vectors, applying discretization and PCA to enhance representation. XGBoost classifiers are trained independently for each prediction task: gender, handedness, years of experience, and playing level.

---

## 🔧 Prediction Pipeline Structure

```bash
🔹 gen_data.py               # Main preprocessing script for feature extraction, discretization, PCA, and outlier detection
🔹 base_line_infernece.py    # Script that loads trained models and performs inference
🔹 base_line_inference_test.py # Script used for validation on held-out data
🔹 distribution.py           # Displays class distribution for training data
🔹 readme.md                 # This documentation
```

---

## 📌 Step-by-Step Execution

To reproduce the full pipeline and obtain predictions, follow this sequence:

```bash
# Step 1: Preprocess raw data and extract features
python3 gen_data.py

# Step 2: Run inference to generate predictions
python3 base_line_infernece.py
```

This will generate processed datasets with features and weights, as well as prediction output files.

If you wish to validate model performance on a held-out set:

```bash
python3 base_line_inference_test.py
```

---

## 📘 Methodology Overview

### 🏓 Data Preprocessing & Feature Engineering

* `gen_data.py` traverses the raw IMU data folder and segments each record using annotated `cut_point` timestamps.
* Initial and final segments—often contaminated by swing preparation or cooldown—are discarded.
* For each valid swing cycle, high-level statistical features are extracted from the 6-axis IMU signals (Ax, Ay, Az, Gx, Gy, Gz).
* Features undergo **discretization** (e.g., quantile, k-means) to highlight distributional patterns and are further transformed using **PCA** to reduce dimensionality and emphasize principal variations.
* Outliers are detected, and a `weight` column is assigned to down-weight or exclude unreliable samples during training or inference.

### 📊 Class Distribution Inspection

* `distribution.py` allows for quick visualization and inspection of class distributions across tasks to assess balance or identify skewed classes.

### 🎯 Model Inference

* `base_line_infernece.py` applies pre-trained XGBoost classifiers to the processed data and outputs prediction CSVs.
* `base_line_inference_test.py` performs validation using separate validation data, useful for assessing model generalization.
* Four independent models are used:

  * Gender (binary)
  * Handedness (binary)
  * Play years (3-class)
  * Playing level (4-class)

Each classifier is optimized separately, using appropriate metrics (e.g., ROC AUC for binary tasks, micro-averaged AUC for multi-class tasks) and trained to handle class imbalance through reweighting if necessary.

---

