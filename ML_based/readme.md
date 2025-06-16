# 🧠 Machine Learning Based Prediction Pipeline

This module uses a classic machine learning approach built upon handcrafted statistical features extracted from IMU swing cycle data. The pipeline focuses on transforming variable-length time-series sensor data into consistent, meaningful feature vectors, then training separate XGBoost classifiers for each prediction task: gender, handedness, years of experience, and playing level. A seperate folder named ML_Discretization_Exp also contains additional code used for testing ML models on data with discretization.
---

## 🔧 Prediction Pipeline Structure

```bash
├── test_data_v2.csv        # Intermediate file after processing 
├── test_data.csv           # Intermediate file after processing 
├── train_data_v2.csv       # Intermediate file after processing 
├── train_data.csv          # Intermediate file after processing 
├── pca_feature.csv         # Intermediate file after processing 
├── avg_cycles_txt/         # Intermediate folder after processing 
├── test_preprocess.py      # Preprocessing script for test data
├── train_preprocess.py     # Preprocessing script for training data
├── main_pred.py            # Main prediction script (XGBoost classifiers)
└── log.txt                 # Output log of the execution (generated automatically)
```

---

## 📌 Step-by-Step Execution

To reproduce predictions using the machine learning pipeline, follow these steps in order:

```bash
{
  python3 test_preprocess.py
  python3 train_preprocess.py
  python3 main_pred.py
} > log.txt 2>&1
```

This will generate prediction CSVs which can be used or merged for final submission.

---

## 📘 Methodology Overview

### 🏓 Data Segmentation & Preprocessing

- Using annotated `cut_point` timestamps from the dataset, each motion record is split into discrete swing cycles.
- To reduce noise caused by motion initialization and termination, the first and last few cycles are discarded.
- The remaining swing cycles are aligned to the shortest segment length and averaged to form a single, representative swing cycle per player.

### 📈 Feature Extraction

From the averaged swing cycle, statistical features are extracted independently from all six IMU sensor axes (Ax, Ay, Az, Gx, Gy, Gz), including:

- **Central tendency**: mean, median  
- **Dispersion**: standard deviation, variance, IQR, range  
- **Shape**: skewness, kurtosis  
- **Signal energy**: RMS, energy  
- **Other**: entropy, ZCR (zero-crossing rate), SMA (signal magnitude area), PTP (peak-to-peak)

These features are concatenated to form the input vector for the classifiers.

### 🎯 Model Training & Prediction

- Four **independent XGBoost classifiers** are trained for:
  - Gender (binary)
  - Handedness (binary)
  - Play years (3 classes)
  - Playing level (4 classes)

Each classifier is trained and evaluated separately, with micro-averaged ROC AUC used for multi-class evaluation to address class imbalance.

---

> 💡 **Note**: This ML pipeline was developed modularly, and each step is independently executable and script-based. Please refer to the log file for detailed outputs.