# 🧠 Deep Learning-Based Solution (DL_based)

This folder contains the deep learning pipeline for predicting table tennis player attributes based on raw IMU sensor data. The model architecture is based on a 1D Convolutional Neural Network (1D-CNN) designed to capture temporal patterns in the 6-axis IMU signal.

## 📁 Folder Structure

```
DL_based/
├── model_pth/
│   ├── model_weights_gd.pth         # Trained model weights for gender prediction
│   ├── model_weights_lv.pth         # Trained model weights for level prediction
│   └── model_weights_yr.pth         # Trained model weights for play years 
├── RawCNN_train.py                  # Script for training deep learning models
├── RawCNN_inference.py              # Script for generating predictions
└── README.md                        # This documentation
```

---

## 🚀 Deep Learning Workflow

### Step 1: Train Models

Run the training script to train the models for:
- Gender (`gender`)
- Play years (`play years`)
- Level (`level`)

```bash
python RawCNN_train.py
```

Trained model weights will be saved in the `model_pth/` directory. Correct any path in `__main__` or create folders if needed.

> Note: We recommend using a GPU for training.

### Step 2: Run Inference

After training, generate predictions on the test set by running:

```bash
python RawCNN_inference.py
```

This will output CSV files for each task (gender, years, level) in the `../csv_folder/`.

---

## 📬 Contact

For questions, please refer to the root [README](../README.md) or open an issue.
