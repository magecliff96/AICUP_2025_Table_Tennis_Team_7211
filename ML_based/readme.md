# Prediction Pipeline

```bash
├── test_data_v2.csv        # intermdeiate file
├── test_data.csv           # intermdeiate file
├── train_data_v2.csv       # intermdeiate file
├── train_data.csv          # intermdeiate file
├── pca_feature.csv         # intermdeiate file
├── avg_cycles_txt/         # intermdeiate folder
├── test_preprocess.py      # Preprocessing script for the test data
├── train_preprocess.py     # Preprocessing script for the training data
├── main_pred.py            # Main prediction script
└── log.txt                 # Output log of the execution (generated automatically)
```

## 📌 Step-by-Step Execution

Please run the following three Python scripts in order, and redirect all output to `log.txt`:

```bash
{
  python3 test_preprocess.py
  python3 train_preprocess.py
  python3 main_pred.py
} > log.txt 2>&1
```
