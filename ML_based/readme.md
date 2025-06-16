# Prediction Pipeline

```bash
├── train_info.csv          # Basic information for the training data
├── test_info.csv           # Basic information for the test data
|
├── train_data/             # Training data CSV files
│   ├── 1.txt
│   ├── 2.txt
│   ├── ...
│   └── 1967.txt
|
├── test_data/              # Test data CSV files
│   ├── 1968.txt
│   ├── 1969.txt
│   ├── ...
│   └── 3411.txt
|
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
