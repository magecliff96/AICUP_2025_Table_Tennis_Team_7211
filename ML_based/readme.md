# Prediction Pipeline


```bash
├── train_info.csv          # 訓練資料的基本資訊
├── test_info.csv           # 測試資料的基本資訊
|
├── train_data/             # 訓練資料 CSV 檔案
│   ├── 1.csv
│   ├── 2.csv
│   ├── ...
│   └── 1967.csv
|
├── test_data/              # 測試資料 CSV 檔案
│   ├── 1968.csv
│   ├── 1969.csv
│   ├── ...
│   └── 3411.csv
|
├── test_preprocess.py      # 測試資料預處理腳本
├── train_preprocess.py     # 訓練資料預處理腳本
├── main_pred.py            # 預測主程式
└── log.txt                 # 執行過程的輸出紀錄（自動產生）
```

## 📌 Step-by-Step Execution

請依照以下順序執行三個 Python 腳本，並將所有輸出記錄到 `log.txt`：

```bash
{
  python3 test_preprocess.py
  python3 train_preprocess.py
  python3 main_pred.py
} > log.txt 2>&1
