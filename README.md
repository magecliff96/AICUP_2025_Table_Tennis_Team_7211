# AICUP 2025 Table Tennis Classification

This repository provides a complete solution for the AICUP 2025 Table Tennis Classification task using both classical machine learning and deep learning approaches. The codebase is organized into the following modules:

- `ML_based/`: Traditional machine learning pipeline with feature engineering
- `DL_based/`: Deep learning models using raw IMU data
- `csv_folder/`: Final CSV generation and merging utilities

---

## 📁 Repository Structure

```
.
├── ML_based/              # Machine Learning models with feature engineering
│   ├── README.md
│   └── ...
├── DL_based/              # Deep Learning models using raw IMU data
│   ├── README.md 
│   └── ...
├── csv_folder/            # Final prediction CSVs and merger script
│   ├── csv_merger.py 
│   └── ...
├── dataset/               # Original and augmented dataset
│   └── imbalance_aug.py
├── README.md              # Main project documentation (this file)
└── ...
```

---

## 🔧 Installation

1. It is highly recommended to use a virtual environment (e.g. `venv` or `conda`).
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## ⚡ Quick Submission Workflow

To directly obtain the final submission result without retraining:

1. Precomputed results are stored in the `csv_folder/`.
2. Simply run the following to merge predictions:

```bash
cd csv_folder
python csv_merger.py
```

3. The final `submission.csv` will be generated and ready for upload.

---

## 🔁 Full Pipeline Workflow

To fully reproduce the training and inference pipeline:

1. Place the dataset inside the `dataset/` directory.
2. Navigate to the dataset folder:

```bash
cd dataset
```

3. Run the data augmentation script to handle class imbalance:

```bash
python imbalance_aug.py
```

4. Train the handedness model using classical ML methods:

```bash
cd ../ML_based
# Follow instructions in ML_based/README.md
```

5. Train the remaining models (gender, play years, level) using DL:

```bash
cd ../DL_based
# Follow instructions in DL_based/README.md
```

6. Merge all predicted CSV files into the final output:

```bash
cd ../csv_folder 
python csv_merger.py # need to change the merge file path.
```

---

## 📄 License

This project is licensed under the **MIT License**.  
You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, provided that the original license and copyright notice are included.

For full license details, see the [LICENSE](./LICENSE) file.


---

## 📬 Contact

For questions, issues, or feedback, please raise an issue or contact the repository maintainer.
