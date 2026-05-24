# 🤖 Cancer Gene Expression Classifier

A machine learning pipeline that classifies **cancer vs normal tissue** using gene expression data.
Uses 4 different ML models, produces ROC curves, confusion matrices, and identifies the most predictive biomarkers.

**Accuracy: 96–98% | AUC-ROC: 0.994**

---

## 🤔 What Does This Do?

Doctors analyze cell samples to determine if tissue is cancerous. This project asks:

> **"Can a machine learning model learn to tell cancer from healthy tissue — and which molecular features matter most?"**

The model learns from **gene expression measurements** (how active each gene is) and classifies samples as:
- 🔴 **Cancer / Malignant**
- 🔵 **Normal / Benign**

It then tells you *which genes/features drove that decision* — which is where the real biology lives.

---

## 📈 What This Produces

### 1. PCA Plot
Visualizes all 569 samples in 2D — you can see cancer and normal samples naturally separate.

### 2. ROC Curve + Confusion Matrix
Standard medical ML evaluation — shows true positive vs false positive rates.

### 3. Feature Importance Plot
Which molecular markers are most predictive of malignancy — the biological insight.

### 4. Model Comparison Chart
Accuracy, AUC, and F1 side-by-side for Random Forest, SVM, Logistic Regression, and Gradient Boosting.

---

## 📊 Results Summary

| Model | Accuracy | AUC-ROC | F1 Score |
|---|---|---|---|
| **Random Forest** | **96.5%** | **0.994** | **0.950** |
| SVM (RBF kernel) | 97.4% | 0.995 | 0.968 |
| Gradient Boosting | 96.5% | 0.995 | 0.950 |
| Logistic Regression | 96.5% | 0.996 | 0.950 |

---

## ✅ What You Need First

1. **Python 3.8 or higher** → [Download from python.org](https://www.python.org/downloads/)
2. **VS Code** → [Download from code.visualstudio.com](https://code.visualstudio.com/)

> 💡 **Check if Python is installed:** Open a terminal and type `python --version`  
> You should see `Python 3.x.x`

---

## 📥 Step-by-Step Installation

### Step 1 — Download this project

**Option A:**  
Click green **Code** button → **Download ZIP** → unzip it

**Option B (using Git):**
```bash
git clone https://github.com/YOUR_USERNAME/cancer-gene-expression-classifier.git
cd cancer-gene-expression-classifier
```

---

### Step 2 — Open the folder in VS Code

1. Open VS Code
2. **File → Open Folder** → select `cancer-gene-expression-classifier`
3. Open the terminal: **View → Terminal** (or press `` Ctrl+` ``)

---

### Step 3 — Install required Python packages

In the VS Code terminal:

```bash
pip install -r requirements.txt
```

> ⚠️ If `pip` doesn't work, try `pip3`  
> This installs: numpy, pandas, scikit-learn, matplotlib, seaborn

---

### Step 4 — Create output folders

```bash
mkdir figures
mkdir results
```

---

## ▶️ Running the Project

### Run 1: Basic run (Random Forest — recommended first run)

```bash
python classifier.py
```

This will:
- Load the breast cancer dataset (automatic — no download needed)
- Train a Random Forest model
- Print accuracy, AUC, F1 score in the terminal
- Save 4 figures to the `figures/` folder

---

### Run 2: Try a different model

```bash
# Support Vector Machine
python classifier.py --model svm

# Logistic Regression
python classifier.py --model lr

# Gradient Boosting
python classifier.py --model gb
```

---

### Run 3: Compare all 4 models at once

```bash
python classifier.py --compare
```

This trains all 4 models and saves a side-by-side comparison chart.

---

### Run 4: All models, all plots

```bash
python classifier.py --model rf --compare
```

---

## 📂 File Structure

```
cancer-gene-expression-classifier/
│
├── classifier.py         ← THE MAIN SCRIPT — run this
├── requirements.txt      ← Python packages list
├── figures/              ← Your plots appear here after running
│   ├── pca_visualization.png
│   ├── evaluation_rf.png
│   ├── feature_importance.png
│   └── model_comparison.png
├── results/              ← CSV data outputs appear here
│   ├── feature_importance.csv
│   └── model_comparison.csv
└── README.md             ← This guide
```

---

## ❓ Common Problems & Fixes

| Error | What to do |
|---|---|
| `python: command not found` | Try `python3 classifier.py` |
| `pip: command not found` | Try `pip3 install -r requirements.txt` |
| `ModuleNotFoundError: numpy` | Run `pip install -r requirements.txt` |
| `No such file or directory: figures/` | Run `mkdir figures` and `mkdir results` first |
| Plots open but don't save | Make sure you're running from inside the project folder |
| Script crashes at model comparison | Make sure all packages installed correctly — re-run `pip install -r requirements.txt` |

---

## 🧬 Biological & ML Background

**Why Random Forest for gene expression?**  
Gene expression data is high-dimensional (thousands of genes) and noisy. Random Forests handle this well — they're robust to outliers, don't overfit easily, and naturally rank features by importance.

**What does AUC-ROC mean in medicine?**  
AUC (Area Under the ROC Curve) measures how well a classifier separates classes across all possible thresholds. AUC = 1.0 is perfect; AUC = 0.5 is random guessing. In clinical diagnostics, AUC > 0.95 is considered excellent.

**What does Feature Importance tell us biologically?**  
In a real RNA-seq dataset (not the proxy used here), the top features would be actual gene names — these become candidate **biomarkers** for cancer diagnosis. That's how ML connects to wet-lab biology.

**Why 5-Fold Cross-Validation?**  
To make sure our accuracy isn't just luck from one train/test split. In 5-fold CV, we train and test 5 times on different data slices and average the result.

---

## 🛠️ Built With

- [scikit-learn](https://scikit-learn.org/) — ML models, evaluation
- [pandas](https://pandas.pydata.org/) — data handling
- [matplotlib](https://matplotlib.org/) + [seaborn](https://seaborn.pydata.org/) — visualization
- [numpy](https://numpy.org/) — numerical operations

---

**Author:** [Your Name] | BSc Bioinformatics, [Your University], Peshawar, Pakistan
