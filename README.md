MY AI built projects


#  Cancer Gene Expression Classifier

A machine learning pipeline that classifies **cancer vs normal tissue** using gene expression data.
Uses 4 different ML models, produces ROC curves, confusion matrices, and identifies the most predictive biomarkers.

**Accuracy: 96–98% | AUC-ROC: 0.994**

---

# What Does This Do?

Doctors analyze cell samples to determine if tissue is cancerous. This project asks:

> **"Can a machine learning model learn to tell cancer from healthy tissue — and which molecular features matter most?"**

The model learns from **gene expression measurements** (how active each gene is) and classifies samples as:
- 🔴 **Cancer / Malignant**
- 🔵 **Normal / Benign**

It then tells you *which genes/features drove that decision* — which is where the real biology lives.

---

# What This Produces

### 1. PCA Plot
Visualizes all 569 samples in 2D — you can see cancer and normal samples naturally separate.

### 2. ROC Curve + Confusion Matrix
Standard medical ML evaluation — shows true positive vs false positive rates.

### 3. Feature Importance Plot
Which molecular markers are most predictive of malignancy — the biological insight.

### 4. Model Comparison Chart
Accuracy, AUC, and F1 side-by-side for Random Forest, SVM, Logistic Regression, and Gradient Boosting.

---

# Results Summary

| Model | Accuracy | AUC-ROC | F1 Score |
|---|---|---|---|
| **Random Forest** | **96.5%** | **0.994** | **0.950** |
| SVM (RBF kernel) | 97.4% | 0.995 | 0.968 |
| Gradient Boosting | 96.5% | 0.995 | 0.950 |
| Logistic Regression | 96.5% | 0.996 | 0.950 |



# Biological & ML Background

**Why Random Forest for gene expression?**  
Gene expression data is high-dimensional (thousands of genes) and noisy. Random Forests handle this well — they're robust to outliers, don't overfit easily, and naturally rank features by importance.

**What does AUC-ROC mean in medicine?**  
AUC (Area Under the ROC Curve) measures how well a classifier separates classes across all possible thresholds. AUC = 1.0 is perfect; AUC = 0.5 is random guessing. In clinical diagnostics, AUC > 0.95 is considered excellent.

**What does Feature Importance tell us biologically?**  
In a real RNA-seq dataset (not the proxy used here), the top features would be actual gene names — these become candidate **biomarkers** for cancer diagnosis. That's how ML connects to wet-lab biology.

**Why 5-Fold Cross-Validation?**  
To make sure our accuracy isn't just luck from one train/test split. In 5-fold CV, we train and test 5 times on different data slices and average the result.

---

# Built With

- [scikit-learn](https://scikit-learn.org/) — ML models, evaluation
- [pandas](https://pandas.pydata.org/) — data handling
- [matplotlib](https://matplotlib.org/) + [seaborn](https://seaborn.pydata.org/) — visualization
- [numpy](https://numpy.org/) — numerical operations

---

**Author:** Waleed Tariq | BSc Bioinformatics, University of Agriculture, Peshawar, Pakistan
