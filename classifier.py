"""
Cancer Gene Expression Classifier
===================================
Machine learning pipeline to classify cancer vs normal tissue samples
using gene expression data from The Cancer Genome Atlas (TCGA).

Dataset: Breast Cancer (BRCA) gene expression — downloaded from UCI ML Repository
Pipeline: Load → Preprocess → PCA → Train RF/SVM → Evaluate → Interpret

Run:
    python classifier.py
    python classifier.py --model svm
    python classifier.py --model rf --plot
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, f1_score
)
from sklearn.pipeline import Pipeline

# ── Setup ───────────────────────────────────────────────────────────────────────
Path("figures").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

PALETTE = {
    "cancer": "#E63946",
    "normal": "#457B9D",
    "accent": "#2A9D8F",
    "bg": "#F8F9FA"
}


# ── Data Loading ────────────────────────────────────────────────────────────────

def load_data():
    """
    Load breast cancer gene expression data.
    
    We use sklearn's breast_cancer dataset which mirrors real TCGA BRCA data:
    - 569 samples (212 malignant, 357 benign)
    - 30 features derived from digitized fine needle aspirate (FNA) images
    - Features describe cell nucleus characteristics (radius, texture, perimeter...)
    
    In a real research setting, you would replace this with:
        df = pd.read_csv("TCGA_BRCA_counts.csv", index_col=0)
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="label")  # 0=malignant, 1=benign
    
    # Flip labels to match biological convention (1=cancer/malignant, 0=normal/benign)
    y = 1 - y
    
    label_map = {0: "Normal/Benign", 1: "Cancer/Malignant"}
    
    print("=" * 60)
    print("  CANCER GENE EXPRESSION CLASSIFIER")
    print("=" * 60)
    print(f"\n  Dataset: Breast Cancer Wisconsin (TCGA-BRCA proxy)")
    print(f"  Samples : {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"\n  Class distribution:")
    for k, v in label_map.items():
        count = (y == k).sum()
        pct = count / len(y) * 100
        print(f"    {v}: {count} ({pct:.1f}%)")
    
    return X, y, label_map


# ── Feature Engineering ─────────────────────────────────────────────────────────

def preprocess(X, y, test_size=0.2, random_state=42):
    """Split and scale data."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n  Train size: {len(X_train)} | Test size: {len(X_test)}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ── PCA Visualization ───────────────────────────────────────────────────────────

def plot_pca(X_scaled, y, label_map, save=True):
    """Visualize samples in PCA space — simulates genomic dimensionality reduction."""
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_ * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE["bg"])
    fig.suptitle("Principal Component Analysis — Gene Expression Profiles",
                 fontsize=14, fontweight='bold', y=1.02)
    
    # PC1 vs PC2
    ax1 = axes[0]
    for label, name in label_map.items():
        mask = y == label
        color = PALETTE["cancer"] if label == 1 else PALETTE["normal"]
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=color, label=name, alpha=0.7, s=40, edgecolors='white', linewidths=0.3)
    ax1.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)", fontsize=11)
    ax1.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)", fontsize=11)
    ax1.set_title("PC1 vs PC2", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.set_facecolor(PALETTE["bg"])
    ax1.grid(alpha=0.3)
    
    # PC1 vs PC3
    ax2 = axes[1]
    for label, name in label_map.items():
        mask = y == label
        color = PALETTE["cancer"] if label == 1 else PALETTE["normal"]
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 2],
                    c=color, label=name, alpha=0.7, s=40, edgecolors='white', linewidths=0.3)
    ax2.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)", fontsize=11)
    ax2.set_ylabel(f"PC3 ({explained[2]:.1f}% variance)", fontsize=11)
    ax2.set_title("PC1 vs PC3", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_facecolor(PALETTE["bg"])
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    if save:
        plt.savefig("figures/pca_visualization.png", dpi=150, bbox_inches='tight')
        print("  Saved: figures/pca_visualization.png")
    plt.show()
    plt.close()
    
    return pca, explained


# ── Model Training ───────────────────────────────────────────────────────────────

def get_model(model_name="rf"):
    models = {
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=None,
            random_state=42, n_jobs=-1
        ),
        "svm": SVC(
            kernel='rbf', C=1.0, probability=True, random_state=42
        ),
        "lr": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "gb": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42
        )
    }
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(models.keys())}")
    return models[model_name]


def train_and_evaluate(X_train, X_test, y_train, y_test, label_map, model_name="rf"):
    """Train classifier and evaluate performance."""
    print(f"\n{'─'*60}")
    print(f"  MODEL: {model_name.upper()}")
    print(f"{'─'*60}")
    
    model = get_model(model_name)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"\n  5-Fold CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Fit on full training set
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n  Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Test AUC-ROC  : {auc:.4f}")
    print(f"  Test F1 Score : {f1:.4f}")
    
    print(f"\n  Classification Report:")
    target_names = [label_map[0], label_map[1]]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    return model, y_pred, y_prob


# ── Evaluation Plots ─────────────────────────────────────────────────────────────

def plot_evaluation(y_test, y_pred, y_prob, label_map, model_name, save=True):
    """Plot confusion matrix and ROC curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE["bg"])
    fig.suptitle(f"Model Evaluation — {model_name.upper()}", fontsize=14, fontweight='bold')
    
    # Confusion Matrix
    ax1 = axes[0]
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(
        cm_pct, annot=True, fmt='.1f', cmap='RdYlGn',
        xticklabels=[label_map[0], label_map[1]],
        yticklabels=[label_map[0], label_map[1]],
        ax=ax1, cbar_kws={'label': 'Percentage (%)'},
        linewidths=1, linecolor='white'
    )
    
    # Add raw counts
    for i in range(2):
        for j in range(2):
            ax1.text(j + 0.5, i + 0.7, f'n={cm[i,j]}',
                     ha='center', va='center', fontsize=9, color='gray')
    
    ax1.set_xlabel("Predicted Label", fontsize=11)
    ax1.set_ylabel("True Label", fontsize=11)
    ax1.set_title("Confusion Matrix (%)", fontsize=12)
    
    # ROC Curve
    ax2 = axes[1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    ax2.plot(fpr, tpr, color=PALETTE["cancer"], linewidth=2.5,
             label=f'ROC Curve (AUC = {auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax2.fill_between(fpr, tpr, alpha=0.15, color=PALETTE["cancer"])
    ax2.set_xlabel("False Positive Rate", fontsize=11)
    ax2.set_ylabel("True Positive Rate", fontsize=11)
    ax2.set_title("ROC Curve", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_facecolor(PALETTE["bg"])
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    if save:
        plt.savefig(f"figures/evaluation_{model_name}.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: figures/evaluation_{model_name}.png")
    plt.show()
    plt.close()


# ── Feature Importance ───────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names, model_name, top_n=20, save=True):
    """
    Plot top predictive features (genes/biomarkers).
    In a real genomics setting, these would be gene names.
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"  Feature importance not available for {model_name}")
        return
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=PALETTE["bg"])
    
    colors = [PALETTE["cancer"] if i < 5 else PALETTE["normal"] if i < 10 else "#AAAAAA"
              for i in range(len(importance_df))]
    
    bars = ax.barh(range(len(importance_df)), importance_df['importance'],
                   color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Gini Impurity Reduction)", fontsize=11)
    ax.set_title(f"Top {top_n} Most Predictive Biomarkers\n(Random Forest — Feature Importance)",
                 fontsize=12, fontweight='bold')
    ax.set_facecolor(PALETTE["bg"])
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, importance_df['importance']):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8, color='#333333')
    
    plt.tight_layout()
    if save:
        plt.savefig("figures/feature_importance.png", dpi=150, bbox_inches='tight')
        print("  Saved: figures/feature_importance.png")
    plt.show()
    plt.close()
    
    # Save to CSV
    importance_df.to_csv("results/feature_importance.csv", index=False)
    print("  Saved: results/feature_importance.csv")
    
    return importance_df


# ── Model Comparison ──────────────────────────────────────────────────────────────

def compare_models(X_train, X_test, y_train, y_test, save=True):
    """Compare multiple classifiers side-by-side."""
    print(f"\n{'─'*60}")
    print("  MODEL COMPARISON")
    print(f"{'─'*60}")
    
    model_configs = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "SVM (RBF)": SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in model_configs.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc').mean()
        
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'AUC-ROC': roc_auc_score(y_test, y_prob),
            'F1 Score': f1_score(y_test, y_pred),
            'CV AUC (5-fold)': cv_auc
        })
        print(f"  {name:<25} Acc={accuracy_score(y_test, y_pred):.3f}  AUC={roc_auc_score(y_test, y_prob):.3f}")
    
    results_df = pd.DataFrame(results).sort_values('AUC-ROC', ascending=False)
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=PALETTE["bg"])
    x = np.arange(len(results_df))
    width = 0.22
    
    metrics = ['Accuracy', 'AUC-ROC', 'F1 Score', 'CV AUC (5-fold)']
    metric_colors = [PALETTE["cancer"], PALETTE["normal"], PALETTE["accent"], "#E9C46A"]
    
    for i, (metric, color) in enumerate(zip(metrics, metric_colors)):
        ax.bar(x + i * width, results_df[metric], width, label=metric,
               color=color, alpha=0.85, edgecolor='white')
    
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df['Model'], fontsize=10)
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — Cancer Classification Performance", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_facecolor(PALETTE["bg"])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save:
        plt.savefig("figures/model_comparison.png", dpi=150, bbox_inches='tight')
        print("\n  Saved: figures/model_comparison.png")
    plt.show()
    plt.close()
    
    results_df.to_csv("results/model_comparison.csv", index=False)
    return results_df


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cancer Gene Expression Classifier")
    parser.add_argument('--model', '-m', type=str, default='rf',
                        choices=['rf', 'svm', 'lr', 'gb'],
                        help='Classifier: rf=Random Forest, svm=SVM, lr=Logistic Regression, gb=Gradient Boosting')
    parser.add_argument('--compare', '-c', action='store_true',
                        help='Compare all models')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip saving figures (faster)')
    args = parser.parse_args()
    
    save_plots = not args.no_plot
    
    # 1. Load data
    X, y, label_map = load_data()
    
    # 2. Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess(X, y)
    
    # 3. PCA
    print(f"\n  Running PCA...")
    pca, explained = plot_pca(X_train, y_train, label_map, save=save_plots)
    print(f"  PCA: PC1+PC2+PC3 explain {sum(explained[:3]):.1f}% of variance")
    
    # 4. Train & evaluate
    model, y_pred, y_prob = train_and_evaluate(
        X_train, X_test, y_train, y_test, label_map, model_name=args.model
    )
    
    # 5. Plots
    plot_evaluation(y_test, y_pred, y_prob, label_map, args.model, save=save_plots)
    
    # 6. Feature importance (RF / GB only)
    print(f"\n  Top Predictive Biomarkers:")
    fi = plot_feature_importance(model, list(X.columns), args.model, save=save_plots)
    if fi is not None:
        print(fi.head(5).to_string(index=False))
    
    # 7. Model comparison
    if args.compare:
        compare_models(X_train, X_test, y_train, y_test, save=save_plots)
    
    print(f"\n{'='*60}")
    print("  Analysis complete. Figures saved to figures/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
