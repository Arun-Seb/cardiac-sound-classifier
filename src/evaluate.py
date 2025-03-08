"""
evaluate.py
───────────
Evaluation utilities: metrics, confusion matrices, ROC curves,
benchmark heatmaps, threshold analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc,
)


# ── Multiclass evaluation ────────────────────────────────────
def evaluate_multiclass(name: str, model, X_test, y_test,
                        class_names, output_dir: Path = None) -> float:
    """Train metrics + confusion matrix for multiclass task."""
    y_pred = model.predict(X_test)
    acc    = (y_pred == y_test).mean()

    print(f"\n{'='*55}")
    print(f"  {name}  —  accuracy: {acc:.1%}")
    print('='*55)
    print(classification_report(y_test, y_pred, target_names=class_names))

    if output_dir:
        cm  = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f"Confusion Matrix\n{name}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.tight_layout()
        fname = name.replace(" ","_").replace("(","").replace(")","").lower()
        out   = Path(output_dir) / f"cm_{fname}.png"
        plt.savefig(out, dpi=150)
        plt.close()
    return acc


# ── Binary evaluation ────────────────────────────────────────
def evaluate_binary(name: str, model, X_test, y_test,
                    output_dir: Path = None) -> dict:
    """Compute sensitivity, specificity, AUC for binary task."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc         = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1          = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score   = auc(fpr, tpr)

    return {
        "name"       : name,
        "Accuracy"   : acc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision"  : precision,
        "F1"         : f1,
        "AUC-ROC"    : auc_score,
        "fpr"        : fpr,
        "tpr"        : tpr,
    }


# ── Benchmark heatmap ────────────────────────────────────────
def plot_benchmark_heatmap(results_df: pd.DataFrame,
                           metric: str = "Accuracy",
                           title: str = "Benchmark",
                           output_path: Path = None,
                           vmin: float = 50,
                           vmax: float = 100,
                           cmap: str = "YlGnBu"):
    """Plot accuracy/AUC heatmap — feature sets × models."""
    pivot = results_df.pivot(index="Feature Set", columns="Model", values=metric)
    data  = pivot * 100 if metric != "AUC-ROC" else pivot

    fig, ax = plt.subplots(figsize=(13, 4))
    annot   = data.round(1).astype(str)
    sns.heatmap(data, annot=annot, fmt="", cmap=cmap,
                vmin=vmin, vmax=vmax, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Model", fontsize=10)
    ax.set_ylabel("Feature Set", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── ROC curves ───────────────────────────────────────────────
def plot_roc_curves(roc_data: dict, title: str = "ROC Curves",
                   output_path: Path = None):
    """
    Plot ROC curves.
    roc_data: {label: (fpr, tpr, auc_score)}
    """
    colors = ["#4C72B0","#DD8452","#55A868","#C44E52",
              "#8172B2","#937860","#DA8BC3","#8C8C8C"]
    fig, ax = plt.subplots(figsize=(8, 7))

    for (label, (fpr, tpr, auc_score)), color in zip(roc_data.items(), colors):
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{label} (AUC={auc_score:.3f})")

    ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)",      fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()


# ── Threshold analysis ───────────────────────────────────────
def find_optimal_threshold(y_test, y_proba) -> tuple[float, dict]:
    """
    Find threshold that maximises balanced sensitivity + specificity.
    Returns (optimal_threshold, metrics_at_threshold).
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    specificity = 1 - fpr
    balanced    = (tpr + specificity) / 2
    best_idx    = np.argmax(balanced)
    best_thresh = thresholds[best_idx]

    y_pred_opt  = (y_proba >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_opt).ravel()

    return best_thresh, {
        "threshold"  : best_thresh,
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
    }


def plot_threshold_tradeoff(y_test, y_proba,
                            optimal_thresh: float = None,
                            output_path: Path = None):
    """Plot sensitivity vs specificity across decision thresholds."""
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    specificity = 1 - fpr

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, tpr,         color="#C44E52", lw=2, label="Sensitivity")
    ax.plot(thresholds, specificity, color="#4C72B0", lw=2, label="Specificity")
    ax.plot(thresholds, (tpr + specificity) / 2, color="#55A868",
            lw=1.5, linestyle="--", label="Balanced Average")

    if optimal_thresh:
        ax.axvline(optimal_thresh, color="gray", linestyle=":", lw=1.5,
                   label=f"Optimal threshold = {optimal_thresh:.2f}")

    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Sensitivity vs Specificity Trade-off\n"
                 "(Lower threshold → catch more abnormal, more false alarms)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()
