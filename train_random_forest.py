"""
train_random_forest.py
----------------------
Train, tune, and evaluate the scratch-built Random Forest on the
Heart Failure Prediction dataset.

Usage
-----
    python train_random_forest.py [--csv path/to/heart.csv]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

from data_preparation import prepare_data
from models.random_forest import RandomForest
from utils.evaluation import evaluate, compare_models
from utils.tuning import grid_search

RANDOM_SEED = 42


def plot_feature_importances(
    importances:   np.ndarray,
    feature_names: list[str],
    save_path:     str,
    top_n:         int = 15,
) -> None:
    """Bar chart of the top-N most important features (MDI)."""
    # Pick top_n features
    indices = np.argsort(importances)[::-1][:top_n]
    top_names  = [feature_names[i] for i in indices]
    top_values = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(range(top_n), top_values[::-1], color="steelblue", edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean Decrease in Impurity (normalised)", fontsize=10)
    ax.set_title("Random Forest — Top Feature Importances", fontsize=12, fontweight="bold")
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in top_values[::-1]],
                 padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] Feature importance chart saved to '{save_path}'")


def plot_confusion_matrix(
    cm:         np.ndarray,
    title:      str,
    save_path:  str,
) -> None:
    """Simple 2×2 confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i,j]}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] Confusion matrix saved to '{save_path}'")


# ---------------------------------------------------------------------------

def main(csv_path: str | None = None) -> None:

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  HEART FAILURE PREDICTION — RANDOM FOREST")
    print("=" * 60)

    splits, scaler, feature_names = prepare_data(csv_path=csv_path)
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    print(f"\nFeatures ({len(feature_names)}): {feature_names}")

    # ------------------------------------------------------------------
    # 2. Hyperparameter tuning on the Validation set
    # ------------------------------------------------------------------
    print("\n--- Tuning Random Forest hyperparameters ---")

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_features": ["sqrt", "log2"],   # the key RF knob
        "max_depth":    [5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf":  [1, 2],
    }

    best_model, best_params, all_results = grid_search(
        model_cls  = RandomForest,
        param_grid = param_grid,
        X_train    = X_train,
        y_train    = y_train,
        X_val      = X_val,
        y_val      = y_val,
        scoring    = "f1_binary",
        verbose    = True,
    )

    # ------------------------------------------------------------------
    # 3. Retrain on train + val, then evaluate on all splits
    # ------------------------------------------------------------------
    print("\n--- Final evaluation with best hyperparameters ---")

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    final_model = RandomForest(**best_params, random_seed=RANDOM_SEED)
    final_model.fit(X_trainval, y_trainval)

    print(f"\nEnsemble size : {len(final_model.estimators_)} trees")
    print(f"Max features  : {best_params.get('max_features')} per split")

    split_results = {}
    for split_name, X, y in [
        ("Train",      X_train, y_train),
        ("Validation", X_val,   y_val),
        ("Test",       X_test,  y_test),
    ]:
        y_pred  = final_model.predict(X)
        y_proba = final_model.predict_proba(X)[:, 1]
        metrics = evaluate(
            y_true=y, y_pred=y_pred, y_proba=y_proba,
            model_name="Random Forest (scratch)",
            split_name=split_name,
        )
        split_results[split_name] = metrics

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)

    # Feature importance
    importances = final_model.feature_importances()
    plot_feature_importances(
        importances, feature_names,
        save_path=os.path.join("results", "rf_feature_importances.png"),
    )

    # Confusion matrix on test set
    plot_confusion_matrix(
        split_results["Test"]["confusion_matrix"],
        title="Random Forest — Test Confusion Matrix",
        save_path=os.path.join("results", "rf_confusion_matrix.png"),
    )

    # ------------------------------------------------------------------
    # 5. Comparison: how much does RF improve over a single Decision Tree?
    # ------------------------------------------------------------------
    print("\n--- Bagging vs Random Forest: what changes? ---")
    print("  Bagging : bootstrap rows  +  ALL features at every split")
    print("  RF      : bootstrap rows  +  RANDOM SUBSET of features at every split")
    print(f"  RF uses : {best_params.get('max_features')!r} features per split  "
          f"(= ~{_n_feats_used(best_params.get('max_features'), len(feature_names))} "
          f"out of {len(feature_names)} total)")

    # ------------------------------------------------------------------
    # 6. Save results summary
    # ------------------------------------------------------------------
    summary_path = os.path.join("results", "random_forest_results.txt")
    with open(summary_path, "w") as f:
        f.write("RANDOM FOREST — RESULTS SUMMARY\n")
        f.write("=" * 52 + "\n")
        f.write(f"Best hyperparameters : {best_params}\n")
        f.write(f"Ensemble size        : {len(final_model.estimators_)} trees\n\n")
        for split_name, m in split_results.items():
            f.write(f"{split_name}\n")
            f.write(f"  Accuracy   : {m['accuracy']:.4f}\n")
            f.write(f"  F1 (macro) : {m['f1_macro']:.4f}\n")
            f.write(f"  F1 (binary): {m['f1_binary']:.4f}\n")
            if "roc_auc" in m:
                f.write(f"  ROC-AUC    : {m['roc_auc']:.4f}\n")
            cm = m["confusion_matrix"]
            f.write(f"  Confusion Matrix: TN={cm[0,0]} FP={cm[0,1]} "
                    f"FN={cm[1,0]} TP={cm[1,1]}\n\n")

    print(f"\n[results] Summary saved to '{summary_path}'")


def _n_feats_used(strategy, n_total: int) -> int:
    if strategy == "sqrt":  return max(1, int(np.sqrt(n_total)))
    if strategy == "log2":  return max(1, int(np.log2(n_total)))
    if isinstance(strategy, int): return strategy
    return n_total


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Random Forest on Heart Failure dataset"
    )
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()
    main(csv_path=args.csv)
