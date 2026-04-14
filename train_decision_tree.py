"""
train_decision_tree.py
----------------------
Train, tune, and evaluate the scratch-built Decision Tree on the
Heart Failure Prediction dataset.

Usage
-----
    python train_decision_tree.py [--csv path/to/heart.csv]

If --csv is omitted, the script attempts to download the dataset
from Kaggle (requires ~/.kaggle/kaggle.json).
"""

import argparse
import sys
import os

# Make sure package root is on the path when running directly
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from data_preparation import prepare_data
from models.decision_tree import DecisionTree
from utils.evaluation import evaluate, compare_models
from utils.tuning import grid_search
from utils.report import save_html_report

RANDOM_SEED = 42


def main(csv_path: str | None = None) -> None:

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  HEART FAILURE PREDICTION — DECISION TREE")
    print("=" * 60)

    splits, scaler, feature_names = prepare_data(csv_path=csv_path)
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    print(f"\nFeatures ({len(feature_names)}): {feature_names}")

    # ------------------------------------------------------------------
    # 2. Hyperparameter tuning on Validation set
    # ------------------------------------------------------------------
    print("\n--- Tuning Decision Tree hyperparameters ---")

    param_grid = {
        "max_depth":         [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 5],
    }

    best_model, best_params, all_results = grid_search(
        model_cls    = DecisionTree,
        param_grid   = param_grid,
        X_train      = X_train,
        y_train      = y_train,
        X_val        = X_val,
        y_val        = y_val,
        scoring      = "f1_binary",
        verbose      = True,
    )

    # ------------------------------------------------------------------
    # 3. Evaluate on all splits with the best model
    # ------------------------------------------------------------------
    print("\n--- Final evaluation with best hyperparameters ---")

    # Retrain on train+val combined for final evaluation
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    final_model = DecisionTree(**best_params, random_seed=RANDOM_SEED)
    final_model.fit(X_trainval, y_trainval)

    print(f"\nTree depth  : {final_model.get_depth()}")
    print(f"Leaf count  : {final_model.count_leaves()}")

    split_results = {}

    for split_name, X, y in [
        ("Train",      X_train,    y_train),
        ("Validation", X_val,      y_val),
        ("Test",       X_test,     y_test),
    ]:
        y_pred  = final_model.predict(X)
        y_proba = final_model.predict_proba(X)[:, 1]
        metrics = evaluate(
            y_true=y, y_pred=y_pred, y_proba=y_proba,
            model_name="Decision Tree (scratch)",
            split_name=split_name,
        )
        split_results[split_name] = metrics

    # ------------------------------------------------------------------
    # 4. Save results summary (text)
    # ------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    summary_path = os.path.join("results", "decision_tree_results.txt")
    with open(summary_path, "w") as f:
        f.write("DECISION TREE — RESULTS SUMMARY\n")
        f.write("=" * 52 + "\n")
        f.write(f"Best hyperparameters: {best_params}\n")
        f.write(f"Tree depth : {final_model.get_depth()}\n")
        f.write(f"Leaf count : {final_model.count_leaves()}\n\n")
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

        f.write("\nALL TUNING ITERATIONS (sorted by iteration)\n")
        f.write("-" * 52 + "\n")
        for r in sorted(all_results, key=lambda x: x["iteration"]):
            f.write(
                f"  #{r['iteration']:>3}  {r['params']}  "
                f"→  f1_binary={r['score']:.4f}\n"
            )

    print(f"\n[results] Text summary saved to '{summary_path}'")

    # ------------------------------------------------------------------
    # 5. Save HTML report
    # ------------------------------------------------------------------
    html_path = os.path.join("results", "decision_tree_report.html")
    save_html_report(
        best_params   = best_params,
        all_results   = all_results,
        split_results = split_results,
        tree_depth    = final_model.get_depth(),
        leaf_count    = final_model.count_leaves(),
        scoring       = "f1_binary",
        output_path   = html_path,
    )
    print(f"[results] HTML report  saved to '{html_path}'")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Decision Tree on Heart Failure dataset"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to heart.csv (downloads from Kaggle if omitted)",
    )
    args = parser.parse_args()
    main(csv_path=args.csv)