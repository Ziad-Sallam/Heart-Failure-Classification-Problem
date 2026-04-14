"""
train_adaboost.py
-----------------
Train, tune, and evaluate the scratch-built AdaBoost ensemble on the
Heart Failure Prediction dataset.

Usage
-----
    python train_adaboost.py [--csv path/to/heart.csv]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from data_preparation import prepare_data
from models.adaboost import AdaBoostClassifier
from utils.evaluation import evaluate
from utils.tuning import grid_search

RANDOM_SEED = 42


def main(csv_path: str | None = None) -> None:

    # 1. Data Prep
    print("\n" + "=" * 60)
    print("  HEART FAILURE PREDICTION — ADABOOST")
    print("=" * 60)

    splits, scaler, feature_names = prepare_data(csv_path=csv_path)
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    # 2. Hyperparameter tuning on the Validation set
    print("\n--- Tuning AdaBoost hyperparameters ---")

    param_grid = {
        "n_estimators": [10, 50, 100],
        "max_depth":    [1, 2],  # Depth 1 is a classic decision stump
    }

    best_model, best_params, all_results = grid_search(
        model_cls  = AdaBoostClassifier,
        param_grid = param_grid,
        X_train    = X_train,
        y_train    = y_train,
        X_val      = X_val,
        y_val      = y_val,
        scoring    = "f1_binary",
        verbose    = True,
    )

    # 3. Retrain on train + val, then evaluate on all splits
    print("\n--- Final evaluation with best hyperparameters ---")

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    final_model = AdaBoostClassifier(**best_params, random_seed=RANDOM_SEED)
    final_model.fit(X_trainval, y_trainval)

    print(f"\nEnsemble size : {len(final_model.estimators_)} weak learners")

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
            model_name="AdaBoost (scratch)",
            split_name=split_name,
        )
        split_results[split_name] = metrics

    # 4. Save results summary
    os.makedirs("results", exist_ok=True)
    summary_path = os.path.join("results", "adaboost_results.txt")
    with open(summary_path, "w") as f:
        f.write("ADABOOST ENSEMBLE — RESULTS SUMMARY\n")
        f.write("=" * 52 + "\n")
        f.write(f"Best hyperparameters : {best_params}\n")
        f.write(f"Ensemble size        : {len(final_model.estimators_)} weak learners\n\n")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train AdaBoost Ensemble on Heart Failure dataset"
    )
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()
    main(csv_path=args.csv)