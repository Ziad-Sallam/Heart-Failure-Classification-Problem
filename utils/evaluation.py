"""
utils/evaluation.py
--------------------
Shared evaluation helpers used by every model in the assignment.

Metrics reported
----------------
- Accuracy
- Precision, Recall, F1  (macro & per-class)
- ROC-AUC
- Confusion matrix
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate(
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    y_proba:      np.ndarray | None = None,
    model_name:   str               = "Model",
    split_name:   str               = "Test",
    verbose:      bool              = True,
) -> dict:
    """
    Compute and (optionally) print a standard set of classification metrics.

    Parameters
    ----------
    y_true     : ground-truth labels.
    y_pred     : predicted class labels.
    y_proba    : predicted probabilities for the positive class (for AUC).
                 If None, AUC is not computed.
    model_name : label shown in printed output.
    split_name : 'Train' / 'Val' / 'Test'.
    verbose    : whether to print a formatted table.

    Returns
    -------
    dict of metric_name → value.
    """
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred,    average="macro", zero_division=0),
        "f1_macro":  f1_score(y_true, y_pred,        average="macro", zero_division=0),
        "f1_binary": f1_score(y_true, y_pred,        average="binary", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = float("nan")

    if verbose:
        _print_metrics(metrics, model_name, split_name)

    return metrics


def _print_metrics(metrics: dict, model_name: str, split_name: str) -> None:
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  {model_name}  |  {split_name} set")
    print(sep)
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}  (macro)")
    print(f"  Recall     : {metrics['recall']:.4f}  (macro)")
    print(f"  F1 (macro) : {metrics['f1_macro']:.4f}")
    print(f"  F1 (binary): {metrics['f1_binary']:.4f}")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    print(sep)


def compare_models(results: dict[str, dict], split: str = "test") -> None:
    """
    Print a side-by-side comparison table for all models on the same split.

    Parameters
    ----------
    results : {model_name: metrics_dict} as returned by evaluate().
    split   : label for the header.
    """
    header = f"\n{'='*70}\n  MODEL COMPARISON — {split.upper()} SET\n{'='*70}"
    print(header)
    row_fmt = "{:<28s} {:>8s} {:>8s} {:>8s} {:>8s}"
    print(row_fmt.format("Model", "Acc", "F1_mac", "F1_bin", "AUC"))
    print("-" * 70)
    for name, m in results.items():
        auc = f"{m['roc_auc']:.4f}" if "roc_auc" in m else "  N/A "
        print(row_fmt.format(
            name[:28],
            f"{m['accuracy']:.4f}",
            f"{m['f1_macro']:.4f}",
            f"{m['f1_binary']:.4f}",
            auc,
        ))
    print("=" * 70)
