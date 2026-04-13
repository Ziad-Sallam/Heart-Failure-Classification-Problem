"""
utils/tuning.py
---------------
Generic hyperparameter search helpers that work with ANY classifier
that exposes a scikit-learn-compatible fit / predict interface.

Supports
--------
- Grid search   : exhaustive search over a parameter grid.
- Random search : randomly sample from distributions / lists.

Both functions return the best estimator and a results table.
"""

import numpy as np
import itertools
import random
from typing import Any, Callable

from .evaluation import evaluate


# ---------------------------------------------------------------------------
# Grid Search
# ---------------------------------------------------------------------------

def grid_search(
    model_cls,
    param_grid:    dict[str, list],
    X_train:       np.ndarray,
    y_train:       np.ndarray,
    X_val:         np.ndarray,
    y_val:         np.ndarray,
    scoring:       str  = "f1_binary",
    verbose:       bool = True,
    proba_method:  str  = "predict_proba",
) -> tuple[Any, dict, list[dict]]:
    """
    Exhaustive grid search over param_grid.

    Parameters
    ----------
    model_cls     : class (not instance) to instantiate for each combo.
    param_grid    : {param_name: [value1, value2, ...]}.
    X_train, y_train : training data.
    X_val, y_val     : validation data used to score.
    scoring       : key from evaluate() output to maximise.
    verbose       : print progress.
    proba_method  : name of probability method on the model.

    Returns
    -------
    best_model   : fitted model with best hyperparameters.
    best_params  : dict of best hyperparameter values.
    all_results  : list of {params, score} dicts (sorted best-first).
    """
    keys   = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))

    if verbose:
        print(f"[tune] Grid search: {len(combos)} combinations | scoring='{scoring}'")

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        score, _ = _fit_and_score(
            model_cls, params,
            X_train, y_train,
            X_val,   y_val,
            scoring, proba_method,
        )
        results.append({"params": params, "score": score})
        if verbose:
            print(f"  {params}  →  {scoring}={score:.4f}")

    results.sort(key=lambda r: r["score"], reverse=True)
    best = results[0]

    if verbose:
        print(f"\n[tune] Best params : {best['params']}")
        print(f"[tune] Best score  : {best['score']:.4f}")

    best_model = model_cls(**best["params"])
    best_model.fit(X_train, y_train)
    return best_model, best["params"], results


def random_search(
    model_cls,
    param_distributions: dict[str, list | Callable],
    X_train:             np.ndarray,
    y_train:             np.ndarray,
    X_val:               np.ndarray,
    y_val:               np.ndarray,
    n_iter:              int  = 20,
    scoring:             str  = "f1_binary",
    random_seed:         int  = 42,
    verbose:             bool = True,
    proba_method:        str  = "predict_proba",
) -> tuple[Any, dict, list[dict]]:
    """
    Random search over param_distributions.

    param_distributions values can be:
    - a list  → uniform sample from the list.
    - a callable → called with no args to generate a value (e.g. np.random.randint).
    """
    rng = random.Random(random_seed)

    if verbose:
        print(f"[tune] Random search: {n_iter} iterations | scoring='{scoring}'")

    results = []
    for _ in range(n_iter):
        params = {}
        for key, dist in param_distributions.items():
            if callable(dist):
                params[key] = dist()
            else:
                params[key] = rng.choice(dist)

        score, _ = _fit_and_score(
            model_cls, params,
            X_train, y_train,
            X_val,   y_val,
            scoring, proba_method,
        )
        results.append({"params": params, "score": score})
        if verbose:
            print(f"  {params}  →  {scoring}={score:.4f}")

    results.sort(key=lambda r: r["score"], reverse=True)
    best = results[0]

    if verbose:
        print(f"\n[tune] Best params : {best['params']}")
        print(f"[tune] Best score  : {best['score']:.4f}")

    best_model = model_cls(**best["params"])
    best_model.fit(X_train, y_train)
    return best_model, best["params"], results


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _fit_and_score(
    model_cls, params,
    X_train, y_train,
    X_val, y_val,
    scoring: str,
    proba_method: str,
) -> tuple[float, dict]:
    model = model_cls(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    y_proba = None
    if hasattr(model, proba_method):
        raw = getattr(model, proba_method)(X_val)
        if raw.ndim == 2:
            y_proba = raw[:, 1]
        else:
            y_proba = raw

    metrics = evaluate(
        y_val, y_pred, y_proba,
        verbose=False,
    )
    return metrics.get(scoring, 0.0), metrics
