"""
models/adaboost.py
------------------
AdaBoost Ensemble classifier implemented FROM SCRATCH.

Base learner : DecisionTree (restricted to max_depth=1 by default for a decision stump).
Aggregation  : Weighted vote for predict(), normalized alpha sum for predict_proba().

No scikit-learn ensemble functions are used.
"""

import numpy as np
from models.decision_tree import DecisionTree


class AdaBoostClassifier:
    """
    AdaBoost ensemble using DecisionTree as the weak learner.

    Parameters
    ----------
    n_estimators : maximum number of boosting rounds.
    max_depth    : max_depth for the base DecisionTree (1 = Decision Stump).
    random_seed  : master seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 1,
        random_seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_seed = random_seed

        self.estimators_: list[DecisionTree] = []
        self.estimator_weights_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostClassifier":
        n_samples = X.shape[0]
        
        # AdaBoost mathematics typically uses labels in {-1, 1} instead of {0, 1}
        y_mapped = np.where(y == 0, -1, 1)

        # 1. Initialize uniform weights
        w = np.ones(n_samples) / n_samples

        rng = np.random.default_rng(self.random_seed)
        self.estimators_ = []
        self.estimator_weights_ = []

        for _ in range(self.n_estimators):
            # 2a. Resample the training set using the current weights as probabilities
            indices = rng.choice(n_samples, size=n_samples, p=w, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # 2b. Train the weak learner (Decision Stump) on the resampled data
            tree = DecisionTree(
                max_depth=self.max_depth, 
                random_seed=int(rng.integers(0, 2**31))
            )
            tree.fit(X_boot, y_boot)

            # 2c. Predict on the ORIGINAL full dataset to compute weighted error
            preds_boot = tree.predict(X)
            preds = np.where(preds_boot == 0, -1, 1)  # Map to {-1, 1}

            incorrect = (preds != y_mapped)
            err = np.sum(w[incorrect]) / np.sum(w)

            # Stop if the learner is perfect (prevents division by zero)
            if err == 0:
                alpha = 1.0
                self.estimators_.append(tree)
                self.estimator_weights_.append(alpha)
                break
            
            # Stop if the learner is worse than random guessing
            if err >= 0.5:
                break

            # 2d. Calculate the amount of say (alpha)
            alpha = 0.5 * np.log((1.0 - err) / err)

            # 2e. Update the weights
            w = w * np.exp(-alpha * y_mapped * preds)
            w = w / np.sum(w)  # Normalize so weights sum to 1

            self.estimators_.append(tree)
            self.estimator_weights_.append(alpha)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted majority vote across all stumps."""
        pred_sum = np.zeros(X.shape[0])
        
        for alpha, tree in zip(self.estimator_weights_, self.estimators_):
            # Get predictions and map from {0, 1} to {-1, 1}
            p = tree.predict(X)
            p_mapped = np.where(p == 0, -1, 1)
            pred_sum += alpha * p_mapped

        # Final prediction is the sign of the weighted sum
        return np.where(pred_sum >= 0, 1, 0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate probabilities based on the normalized sum of alphas.
        Returns: (n_samples, 2) array for [P(class=0), P(class=1)]
        """
        prob_1 = np.zeros(X.shape[0], dtype=np.float64)
        total_alpha = sum(self.estimator_weights_)

        if total_alpha == 0:
            return np.ones((X.shape[0], 2)) * 0.5

        # Sum the alphas ONLY for trees that predicted class 1
        for alpha, tree in zip(self.estimator_weights_, self.estimators_):
            p = tree.predict(X)
            prob_1 += alpha * p

        prob_1 = prob_1 / total_alpha
        prob_0 = 1.0 - prob_1
        return np.vstack((prob_0, prob_1)).T

    def __repr__(self) -> str:
        return (
            f"AdaBoostClassifier("
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth})"
        )