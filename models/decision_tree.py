"""
models/decision_tree.py
-----------------------
Decision Tree classifier implemented FROM SCRATCH.

Splitting criterion : Information Gain (based on Shannon entropy).
Supports           : max_depth, min_samples_split, min_samples_leaf
                     to prevent overfitting.

No scikit-learn tree functions are used.
"""

import numpy as np
from collections import Counter


# ---------------------------------------------------------------------------
# Utility: entropy & information gain
# ---------------------------------------------------------------------------

def _entropy(y: np.ndarray) -> float:
    """Shannon entropy of a label array."""
    n = len(y)
    if n == 0:
        return 0.0
    counts = np.bincount(y)
    probs  = counts[counts > 0] / n
    return float(-np.sum(probs * np.log2(probs)))


def _information_gain(y_parent: np.ndarray,
                      y_left:   np.ndarray,
                      y_right:  np.ndarray) -> float:
    """IG = H(parent) - weighted_avg H(children)."""
    n  = len(y_parent)
    nl = len(y_left)
    nr = len(y_right)
    if nl == 0 or nr == 0:
        return 0.0
    weighted_child_entropy = (nl / n) * _entropy(y_left) + \
                             (nr / n) * _entropy(y_right)
    return _entropy(y_parent) - weighted_child_entropy


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class _Node:
    """A single node in the decision tree."""

    __slots__ = (
        "feature_idx", "threshold",
        "left", "right",
        "value",          # set for leaf nodes (majority class)
        "n_samples",
        "impurity",
    )

    def __init__(self):
        self.feature_idx: int | None  = None
        self.threshold:   float | None = None
        self.left:  "_Node | None"    = None
        self.right: "_Node | None"    = None
        self.value: int | None        = None
        self.n_samples: int           = 0
        self.impurity: float          = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


# ---------------------------------------------------------------------------
# DecisionTree
# ---------------------------------------------------------------------------

class DecisionTree:
    """
    Binary Decision Tree for classification, built with Information Gain.

    Parameters
    ----------
    max_depth          : maximum tree depth  (None = unlimited).
    min_samples_split  : minimum samples in a node to attempt a split.
    min_samples_leaf   : minimum samples required in each child after split.
    n_features         : number of features to consider per split
                         (None = all; 'sqrt' / 'log2' for random subsets –
                          useful for Random Forest building blocks).
    random_seed        : seed used when n_features is a subset strategy.
    """

    def __init__(
        self,
        max_depth:         int | None = None,
        min_samples_split: int        = 2,
        min_samples_leaf:  int        = 1,
        n_features:        int | str | None = None,
        random_seed:       int        = 42,
    ):
        self.max_depth          = max_depth
        self.min_samples_split  = min_samples_split
        self.min_samples_leaf   = min_samples_leaf
        self.n_features         = n_features
        self.random_seed        = random_seed
        self._rng               = np.random.default_rng(random_seed)
        self.root: _Node | None = None
        self.n_classes_: int    = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        """Fit the tree on training data."""
        self.n_classes_ = len(np.unique(y))
        self.n_features_in_ = X.shape[1]
        self.root = self._build(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        return np.array([self._predict_row(row, self.root) for row in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return per-class probability estimates (leaf majority vote)."""
        return np.array([self._predict_proba_row(row, self.root) for row in X])

    # ------------------------------------------------------------------
    # Tree building (recursive)
    # ------------------------------------------------------------------

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        node = _Node()
        node.n_samples = len(y)
        node.impurity  = _entropy(y)

        # ---- Stop criteria → leaf ----
        if self._should_stop(X, y, depth):
            node.value = self._majority_class(y)
            return node

        # ---- Find best split ----
        best = self._best_split(X, y)

        if best is None:                          # no informative split found
            node.value = self._majority_class(y)
            return node

        feat, thresh, left_mask, right_mask = best

        node.feature_idx = feat
        node.threshold   = thresh
        node.left  = self._build(X[left_mask],  y[left_mask],  depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        return node

    def _should_stop(self, X: np.ndarray, y: np.ndarray, depth: int) -> bool:
        if len(np.unique(y)) == 1:               # pure node
            return True
        if len(y) < self.min_samples_split:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        return False

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[int, float, np.ndarray, np.ndarray] | None:
        """
        Search all features × candidate thresholds for the highest IG split.

        Returns (feature_idx, threshold, left_mask, right_mask) or None.
        """
        best_gain  = -np.inf
        best_split = None

        feature_indices = self._feature_subset(X.shape[1])

        for feat in feature_indices:
            col = X[:, feat]
            # Candidate thresholds: midpoints between consecutive sorted unique values
            unique_vals = np.unique(col)
            if len(unique_vals) < 2:
                continue
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thresh in thresholds:
                left_mask  = col <= thresh
                right_mask = ~left_mask

                if (left_mask.sum()  < self.min_samples_leaf or
                        right_mask.sum() < self.min_samples_leaf):
                    continue

                gain = _information_gain(y, y[left_mask], y[right_mask])

                if gain > best_gain:
                    best_gain  = gain
                    best_split = (feat, thresh, left_mask, right_mask)

        if best_gain <= 0:      # no improvement
            return None
        return best_split

    def _feature_subset(self, n_total: int) -> np.ndarray:
        if self.n_features is None:
            return np.arange(n_total)
        if self.n_features == "sqrt":
            k = max(1, int(np.sqrt(n_total)))
        elif self.n_features == "log2":
            k = max(1, int(np.log2(n_total)))
        else:
            k = int(self.n_features)
        return self._rng.choice(n_total, size=k, replace=False)

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _majority_class(y: np.ndarray) -> int:
        return int(Counter(y).most_common(1)[0][0])

    def _predict_row(self, row: np.ndarray, node: _Node) -> int:
        if node.is_leaf:
            return node.value  # type: ignore[return-value]
        if row[node.feature_idx] <= node.threshold:
            return self._predict_row(row, node.left)
        return self._predict_row(row, node.right)

    def _predict_proba_row(self, row: np.ndarray, node: _Node) -> np.ndarray:
        """Walk to a leaf and return [P(class=0), P(class=1)]."""
        if node.is_leaf:
            proba = np.zeros(2)
            proba[node.value] = 1.0
            return proba
        if row[node.feature_idx] <= node.threshold:
            return self._predict_proba_row(row, node.left)
        return self._predict_proba_row(row, node.right)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_depth(self) -> int:
        """Return the actual depth of the fitted tree."""
        return self._depth(self.root)

    def _depth(self, node: _Node | None) -> int:
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self._depth(node.left), self._depth(node.right))

    def count_leaves(self) -> int:
        return self._count_leaves(self.root)

    def _count_leaves(self, node: _Node | None) -> int:
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def __repr__(self) -> str:
        return (
            f"DecisionTree("
            f"max_depth={self.max_depth}, "
            f"min_samples_split={self.min_samples_split}, "
            f"min_samples_leaf={self.min_samples_leaf})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score

    rng = np.random.default_rng(42)
    X, y = make_classification(n_samples=500, n_features=10,
                                random_state=42)
    split = int(0.8 * len(y))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    tree = DecisionTree(max_depth=5, min_samples_split=5, min_samples_leaf=2)
    tree.fit(X_tr, y_tr)
    preds = tree.predict(X_te)
    acc   = accuracy_score(y_te, preds)
    print(f"Smoke-test accuracy: {acc:.4f}")
    print(f"Tree depth : {tree.get_depth()}")
    print(f"Leaf count : {tree.count_leaves()}")
