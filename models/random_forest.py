"""
models/random_forest.py
-----------------------
Random Forest classifier implemented FROM SCRATCH.

How it differs from plain Bagging
----------------------------------
Bagging         : bootstrap samples  +  all features considered at every split
Random Forest   : bootstrap samples  +  RANDOM FEATURE SUBSET at every split
                                        (default = sqrt(n_features))

The extra feature randomness decorrelates the trees further, which reduces
variance more than Bagging alone and typically gives better generalisation.

The heavy lifting is done by your existing DecisionTree, which already
supports the  n_features  parameter for per-split feature subsampling.
We just wire everything together here.

No scikit-learn ensemble functions are used.
"""

import numpy as np
from models.decision_tree import DecisionTree          


class RandomForest:
    """
    Random Forest ensemble of DecisionTree classifiers.

    Parameters
    ----------
    n_estimators       : number of trees.
    max_features       : features to consider at each split.
                         'sqrt'  → sqrt(n_features)   [RF default, best for classification]
                         'log2'  → log2(n_features)
                         int     → exact number
                         float   → fraction of total features
                         None    → all features  (degenerates to plain Bagging)
    max_samples        : fraction of training rows per bootstrap  (1.0 = same size).
    max_depth          : maximum depth of each tree.
    min_samples_split  : minimum samples needed to attempt a split.
    min_samples_leaf   : minimum samples required in each child after a split.
    random_seed        : master seed for full reproducibility.
    """

    def __init__(
        self,
        n_estimators:      int              = 100,
        max_features:      int | float | str | None = "sqrt",
        max_samples:       float            = 1.0,
        max_depth:         int | None       = None,
        min_samples_split: int              = 2,
        min_samples_leaf:  int              = 1,
        random_seed:       int              = 42,
    ):
        self.n_estimators      = n_estimators
        self.max_features      = max_features
        self.max_samples       = max_samples
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.random_seed       = random_seed

        self.estimators_: list[DecisionTree] = []   
        self.n_features_in_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        """
        Train n_estimators trees on bootstrap samples with random feature splits.

        Key difference from Bagging
        ---------------------------
        We pass  n_features=self.max_features  to every DecisionTree.
        Inside _best_split(), the tree will only evaluate a random subset
        of features at EACH NODE (not just once per tree), which is the
        defining property of Random Forests.
        """
        self.estimators_     = []
        self.n_features_in_  = X.shape[1]
        n_train              = X.shape[0]
        n_boot               = max(1, round(self.max_samples * n_train))

        # Resolve float max_features → int so the tree can use it
        n_features_for_tree = self._resolve_max_features(self.n_features_in_)

        rng = np.random.default_rng(self.random_seed)

        for _ in range(self.n_estimators):

            indices = rng.integers(0, n_train, size=n_boot)   
            X_boot  = X[indices]
            y_boot  = y[indices]


            tree = DecisionTree(
                max_depth         = self.max_depth,
                min_samples_split = self.min_samples_split,
                min_samples_leaf  = self.min_samples_leaf,
                n_features        = n_features_for_tree,    
                random_seed       = int(rng.integers(0, 2**31)),
            )
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)

        return self

    def _resolve_max_features(self, n_total: int) -> int | str:
        """
        Convert float fractions to an integer count.
        String strategies ('sqrt', 'log2') are passed straight to DecisionTree.
        """
        mf = self.max_features
        if isinstance(mf, float):
            return max(1, int(mf * n_total))
        return mf      # None / 'sqrt' / 'log2' / int — DecisionTree handles these


    def predict(self, X: np.ndarray) -> np.ndarray:
        """Majority vote across all trees."""
        all_preds = np.array([tree.predict(X) for tree in self.estimators_])

        n_samples   = X.shape[0]
        final_preds = np.empty(n_samples, dtype=int)
        for j in range(n_samples):
            votes = all_preds[:, j]
            final_preds[j] = int(np.bincount(votes).argmax())
        return final_preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Averaged soft probabilities across all trees.

        Returns
        -------
        proba : (n_samples, 2)  —  [P(class=0), P(class=1)]
        """
        proba_sum = np.zeros((X.shape[0], 2), dtype=np.float64)
        for tree in self.estimators_:
            proba_sum += tree.predict_proba(X)
        return proba_sum / self.n_estimators

    def feature_importances(self) -> np.ndarray:
        """
        Compute mean decrease in impurity (MDI) feature importances.

        For every internal node in every tree we accumulate:
            importance[feature] += n_samples_node * information_gain_node

        Then we normalise so importances sum to 1.

        Returns
        -------
        importances : (n_features_in_,) array, sums to 1.
        """
        importances = np.zeros(self.n_features_in_, dtype=np.float64)

        for tree in self.estimators_:
            self._accumulate_importances(tree.root, importances)

        total = importances.sum()
        if total > 0:
            importances /= total
        return importances

    def _accumulate_importances(self, node, importances: np.ndarray) -> None:
        """Recursive helper: walk tree and accumulate weighted IG per feature."""
        if node is None or node.is_leaf:
            return
        feat = node.feature_idx
        if feat is not None and feat < len(importances):
            # weight by number of samples reaching this node
            importances[feat] += node.n_samples * node.impurity
        self._accumulate_importances(node.left,  importances)
        self._accumulate_importances(node.right, importances)


    def __repr__(self) -> str:
        return (
            f"RandomForest("
            f"n_estimators={self.n_estimators}, "
            f"max_features={self.max_features!r}, "
            f"max_depth={self.max_depth})"
        )


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=600, n_features=10, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForest(n_estimators=50, max_features="sqrt", max_depth=5, random_seed=42)
    rf.fit(X_tr, y_tr)

    preds = rf.predict(X_te)
    acc   = accuracy_score(y_te, preds)
    print(f"Smoke-test accuracy  : {acc:.4f}")
    print(f"Ensemble size        : {len(rf.estimators_)} trees")
    print(f"Feature importances  : {rf.feature_importances().round(3)}")
