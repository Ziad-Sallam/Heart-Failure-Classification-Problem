"""
models/bagging.py
-----------------
Bagging (Bootstrap Aggregating) ensemble classifier implemented FROM SCRATCH.

Base learner : DecisionTree (your scratch implementation).
Aggregation  : Majority vote for predict(), averaged probabilities for predict_proba().

No scikit-learn ensemble functions are used.
"""

import numpy as np
from models.decision_tree import DecisionTree          


class BaggingClassifier:
    """
    Bagging ensemble of DecisionTree classifiers.

    Parameters
    ----------
    n_estimators       : number of trees in the ensemble.
    max_samples        : fraction of training samples drawn per bootstrap
                         (1.0 = same size as training set, with replacement).
    max_depth          : max_depth forwarded to each DecisionTree.
    min_samples_split  : forwarded to each DecisionTree.
    min_samples_leaf   : forwarded to each DecisionTree.
    n_features         : feature-subset strategy forwarded to each tree
                         (None = all features; 'sqrt' / 'log2' for subsets).
    random_seed        : master seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators:      int        = 10,
        max_samples:       float      = 1.0,
        max_depth:         int | None = None,
        min_samples_split: int        = 2,
        min_samples_leaf:  int        = 1,
        n_features:        int | str | None = None,
        random_seed:       int        = 42,
    ):
        self.n_estimators      = n_estimators
        self.max_samples       = max_samples
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.n_features        = n_features
        self.random_seed       = random_seed

        self.estimators_: list[DecisionTree] = []  


    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggingClassifier":
        """
        Train n_estimators trees, each on a bootstrap sample of (X, y).

        Bootstrap sampling
        ------------------
        For each tree i, draw n_boot = round(max_samples * n_train) indices
        WITH REPLACEMENT from [0, n_train).  Each tree therefore sees a
        slightly different view of the data, which decorrelates the ensemble.
        """
        self.estimators_ = []
        n_train = X.shape[0]
        n_boot  = max(1, round(self.max_samples * n_train))

        rng = np.random.default_rng(self.random_seed)

        for i in range(self.n_estimators):
            indices = rng.integers(0, n_train, size=n_boot)  
            X_boot  = X[indices]
            y_boot  = y[indices]

            tree = DecisionTree(
                max_depth         = self.max_depth,
                min_samples_split = self.min_samples_split,
                min_samples_leaf  = self.min_samples_leaf,
                n_features        = self.n_features,
                random_seed       = int(rng.integers(0, 2**31)),  
            )
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Majority-vote aggregation across all trees.

        For each sample, collect the predicted class from every tree, then
        return the class that got the most votes.
        """
        # Shape: (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.estimators_])

        # Majority vote along axis=0  (over trees, for each sample)
        # np.apply_along_axis with a mode function, or a simple loop:
        n_samples = X.shape[0]
        final_preds = np.empty(n_samples, dtype=int)
        for j in range(n_samples):
            votes = all_preds[:, j]
            # bincount gives counts for each class; argmax picks the winner
            final_preds[j] = int(np.bincount(votes).argmax())
        return final_preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Averaged soft probabilities across all trees.

        Each tree returns [P(class=0), P(class=1)] per sample.
        We average these vectors across the ensemble.

        Returns
        -------
        proba : (n_samples, 2) array
        """
        # Sum probability arrays from every tree, then divide by n_estimators
        proba_sum = np.zeros((X.shape[0], 2), dtype=np.float64)
        for tree in self.estimators_:
            proba_sum += tree.predict_proba(X)          # (n_samples, 2)
        return proba_sum / self.n_estimators


    def __repr__(self) -> str:
        return (
            f"BaggingClassifier("
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, "
            f"max_samples={self.max_samples})"
        )



if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=600, n_features=10, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    bag = BaggingClassifier(
        n_estimators=20,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_seed=42,
    )
    bag.fit(X_tr, y_tr)
    preds = bag.predict(X_te)
    acc   = accuracy_score(y_te, preds)
    print(f"Smoke-test accuracy : {acc:.4f}")
    print(f"Ensemble size       : {len(bag.estimators_)} trees")
