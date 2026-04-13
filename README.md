# Heart-Failure-Classification-Problem

A Decision Tree classifier built **from scratch** using Information Gain (Shannon entropy) for the [Heart Failure Prediction dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (918 samples, 11 features, binary classification).

---

## Project Structure

```
heart_failure/
├── data_preparation.py       # Data loading, encoding, scaling, splitting
├── models/
│   ├── __init__.py
│   └── decision_tree.py      # Decision Tree implementation (no sklearn)
├── utils/
│   ├── __init__.py
│   ├── evaluation.py         # Metrics: accuracy, F1, AUC, confusion matrix
│   └── tuning.py             # Grid search & random search
├── train_decision_tree.py    # Full training + tuning + evaluation script
├── results/                  # Output summaries saved here
└── requirements.txt
```

---

## Requirements

```bash
pip install -r requirements.txt
```

| Package      | Version  |
|--------------|----------|
| pandas       | ≥ 1.5.0  |
| numpy        | ≥ 1.23.0 |
| scikit-learn | ≥ 1.2.0  |
| kaggle       | ≥ 1.5.12 |

---

## Dataset Setup

### Option A — Manual download (recommended)
1. Download `heart.csv` from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
2. Place it in the `data/` folder:
   ```
   heart_failure/data/heart.csv
   ```

### Option B — Kaggle API (automatic)
1. Create a Kaggle account and generate an API token from **Account → API → Create New Token**
2. Place `kaggle.json` at `~/.kaggle/kaggle.json`
3. Run the script without `--csv` and it will download automatically

---

## Quickstart

### Run the full pipeline
```bash
python train_decision_tree.py --csv data/heart.csv
```

This will:
1. Load and preprocess the dataset
2. Split into 70% train / 10% val / 20% test (stratified, seed 42)
3. Run grid search over hyperparameters using the validation set
4. Evaluate the best model on all three splits
5. Save a summary to `results/decision_tree_results.txt`

---

## Using the Decision Tree in Your Own Code

### Basic usage
```python
from models.decision_tree import DecisionTree
import numpy as np

# Training data
X_train = np.array([[2.5, 1.0], [1.0, 3.0], [3.0, 2.0], [1.5, 0.5]])
y_train = np.array([0, 1, 0, 1])

# Create and train the tree
tree = DecisionTree(max_depth=5, min_samples_split=2, min_samples_leaf=1)
tree.fit(X_train, y_train)

# Predict
X_new = np.array([[2.0, 1.5], [1.2, 2.8]])
predictions = tree.predict(X_new)
print(predictions)  # e.g. [0, 1]

# Predict probabilities
probabilities = tree.predict_proba(X_new)
print(probabilities)  # e.g. [[1.0, 0.0], [0.0, 1.0]]
```

### With the Heart Failure dataset
```python
from data_preparation import prepare_data
from models.decision_tree import DecisionTree
from utils.evaluation import evaluate

# Load and split data
splits, scaler, feature_names = prepare_data(csv_path="data/heart.csv")
X_train, y_train = splits["train"]
X_val,   y_val   = splits["val"]
X_test,  y_test  = splits["test"]

# Train
tree = DecisionTree(max_depth=10, min_samples_split=2, min_samples_leaf=2)
tree.fit(X_train, y_train)

# Evaluate on test set
y_pred  = tree.predict(X_test)
y_proba = tree.predict_proba(X_test)[:, 1]  # probability of positive class

metrics = evaluate(y_test, y_pred, y_proba, model_name="My Tree", split_name="Test")
```

---

## Hyperparameter Reference

| Parameter           | Type        | Default | Description                                              |
|---------------------|-------------|---------|----------------------------------------------------------|
| `max_depth`         | int or None | None    | Maximum depth of the tree. `None` = grow until pure.     |
| `min_samples_split` | int         | 2       | Minimum samples in a node required to attempt a split.   |
| `min_samples_leaf`  | int         | 1       | Minimum samples required in each child after a split.    |
| `n_features`        | int / str / None | None | Features to consider per split: `None` (all), `'sqrt'`, `'log2'`, or an integer. Useful for building Random Forests. |
| `random_seed`       | int         | 42      | Seed for feature subsampling reproducibility.            |

### Recommended ranges for tuning
```python
param_grid = {
    "max_depth":         [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 5],
}
```

---

## Hyperparameter Tuning

### Grid search
```python
from utils.tuning import grid_search
from models.decision_tree import DecisionTree

param_grid = {
    "max_depth":         [3, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 5],
}

best_model, best_params, all_results = grid_search(
    model_cls = DecisionTree,
    param_grid = param_grid,
    X_train = X_train, y_train = y_train,
    X_val   = X_val,   y_val   = y_val,
    scoring = "f1_binary",   # or "accuracy", "f1_macro", "roc_auc"
    verbose = True,
)

print("Best params:", best_params)
```

### Random search
```python
from utils.tuning import random_search

param_distributions = {
    "max_depth":         [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 3, 5],
}

best_model, best_params, all_results = random_search(
    model_cls            = DecisionTree,
    param_distributions  = param_distributions,
    X_train = X_train, y_train = y_train,
    X_val   = X_val,   y_val   = y_val,
    n_iter      = 20,
    scoring     = "f1_binary",
    random_seed = 42,
)
```

---

## Tree Diagnostics

```python
tree.fit(X_train, y_train)

print(tree.get_depth())    # actual depth of the fitted tree
print(tree.count_leaves()) # number of leaf nodes
print(repr(tree))          # DecisionTree(max_depth=10, ...)
```

---

## Evaluation Metrics

`evaluate()` returns a dict and prints a formatted summary:

```python
from utils.evaluation import evaluate

metrics = evaluate(
    y_true     = y_test,
    y_pred     = y_pred,
    y_proba    = y_proba,   # optional, needed for ROC-AUC
    model_name = "Decision Tree",
    split_name = "Test",
)

# Available keys:
# metrics["accuracy"]          float
# metrics["precision"]         float  (macro)
# metrics["recall"]            float  (macro)
# metrics["f1_macro"]          float
# metrics["f1_binary"]         float
# metrics["roc_auc"]           float  (if y_proba provided)
# metrics["confusion_matrix"]  2x2 numpy array
```

### Compare multiple models side by side
```python
from utils.evaluation import compare_models

results = {
    "Decision Tree": dt_metrics,
    "KNN":           knn_metrics,
}
compare_models(results, split="test")
```

---

## How It Works

The tree is built recursively using **Information Gain**:

```
IG(parent, left, right) = H(parent) − [|left|/|parent| · H(left) + |right|/|parent| · H(right)]
```

where `H` is Shannon entropy: `H(y) = −Σ p(c) · log₂ p(c)`

At each node the algorithm:
1. Iterates over all features (or a random subset if `n_features` is set)
2. Tries every midpoint between consecutive unique values as a threshold
3. Picks the split with the highest Information Gain
4. Recurses on left (`≤ threshold`) and right (`> threshold`) children
5. Stops when `max_depth` is reached, the node is pure, or `min_samples_split` / `min_samples_leaf` constraints are violated

---

## Adding More Models

The modular design makes adding new classifiers straightforward. Create a new file in `models/` (e.g. `models/knn.py`) with `.fit()` and `.predict()` methods, then reuse `data_preparation`, `utils/evaluation`, and `utils/tuning` without any changes.