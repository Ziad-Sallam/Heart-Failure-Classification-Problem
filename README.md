# Heart-Failure-Classification-Problem

A Decision Tree classifier built **from scratch** using Information Gain (Shannon entropy) for the [Heart Failure Prediction dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (918 samples, 11 features, binary classification).

---

## Project Structure

```
heart_failure/
├── data_preparation.py       # Data loading, encoding, scaling, splitting
├── models/
│   ├── __init__.py
│   ├── decision_tree.py      # Decision Tree implementation (no sklearn)
│   ├── bagging.py            # Bagging ensemble (Bootstrap Aggregating)
│   ├── random_forest.py      # Random Forest (Bagging + random feature subsets)
│   └── adaboost.py           # AdaBoost ensemble (Adaptive Boosting)
├── utils/
│   ├── __init__.py
│   ├── evaluation.py         # Metrics: accuracy, F1, AUC, confusion matrix
│   └── tuning.py             # Grid search & random search
├── train_decision_tree.py    # Full training + tuning + evaluation script
├── train_bagging.py          # Train, tune, and evaluate the Bagging ensemble
├── train_random_forest.py    # Train, tune, and evaluate the Random Forest
├── train_adaboost.py         # Train, tune, and evaluate the AdaBoost ensemble
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

## Using the Decision Tree 

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

## Using Bagging 

### Basic usage
```python
from models.bagging import BaggingClassifier
import numpy as np

# Training data
X_train = np.array([[2.5, 1.0], [1.0, 3.0], [3.0, 2.0], [1.5, 0.5]])
y_train = np.array([0, 1, 0, 1])

# Create and train the ensemble
bag = BaggingClassifier(n_estimators=10, max_depth=5, random_seed=42)
bag.fit(X_train, y_train)

# Predict
X_new = np.array([[2.0, 1.5], [1.2, 2.8]])
predictions = bag.predict(X_new)
print(predictions)  # e.g. [0, 1]

# Predict probabilities
probabilities = bag.predict_proba(X_new)
print(probabilities)  # e.g. [[0.6, 0.4], [0.3, 0.7]]
```

### With the Heart Failure dataset
```python
from data_preparation import prepare_data
from models.bagging import BaggingClassifier
from utils.evaluation import evaluate

# Load and split data
splits, scaler, feature_names = prepare_data(csv_path="data/heart.csv")
X_train, y_train = splits["train"]
X_val,   y_val   = splits["val"]
X_test,  y_test  = splits["test"]

# Train
bag = BaggingClassifier(n_estimators=50, max_depth=5, min_samples_split=2, min_samples_leaf=1, random_seed=42)
bag.fit(X_train, y_train)

# Evaluate on test set
y_pred  = bag.predict(X_test)
y_proba = bag.predict_proba(X_test)[:, 1]  # probability of positive class

metrics = evaluate(y_test, y_pred, y_proba, model_name="Bagging", split_name="Test")
```

---

## Using Random Forest 

### Basic usage
```python
from models.random_forest import RandomForest
import numpy as np

# Training data
X_train = np.array([[2.5, 1.0], [1.0, 3.0], [3.0, 2.0], [1.5, 0.5]])
y_train = np.array([0, 1, 0, 1])

# Create and train the ensemble
rf = RandomForest(n_estimators=50, max_features='sqrt', max_depth=10, random_seed=42)
rf.fit(X_train, y_train)

# Predict
X_new = np.array([[2.0, 1.5], [1.2, 2.8]])
predictions = rf.predict(X_new)
print(predictions)  # e.g. [0, 1]

# Predict probabilities
probabilities = rf.predict_proba(X_new)
print(probabilities)  # e.g. [[0.7, 0.3], [0.2, 0.8]]

# Feature importances
importances = rf.feature_importances()
print(importances)  # dict of feature -> importance score
```

### With the Heart Failure dataset
```python
from data_preparation import prepare_data
from models.random_forest import RandomForest
from utils.evaluation import evaluate

# Load and split data
splits, scaler, feature_names = prepare_data(csv_path="data/heart.csv")
X_train, y_train = splits["train"]
X_val,   y_val   = splits["val"]
X_test,  y_test  = splits["test"]

# Train
rf = RandomForest(n_estimators=50, max_features='sqrt', max_depth=10, min_samples_split=2, min_samples_leaf=1, random_seed=42)
rf.fit(X_train, y_train)

# Evaluate on test set
y_pred  = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]  # probability of positive class

metrics = evaluate(y_test, y_pred, y_proba, model_name="Random Forest", split_name="Test")

# Get feature importances
importances = rf.feature_importances()
print("Top features:", sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5])
```

---

## Using AdaBoost 

### Basic usage
```python
from models.adaboost import AdaBoostClassifier
import numpy as np

# Training data
X_train = np.array([[2.5, 1.0], [1.0, 3.0], [3.0, 2.0], [1.5, 0.5]])
y_train = np.array([0, 1, 0, 1])

# Create and train the ensemble
ada = AdaBoostClassifier(n_estimators=50, max_depth=1, random_seed=42)
ada.fit(X_train, y_train)

# Predict
X_new = np.array([[2.0, 1.5], [1.2, 2.8]])
predictions = ada.predict(X_new)
print(predictions)  # e.g. [0, 1]

# Predict probabilities
probabilities = ada.predict_proba(X_new)
print(probabilities)  # e.g. [[0.6, 0.4], [0.3, 0.7]]
```

### With the Heart Failure dataset
```python
from data_preparation import prepare_data
from models.adaboost import AdaBoostClassifier
from utils.evaluation import evaluate

# Load and split data
splits, scaler, feature_names = prepare_data(csv_path="data/heart.csv")
X_train, y_train = splits["train"]
X_val,   y_val   = splits["val"]
X_test,  y_test  = splits["test"]

# Train
ada = AdaBoostClassifier(n_estimators=50, max_depth=2, random_seed=42)
ada.fit(X_train, y_train)

# Evaluate on test set
y_pred  = ada.predict(X_test)
y_proba = ada.predict_proba(X_test)[:, 1]  # probability of positive class

metrics = evaluate(y_test, y_pred, y_proba, model_name="AdaBoost", split_name="Test")
```

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

## Hyperparameter Reference for Bagging

| Parameter           | Type        | Default | Description                                              |
|---------------------|-------------|---------|----------------------------------------------------------|
| `n_estimators`      | int         | 10      | Number of trees in the ensemble.                         |
| `max_samples`       | float       | 1.0     | Fraction of training samples per bootstrap (1.0 = same size). |
| `max_depth`         | int or None | None    | Maximum depth of each tree. `None` = grow until pure.    |
| `min_samples_split` | int         | 2       | Minimum samples in a node required to attempt a split.   |
| `min_samples_leaf`  | int         | 1       | Minimum samples required in each child after a split.    |
| `n_features`        | int / str / None | None | Features to consider per split: `None` (all), `'sqrt'`, `'log2'`, or an integer. |
| `random_seed`       | int         | 42      | Seed for bootstrap sampling and tree reproducibility.    |

### Recommended ranges for tuning Bagging
```python
param_grid = {
    "n_estimators":      [10, 20, 50],
    "max_depth":         [3, 5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
}
```

---

## Hyperparameter Reference for Random Forest

| Parameter           | Type              | Default | Description                                              |
|---------------------|-------------------|---------|----------------------------------------------------------|
| `n_estimators`      | int               | 100     | Number of trees in the ensemble.                         |
| `max_features`      | int / float / str / None | 'sqrt' | Features to consider per split: `'sqrt'`, `'log2'`, float fraction, or integer. |
| `max_samples`       | float             | 1.0     | Fraction of training samples per bootstrap (1.0 = same size). |
| `max_depth`         | int or None       | None    | Maximum depth of each tree. `None` = grow until pure.    |
| `min_samples_split` | int               | 2       | Minimum samples in a node required to attempt a split.   |
| `min_samples_leaf`  | int               | 1       | Minimum samples required in each child after a split.    |
| `random_seed`       | int               | 42      | Seed for bootstrap sampling and tree reproducibility.    |

### Recommended ranges for tuning Random Forest
```python
param_grid = {
    "n_estimators":      [50, 100, 200],
    "max_features":      ['sqrt', 'log2'],
    "max_depth":         [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
}
```

---

## Hyperparameter Reference for AdaBoost

| Parameter      | Type | Default | Description                                              |
|----------------|------|---------|----------------------------------------------------------|
| `n_estimators` | int  | 50      | Maximum number of boosting rounds (weak learners).      |
| `max_depth`    | int  | 1       | Maximum depth of each weak learner (1 = decision stump).|
| `random_seed`  | int  | 42      | Seed for sample weighting and learner reproducibility.   |

### Recommended ranges for tuning AdaBoost
```python
param_grid = {
    "n_estimators": [10, 25, 50, 75, 100],
    "max_depth":    [1, 2],
}
```

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

### Grid search for Bagging
```python
from utils.tuning import grid_search
from models.bagging import BaggingClassifier

param_grid = {
    "n_estimators":      [10, 20, 50],
    "max_depth":         [3, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
}

best_model, best_params, all_results = grid_search(
    model_cls = BaggingClassifier,
    param_grid = param_grid,
    X_train = X_train, y_train = y_train,
    X_val   = X_val,   y_val   = y_val,
    scoring = "f1_binary",
    verbose = True,
)
```

### Grid search for Random Forest
```python
from utils.tuning import grid_search
from models.random_forest import RandomForest

param_grid = {
    "n_estimators":      [50, 100],
    "max_features":      ['sqrt', 'log2'],
    "max_depth":         [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
}

best_model, best_params, all_results = grid_search(
    model_cls = RandomForest,
    param_grid = param_grid,
    X_train = X_train, y_train = y_train,
    X_val   = X_val,   y_val   = y_val,
    scoring = "f1_binary",
    verbose = True,
)
```

### Grid search for AdaBoost
```python
from utils.tuning import grid_search
from models.adaboost import AdaBoostClassifier

param_grid = {
    "n_estimators": [10, 25, 50, 75, 100],
    "max_depth":    [1, 2],
}

best_model, best_params, all_results = grid_search(
    model_cls = AdaBoostClassifier,
    param_grid = param_grid,
    X_train = X_train, y_train = y_train,
    X_val   = X_val,   y_val   = y_val,
    scoring = "f1_binary",
    verbose = True,
)
```

## Tree Diagnostics

```python
tree.fit(X_train, y_train)

print(tree.get_depth())    # actual depth of the fitted tree
print(tree.count_leaves()) # number of leaf nodes
print(repr(tree))          # DecisionTree(max_depth=10, ...)
```

## Ensemble Diagnostics

```python
bag.fit(X_train, y_train)

print(len(bag.estimators_))  # number of trees
print(repr(bag))             # BaggingClassifier(n_estimators=50, ...)

rf.fit(X_train, y_train)

print(len(rf.estimators_))   # number of trees
print(repr(rf))              # RandomForest(n_estimators=100, ...)
# Feature importances (Random Forest only)
importances = rf.feature_importances()
for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    print(f"{feat}: {imp:.3f}")

ada.fit(X_train, y_train)

print(len(ada.estimators_))     # number of weak learners
print(len(ada.estimator_weights_))  # number of weights (same as estimators)
print(repr(ada))                # AdaBoostClassifier(n_estimators=50, ...)
```

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

## Ensemble Methods: Bagging, Random Forest & AdaBoost

This part extends the project with three from-scratch ensemble classifiers built on top of the existing `DecisionTree` implementation.

---

### New Files

```
heart_failure/
├── models/
│   ├── bagging.py            # Bagging ensemble (Bootstrap Aggregating)
│   ├── random_forest.py      # Random Forest (Bagging + random feature subsets)
│   └── adaboost.py           # AdaBoost ensemble (Adaptive Boosting)
├── train_bagging.py          # Train, tune, and evaluate the Bagging ensemble
├── train_random_forest.py    # Train, tune, and evaluate the Random Forest
└── train_adaboost.py         # Train, tune, and evaluate the AdaBoost ensemble
```

---

### How to Run

```bash
# Train and evaluate Bagging
python train_bagging.py --csv data/heart.csv

# Train and evaluate Random Forest
python train_random_forest.py --csv data/heart.csv

# Train and evaluate AdaBoost
python train_adaboost.py --csv data/heart.csv
```

Each script will:
1. Load and preprocess the dataset via `data_preparation.py`
2. Run grid search over hyperparameters on the validation set
3. Retrain the best model on train + validation combined
4. Evaluate on all three splits (Train / Validation / Test)
5. Save a results summary to `results/`

The Random Forest script additionally saves:
- `results/rf_feature_importances.png` — Top-15 features by Mean Decrease in Impurity
- `results/rf_confusion_matrix.png` — Test set confusion matrix heatmap

---

### How Bagging Works

`BaggingClassifier` trains `n_estimators` independent `DecisionTree` instances, each on a **bootstrap sample** (random rows drawn *with replacement*) of the training data. Predictions are aggregated by **majority vote** (`predict`) or **averaged probabilities** (`predict_proba`). Because each tree sees a slightly different dataset, the ensemble reduces variance compared to a single tree.

Key parameters used in tuning:

| Parameter | Values searched |
|---|---|
| `n_estimators` | 10, 20, 50 |
| `max_depth` | 3, 5, 10, None |
| `min_samples_split` | 2, 5 |
| `min_samples_leaf` | 1, 2 |

**Best configuration:** `n_estimators=50, max_depth=5, min_samples_split=2, min_samples_leaf=1`

---

### How Random Forest Works

`RandomForest` adds one critical change on top of Bagging: at **every split in every tree**, only a random subset of features is considered (`max_features='sqrt'` by default). This further decorrelates the trees beyond what bootstrap sampling alone achieves, reducing variance more and typically improving generalisation.

> Bagging = bootstrap rows + **all features** per split  
> Random Forest = bootstrap rows + **random feature subset** per split

Key parameters used in tuning:

| Parameter | Values searched |
|---|---|
| `n_estimators` | 50, 100, 200 |
| `max_features` | `'sqrt'`, `'log2'` |
| `max_depth` | 5, 10, None |
| `min_samples_split` | 2, 5 |
| `min_samples_leaf` | 1, 2 |

**Best configuration:** `n_estimators=50, max_features='sqrt', max_depth=10, min_samples_split=2, min_samples_leaf=1`  
With 11 features, `sqrt` ≈ **3 features** evaluated per split.

---

### How AdaBoost Works

`AdaBoostClassifier` trains `n_estimators` weak learners (Decision Trees with `max_depth=1` by default, called "decision stumps") sequentially. Each learner focuses on the examples that the previous learners misclassified by updating sample weights. Predictions are aggregated by **weighted majority vote** (`predict`) or **normalized alpha sum** (`predict_proba`).

Key differences from Bagging/Random Forest:
- **Sequential** training (learners depend on each other) vs. parallel
- **Weighted** sampling (harder examples get higher weights) vs. uniform bootstrap
- **Adaptive** (each learner corrects the mistakes of the previous) vs. independent

Key parameters used in tuning:

| Parameter | Values searched |
|---|---|
| `n_estimators` | 10, 25, 50, 75, 100 |
| `max_depth` | 1, 2 |

**Best configuration:** `n_estimators=50, max_depth=2`

### Results Summary

#### Decision Tree

| Split | Accuracy | F1 (binary) | ROC-AUC |
|---|---|---|---|
| Train | 0.9766 | 0.9790 | 0.9755 |
| Validation | 0.9565 | 0.9623 | 0.9512 |
| Test | 0.7826 | 0.8039| 0.7800 |

Test confusion matrix: TN=62, FP=20, FN=20, TP=82

#### Bagging

| Split | Accuracy | F1 (binary) | ROC-AUC |
|---|---|---|---|
| Train | 0.9237 | 0.9332 | 0.9726 |
| Validation | 0.9239 | 0.9333 | 0.9605 |
| Test | 0.8641 | 0.8804 | 0.9148 |

Test confusion matrix: TN=67, FP=15, FN=10, TP=92

#### Random Forest

| Split | Accuracy | F1 (binary) | ROC-AUC |
|---|---|---|---|
| Train | 0.9735 | 0.9763 | 0.9983 |
| Validation | 0.9565 | 0.9623 | 0.9959 |
| Test | 0.9022 | 0.9143 | 0.9333 |

Test confusion matrix: TN=70, FP=12, FN=6, TP=96

#### AdaBoost

| Split | Accuracy | F1 (binary) | ROC-AUC |
|---|---|---|---|
| Train | 0.8972 | 0.9081 | 0.9698 |
| Validation | 0.9130 | 0.9216 | 0.9646 |
| Test | 0.8696 | 0.8800 | 0.9134 |

Test confusion matrix: TN=72, FP=10, FN=14, TP=88

Random Forest outperforms Bagging across every metric and every split, with the test F1 improving from **0.880 → 0.914** and ROC-AUC from **0.915 → 0.933**. AdaBoost performs similarly to Bagging with slightly lower variance.

---

### Top Predictive Features (Random Forest)

Based on Mean Decrease in Impurity across all 50 trees:

| Rank | Feature | Importance |
|---|---|---|
| 1 | ChestPainType_ASY | 0.100 |
| 2 | Cholesterol | 0.096 |
| 3 | Oldpeak | 0.090 |
| 4 | MaxHR | 0.087 |
| 5 | ST_Slope_Up | 0.085 |

Asymptomatic chest pain is the single strongest predictor, consistent with clinical knowledge that silent ischemia is a major heart failure risk factor.

---

### Feature Importance Implementation

Feature importances are computed via **Mean Decrease in Impurity (MDI)**: for every internal node across all trees, the weighted information gain is accumulated per feature, then normalised to sum to 1:

```python
importances[feature] += n_samples_at_node × impurity_at_node
importances /= importances.sum()
```

This is exposed through `RandomForest.feature_importances()` and visualised automatically when running `train_random_forest.py`.
