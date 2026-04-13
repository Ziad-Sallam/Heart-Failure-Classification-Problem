"""
data_preparation.py
-------------------
Handles dataset download (optional), loading, preprocessing,
and stratified train/val/test splitting.

All operations use RANDOM_SEED = 42 for reproducibility.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20

# ---------------------------------------------------------------------------
# 1. Download helper (requires ~/.kaggle/kaggle.json)
# ---------------------------------------------------------------------------

def download_dataset(dest_dir: str = "data") -> str:
    """
    Download the Heart Failure Prediction dataset from Kaggle.

    Requires the Kaggle API token at ~/.kaggle/kaggle.json
    (or KAGGLE_USERNAME / KAGGLE_KEY env vars).

    Returns
    -------
    str : path to the downloaded CSV file.
    """
    import subprocess

    os.makedirs(dest_dir, exist_ok=True)
    cmd = [
        "kaggle", "datasets", "download",
        "-d", "fedesoriano/heart-failure-prediction",
        "--unzip", "-p", dest_dir,
    ]
    subprocess.run(cmd, check=True)
    csv_path = os.path.join(dest_dir, "heart.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Expected '{csv_path}' after download. "
            "Check the dataset contents."
        )
    print(f"[data] Dataset downloaded to: {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# 2. Load
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV and return a DataFrame."""
    df = pd.read_csv(csv_path)
    print(f"[data] Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# 3. Preprocess
# ---------------------------------------------------------------------------

CATEGORICAL_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
TARGET_COL       = "HeartDisease"


def preprocess(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Encode categoricals, separate target, and optionally scale features.

    Parameters
    ----------
    df          : raw DataFrame (may include target column).
    scaler      : existing StandardScaler to reuse (pass None to create one).
    fit_scaler  : if True, fit the scaler on this data (use for training set only).

    Returns
    -------
    X      : float32 feature matrix (n_samples, n_features)
    y      : int array of labels (n_samples,)
    scaler : fitted StandardScaler (reuse for val/test)
    """
    df = df.copy()

    # One-hot encode categorical columns present in the data
    cols_to_encode = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=False)

    # Separate features and target
    y = df[TARGET_COL].values.astype(int)
    X_df = df.drop(columns=[TARGET_COL])

    # Convert booleans produced by get_dummies to int
    bool_cols = X_df.select_dtypes(include="bool").columns
    X_df[bool_cols] = X_df[bool_cols].astype(int)

    X = X_df.values.astype(np.float32)

    # Scale
    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y, scaler


# ---------------------------------------------------------------------------
# 4. Split
# ---------------------------------------------------------------------------

def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    test_ratio:  float = TEST_RATIO,
    random_seed: int   = RANDOM_SEED,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Stratified split into train / val / test.

    Returns
    -------
    dict with keys 'train', 'val', 'test', each a (X, y) tuple.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1."

    # First cut off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        stratify=y,
        random_state=random_seed,
    )

    # Split remainder into train / val
    relative_val = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=relative_val,
        stratify=y_temp,
        random_state=random_seed,
    )

    splits = {
        "train": (X_train, y_train),
        "val":   (X_val,   y_val),
        "test":  (X_test,  y_test),
    }

    for name, (Xs, ys) in splits.items():
        pos_rate = ys.mean()
        print(f"[data] {name:5s}: {len(ys):4d} samples  "
              f"positive-class rate = {pos_rate:.3f}")

    return splits


# ---------------------------------------------------------------------------
# 5. End-to-end helper
# ---------------------------------------------------------------------------

def prepare_data(
    csv_path: str | None = None,
    dest_dir: str = "data",
) -> tuple[dict, StandardScaler, list[str]]:
    """
    Full pipeline: download (if needed) → load → preprocess → split.

    Parameters
    ----------
    csv_path : path to heart.csv; if None, dataset is downloaded via Kaggle API.
    dest_dir : directory used when downloading.

    Returns
    -------
    splits  : dict{'train':(X,y), 'val':(X,y), 'test':(X,y)}
    scaler  : fitted StandardScaler (so you can inverse-transform if needed)
    feature_names : list of feature column names after encoding
    """
    # --- Locate or download data ---
    if csv_path is None:
        default = os.path.join(dest_dir, "heart.csv")
        if os.path.exists(default):
            csv_path = default
            print(f"[data] Found existing file at {csv_path}")
        else:
            csv_path = download_dataset(dest_dir)

    df = load_dataset(csv_path)

    # Build feature names list (after encoding) for reference
    df_encoded = pd.get_dummies(
        df.drop(columns=[TARGET_COL]),
        columns=[c for c in CATEGORICAL_COLS if c in df.columns],
        drop_first=False,
    )
    bool_cols = df_encoded.select_dtypes(include="bool").columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    feature_names = list(df_encoded.columns)

    # Preprocess (fit scaler on full data temporarily to get object, then redo per split)
    X_all, y_all, _ = preprocess(df, fit_scaler=True)

    # Stratified split (on raw scaled data — scaler fitted on train below)
    # We redo preprocessing properly: fit only on train
    splits_idx = split_dataset(X_all, y_all)   # used only to get index masks

    # Re-split the raw (unscaled) data by replaying the split logic on df rows
    df_encoded[TARGET_COL] = df[TARGET_COL].values
    X_raw = df_encoded.drop(columns=[TARGET_COL]).values.astype(np.float32)
    y_all2 = df_encoded[TARGET_COL].values.astype(int)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_raw, y_all2,
        test_size=TEST_RATIO, stratify=y_all2, random_state=RANDOM_SEED,
    )
    relative_val = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=relative_val, stratify=y_temp, random_state=RANDOM_SEED,
    )

    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    splits = {
        "train": (X_train_s, y_train),
        "val":   (X_val_s,   y_val),
        "test":  (X_test_s,  y_test),
    }

    print("\n[data] Final split sizes (after fitting scaler on train only):")
    for name, (Xs, ys) in splits.items():
        print(f"  {name:5s}: {Xs.shape[0]:4d} samples | "
              f"features={Xs.shape[1]} | "
              f"positive rate={ys.mean():.3f}")

    return splits, scaler, feature_names


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    splits, scaler, features = prepare_data(csv_path="data/heart.csv")
    print("\nFeature names:", features)
