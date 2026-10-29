"""Nested cross-validation splits and leakage-safe preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def outer_splits(n_samples: int, n_splits: int = 5, seed: int = 2026) -> list[tuple[np.ndarray, np.ndarray]]:
    """Outer folds: every specimen is a blind test case exactly once."""

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.arange(n_samples)))


def inner_splits(
    train_idx: np.ndarray,
    n_repeats: int = 3,
    validation_fraction: float = 0.25,
    seed: int = 2026,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Repeated train/validation splits drawn *only* from the outer train set."""

    splitter = ShuffleSplit(n_splits=n_repeats, test_size=validation_fraction, random_state=seed)
    train_idx = np.asarray(train_idx)
    return [(train_idx[a], train_idx[b]) for a, b in splitter.split(train_idx)]


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer([("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)])


def fit_transform_features(
    table: pd.DataFrame,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    numeric: list[str],
    categorical: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit scaling/encoding on ``train_idx`` only, then transform both sets."""

    columns = numeric + categorical
    preprocessor = make_preprocessor(numeric, categorical)
    preprocessor.fit(table.iloc[train_idx][columns])
    x_train = preprocessor.transform(table.iloc[train_idx][columns]).astype(np.float32)
    x_eval = preprocessor.transform(table.iloc[eval_idx][columns]).astype(np.float32)
    return x_train, x_eval
