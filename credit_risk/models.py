"""Model training and calibration."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def numeric_cols(X: pd.DataFrame) -> list[str]:
    """Numeric (including bool) columns. Uses pandas' own dtype check so it
    handles object, StringDtype, the new pandas 3.x `str` dtype, and
    PyArrow-backed strings uniformly."""
    return [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]


def categorical_cols(X: pd.DataFrame) -> list[str]:
    """Anything not numeric. Datetime columns should be removed by the caller."""
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


def train_logistic(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """LR with median-impute + scale + one-hot.

    `min_frequency=100` collapses rare addr_state / sub_grade levels into an
    "infrequent" bucket, which keeps the encoded feature count manageable and
    prevents overfit on tiny strata.
    """
    num_cols = numeric_cols(X)
    cat_cols = categorical_cols(X)

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("encode", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=100,
            sparse_output=False,
        )),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X, y)
    return pipe
