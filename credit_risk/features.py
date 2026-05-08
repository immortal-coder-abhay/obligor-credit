"""Feature engineering for the credit risk model."""

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "total_acc",
    "revol_bal",
    "revol_util",
    "collections_12_mths_ex_med",
    "mort_acc",
    "tot_cur_bal",
    "total_rev_hi_lim",
    "acc_now_delinq",
]

CATEGORICAL_FEATURES = [
    "term",
    "grade",
    "sub_grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "application_type",
    "initial_list_status",
]


def _parse_emp_length(s) -> float:
    """Convert '10+ years' / '< 1 year' / '5 years' / NaN to float years."""
    if pd.isna(s):
        return float("nan")
    s = str(s).strip()
    if s == "< 1 year":
        return 0.0
    if s == "10+ years":
        return 10.0
    parts = s.split()
    return float(parts[0]) if parts and parts[0].isdigit() else float("nan")


def _parse_pct(series: pd.Series) -> pd.Series:
    """LC stores int_rate / revol_util as either 13.49 (numeric) or '13.49%' (string)."""
    if not pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series.astype(str).str.rstrip("%"), errors="coerce")
    return series
