"""Group-level fairness metrics for credit decisions."""

import numpy as np
import pandas as pd


def group_metrics(
    df: pd.DataFrame,
    group_col: str,
    approved_col: str = "approved",
    outcome_col: str = "defaulted",
    profit_col: str | None = "realized_profit",
    min_n: int = 100,
) -> pd.DataFrame:
    """Per-group: sample size, approval rate, default rate among approved, profit per approved loan.

    Groups smaller than `min_n` are dropped to avoid noisy ratios.
    """
    rows = []
    for g, sub in df.groupby(group_col, observed=True):
        if len(sub) < min_n:
            continue
        approved = sub[sub[approved_col]]
        rows.append({
            group_col: g,
            "n": len(sub),
            "n_approved": len(approved),
            "approval_rate": float(sub[approved_col].mean()),
            "default_rate_all": float(sub[outcome_col].mean()),
            "default_rate_approved": (
                float(approved[outcome_col].mean()) if len(approved) > 0 else np.nan
            ),
            "profit_per_approved": (
                float(approved[profit_col].mean())
                if profit_col is not None and len(approved) > 0
                else np.nan
            ),
        })
    # Explicit columns so an all-groups-below-min_n call returns an empty typed
    # frame rather than blowing up in sort_values on a column-less DataFrame.
    columns = [
        group_col,
        "n",
        "n_approved",
        "approval_rate",
        "default_rate_all",
        "default_rate_approved",
        "profit_per_approved",
    ]
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("approval_rate", ascending=False)
        .reset_index(drop=True)
    )
