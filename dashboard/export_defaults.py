"""Export dashboard feature defaults from the training parquet.

The Streamlit app fills in features the user doesn't enter with medians
(numeric) and modes (categorical) from the training data. Computing those
requires data/processed/train.parquet, which is gitignored and doesn't
exist on Streamlit Community Cloud. This script snapshots them into a
small committed JSON that the app reads instead.

Rerun after notebook 02 changes the feature set:

    uv run python dashboard/export_defaults.py
"""

import json
from pathlib import Path

import pandas as pd

from credit_risk import data as cr_data

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def export_feature_defaults() -> Path:
    """Compute medians/modes/category lists from train.parquet and write
    them to dashboard/assets/feature_defaults.json. Mirrors the logic the
    dashboard previously ran inline, so the app behaves identically."""
    train = pd.read_parquet(cr_data.PROCESSED_DIR / "train.parquet")
    meta = ["defaulted", "realized_profit", "issue_d"]
    X = train.drop(columns=meta)

    defaults = {}
    cat_options = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            defaults[col] = float(X[col].median())
        else:
            s = X[col].astype(object)
            defaults[col] = str(s.mode().iloc[0])
            cat_options[col] = sorted(s.dropna().unique().tolist())

    payload = {
        "defaults": defaults,
        "cat_options": cat_options,
        "feature_order": list(X.columns),
    }

    ASSETS_DIR.mkdir(exist_ok=True)
    out_path = ASSETS_DIR / "feature_defaults.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return out_path


if __name__ == "__main__":
    path = export_feature_defaults()
    print(f"Wrote {path} ({path.stat().st_size / 1024:.1f} KB)")
