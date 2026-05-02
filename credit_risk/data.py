"""Loading and temporal splitting for Lending Club data."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

TRAIN_END_YEAR = 2015
TEST_START_YEAR = 2016

COMPLETED_STATUSES = {"Fully Paid", "Charged Off"}

# Columns that are only known after a loan is issued — must not be used as features.
# Kept in the DataFrame for computing realized profit on the test period.
POST_ISSUANCE_COLUMNS = [
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_credit_pull_d",
    "last_fico_range_high",
    "last_fico_range_low",
    "out_prncp",
    "out_prncp_inv",
    "hardship_flag",
    "hardship_type",
    "hardship_reason",
    "hardship_status",
    "hardship_amount",
    "hardship_start_date",
    "hardship_end_date",
    "payment_plan_start_date",
    "hardship_length",
    "hardship_dpd",
    "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount",
    "hardship_last_payment_amount",
    "debt_settlement_flag",
    "debt_settlement_flag_date",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
]


def load_raw(filename: str = "accepted_2007_to_2018Q4.csv") -> pd.DataFrame:
    """Load the raw Lending Club CSV.

    Parses `issue_d` (e.g. "Dec-2015") into a real datetime so downstream code
    can group by year without re-parsing.
    """
    path = RAW_DIR / filename
    df = pd.read_csv(path, low_memory=False)
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    return df
