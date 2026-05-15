"""Economic model: turn predicted default probabilities into approval decisions and profit."""

import numpy as np
import pandas as pd


def expected_profit_per_loan(
    proba_default: np.ndarray,
    interest_if_paid: np.ndarray,
    loss_if_default: np.ndarray,
) -> np.ndarray:
    """E[profit] = (1 - p) * interest_if_paid - p * loss_if_default."""
    return (1 - proba_default) * interest_if_paid - proba_default * loss_if_default


def realized_profit(
    approved: np.ndarray,
    outcomes: np.ndarray,
    interest_if_paid: np.ndarray,
    loss_if_default: np.ndarray,
) -> float:
    """Sum of realized profit on the approved set, using actual outcomes (0/1 default)."""
    paid_mask = approved & (outcomes == 0)
    default_mask = approved & (outcomes == 1)
    return float(interest_if_paid[paid_mask].sum() - loss_if_default[default_mask].sum())
