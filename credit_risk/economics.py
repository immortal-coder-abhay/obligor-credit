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
