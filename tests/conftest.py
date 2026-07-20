"""
tests/conftest.py
─────────────────────────────────────────────────────────────────────────────
Shared pytest fixtures and configuration for the FraudGuard test suite.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure the project root is on sys.path so `src.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Tiny synthetic fixtures (no real data needed) ─────────────────────────


@pytest.fixture(scope="session")
def synthetic_ieee_df() -> pd.DataFrame:
    """
    A tiny (~20 row) synthetic DataFrame that mimics the IEEE-CIS merged schema.
    Used for unit tests that must not require the real downloaded data.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    # Use 100 rows with 10 fraud cases (10% rate) so that after a 20% test
    # split (~80 train_val rows) there are still ≥2 fraud samples available
    # for the stratified val split. This mirrors real data proportions for
    # unit-testing purposes.
    n = 100
    fraud_flags = [0] * 90 + [1] * 10
    rng.shuffle(fraud_flags)  # shuffle so fraud is spread across TransactionDT

    return pd.DataFrame(
        {
            "TransactionID": range(3000000, 3000000 + n),
            "isFraud": fraud_flags,
            # Evenly spaced so temporal ordering is deterministic
            "TransactionDT": list(range(86400, 86400 + n * 10000, 10000)),
            "TransactionAmt": rng.uniform(1.0, 1000.0, size=n).round(2),
            "ProductCD": rng.choice(["W", "H", "C", "S", "R"], size=n),
            "card1": rng.integers(1000, 9999, size=n),
            "card2": rng.integers(100, 600, size=n).astype(float),
            "card3": rng.integers(100, 200, size=n).astype(float),
            "card4": rng.choice(["visa", "mastercard", "discover"], size=n),
            "card5": rng.integers(100, 250, size=n).astype(float),
            "card6": rng.choice(["credit", "debit"], size=n),
            "addr1": rng.integers(100, 500, size=n).astype(float),
            "addr2": rng.integers(10, 100, size=n).astype(float),
            "dist1": rng.uniform(0, 500, size=n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], size=n),
            "R_emaildomain": rng.choice(["gmail.com", "hotmail.com", None], size=n),
        }
    )


@pytest.fixture(scope="session")
def synthetic_ulb_df() -> pd.DataFrame:
    """Tiny synthetic ULB-schema DataFrame for unit tests."""
    import numpy as np

    rng = np.random.default_rng(42)
    n = 100
    data = {"Time": rng.integers(0, 172800, size=n).astype(float)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.standard_normal(size=n)
    data["Amount"] = rng.uniform(0.0, 500.0, size=n).round(2)
    data["Class"] = [0] * 90 + [1] * 10
    return pd.DataFrame(data)
