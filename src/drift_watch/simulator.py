"""Production traffic simulator with configurable drift injection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from drift_watch.model import (
    FEATURE_NAMES,
    FEATURE_MEANS,
    FEATURE_STDS,
    generate_training_data,
)


def simulate_production_traffic(
    n_samples: int = 1_000,
    drift_factor: float = 2.0,
    drift_fraction: float = 0.3,
    seed: int = 99,
) -> pd.DataFrame:
    """Generate realistic production samples with injected covariate shift.

    Parameters
    ----------
    n_samples:
        Total number of production samples to generate.
    drift_factor:
        Number of standard deviations by which drifted features are shifted.
        Default 2.0 — noticeable but not extreme.
    drift_fraction:
        Fraction of features to inject drift into (0 → no drift, 1 → all).
        Default 0.3 — roughly one third of features.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with the same schema as the training data (no churn column —
    production data typically lacks labels at inference time).
    """
    rng = np.random.default_rng(seed)

    # Start from the baseline distribution (no labels needed)
    df = generate_training_data(n_samples=n_samples, seed=seed).drop(columns=["churn"])

    # Randomly choose which features to drift
    n_drifted = max(1, int(len(FEATURE_NAMES) * drift_fraction))
    drifted_features = rng.choice(FEATURE_NAMES, size=n_drifted, replace=False).tolist()

    for feat in drifted_features:
        std = FEATURE_STDS[feat]
        shift = drift_factor * std
        df[feat] = df[feat] + shift

    # Clip binary features back to [0, 1]
    for binary_col in ["has_credit_card", "is_active_member", "geography_germany", "geography_spain"]:
        if binary_col in df.columns:
            df[binary_col] = df[binary_col].clip(0, 1).round()

    # Clip obviously impossible values
    if "credit_score" in df.columns:
        df["credit_score"] = df["credit_score"].clip(300, 900)
    if "age" in df.columns:
        df["age"] = df["age"].clip(18, 95)
    if "balance" in df.columns:
        df["balance"] = df["balance"].clip(0)
    if "estimated_salary" in df.columns:
        df["estimated_salary"] = df["estimated_salary"].clip(0)

    return df


def simulate_stable_traffic(n_samples: int = 1_000, seed: int = 77) -> pd.DataFrame:
    """Generate production samples with NO injected drift — for baseline tests."""
    return generate_training_data(n_samples=n_samples, seed=seed).drop(columns=["churn"])
