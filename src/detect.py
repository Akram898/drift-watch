"""Drift detection module: PSI and Kolmogorov-Smirnov tests.

Public API:
    calculate_psi(expected, actual, buckets=10) -> float
    ks_test(expected, actual)                   -> (statistic, pvalue)
    detect_drift(reference_data, production_data, ...) -> DriftReport
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import stats


# --------------------------------------------------------------------------- #
#  Constants                                                                   #
# --------------------------------------------------------------------------- #

#  PSI industry-standard thresholds
#  PSI < 0.10  → negligible / stable
#  PSI < 0.20  → minor shift (warning)
#  PSI >= 0.20 → significant drift
PSI_WARNING = 0.10
PSI_DRIFT = 0.20
KS_ALPHA = 0.05  # significance level for KS test


# --------------------------------------------------------------------------- #
#  DriftReport dataclass                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class DriftReport:
    """Aggregated drift detection result returned by ``detect_drift``."""

    drifted: bool
    """True when at least one feature exceeds drift thresholds."""

    psi_score: float
    """Mean PSI score across all analysed features."""

    ks_pvalue: float
    """Minimum KS p-value across all analysed features (most significant)."""

    affected_features: list[str] = field(default_factory=list)
    """Names of features where drift was detected."""

    severity: str = "none"
    """Overall severity: 'none' | 'minor' | 'major'."""

    feature_details: list[dict] = field(default_factory=list)
    """Per-feature breakdown: [{feature, psi, ks_stat, ks_pvalue, status}, ...]"""

    def to_dict(self) -> dict:
        return {
            "drifted": self.drifted,
            "psi_score": self.psi_score,
            "ks_pvalue": self.ks_pvalue,
            "affected_features": self.affected_features,
            "severity": self.severity,
            "feature_details": self.feature_details,
        }


# --------------------------------------------------------------------------- #
#  PSI                                                                         #
# --------------------------------------------------------------------------- #


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10,
) -> float:
    """Compute the Population Stability Index (PSI).

    PSI = sum( (actual_pct - expected_pct) * ln(actual_pct / expected_pct) )

    PSI < 0.1  → stable
    PSI < 0.2  → minor shift
    PSI >= 0.2 → significant drift

    Args:
        expected: Reference (training-time) distribution as a 1-D array-like.
        actual:   Production distribution as a 1-D array-like.
        buckets:  Number of quantile-equal bins derived from *expected*.

    Returns:
        Non-negative PSI score (float).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Derive bin edges from quantiles of the reference distribution
    percentiles = np.linspace(0, 100, buckets + 1)
    breakpoints = np.nanpercentile(expected, percentiles)
    breakpoints = np.unique(breakpoints)  # remove duplicates from flat regions

    if len(breakpoints) < 2:
        # All reference values are identical — distribution cannot change
        return 0.0

    # Count samples in each bin
    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    # Convert to proportions; clip to avoid log(0) / division by zero
    eps = 1e-4
    expected_pct = (expected_counts / expected_counts.sum()).clip(eps)
    actual_pct = (actual_counts / actual_counts.sum()).clip(eps)

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return round(abs(psi), 6)


# --------------------------------------------------------------------------- #
#  KS test                                                                     #
# --------------------------------------------------------------------------- #


def ks_test(
    expected: np.ndarray,
    actual: np.ndarray,
) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Args:
        expected: Reference distribution (1-D array-like).
        actual:   Production distribution (1-D array-like).

    Returns:
        (statistic, pvalue) — low p-value indicates the distributions differ.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    result = stats.ks_2samp(expected, actual)
    return round(float(result.statistic), 6), round(float(result.pvalue), 6)


# --------------------------------------------------------------------------- #
#  High-level detect_drift                                                     #
# --------------------------------------------------------------------------- #


def detect_drift(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame,
    threshold_psi: float = PSI_DRIFT,
    threshold_ks: float = KS_ALPHA,
) -> DriftReport:
    """Run per-feature PSI + KS drift detection.

    Args:
        reference_data:  DataFrame of training / reference samples.
        production_data: DataFrame of production samples.
        threshold_psi:   PSI above which a feature is flagged (default 0.2).
        threshold_ks:    KS p-value below which a feature is flagged (default 0.05).

    Returns:
        DriftReport with overall verdict and per-feature details.
    """
    # Only analyse numeric columns; skip obvious label columns
    _SKIP = {"churn", "label", "target", "class"}
    feature_cols = [
        c
        for c in reference_data.select_dtypes(include="number").columns
        if c.lower() not in _SKIP and c in production_data.columns
    ]

    feature_details: list[dict] = []
    affected: list[str] = []
    psi_scores: list[float] = []
    ks_pvalues: list[float] = []

    for col in feature_cols:
        ref_vals = reference_data[col].dropna().to_numpy(dtype=float)
        prod_vals = production_data[col].dropna().to_numpy(dtype=float)

        psi = calculate_psi(ref_vals, prod_vals)
        ks_stat, ks_pval = ks_test(ref_vals, prod_vals)

        psi_scores.append(psi)
        ks_pvalues.append(ks_pval)

        if psi >= threshold_psi or ks_pval < threshold_ks:
            status = "drift"
            affected.append(col)
        elif psi >= PSI_WARNING:
            status = "warning"
        else:
            status = "stable"

        feature_details.append(
            {
                "feature": col,
                "psi": psi,
                "ks_stat": ks_stat,
                "ks_pvalue": ks_pval,
                "status": status,
            }
        )

    mean_psi = round(float(np.mean(psi_scores)), 6) if psi_scores else 0.0
    min_ks_pval = round(float(np.min(ks_pvalues)), 6) if ks_pvalues else 1.0
    drifted = bool(affected)

    # Severity classification
    if not drifted:
        severity = "none"
    elif mean_psi >= 0.25 or len(affected) > len(feature_cols) // 2:
        severity = "major"
    else:
        severity = "minor"

    return DriftReport(
        drifted=drifted,
        psi_score=mean_psi,
        ks_pvalue=min_ks_pval,
        affected_features=affected,
        severity=severity,
        feature_details=feature_details,
    )
