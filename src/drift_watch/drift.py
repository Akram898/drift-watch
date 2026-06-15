"""Statistical drift detection — PSI and Kolmogorov-Smirnov tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
#  Thresholds (PSI industry standard)                                         #
#  PSI < 0.1  → negligible shift                                              #
#  PSI < 0.2  → minor shift (warning)                                         #
#  PSI >= 0.2 → significant shift (drift)                                     #
# --------------------------------------------------------------------------- #

PSI_THRESHOLD_WARNING = 0.1
PSI_THRESHOLD_DRIFT = 0.2
KS_PVALUE_THRESHOLD = 0.05


@dataclass
class FeatureDriftResult:
    """Drift statistics for a single feature."""

    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    status: str  # "stable" | "warning" | "drift"

    @property
    def emoji(self) -> str:
        return {"stable": "✅", "warning": "⚠️ ", "drift": "🚨"}[self.status]

    @property
    def label(self) -> str:
        return {"stable": "STABLE", "warning": "WARNING", "drift": "DRIFT DETECTED"}[self.status]


@dataclass
class DriftReport:
    """Aggregated drift detection report across all features."""

    features: list[FeatureDriftResult] = field(default_factory=list)
    overall_drift_detected: bool = False
    threshold_psi: float = PSI_THRESHOLD_DRIFT
    threshold_ks_pvalue: float = KS_PVALUE_THRESHOLD
    n_reference: int = 0
    n_production: int = 0
    timestamp: Optional[str] = None

    @property
    def drifted_features(self) -> list[FeatureDriftResult]:
        return [f for f in self.features if f.status == "drift"]

    @property
    def warning_features(self) -> list[FeatureDriftResult]:
        return [f for f in self.features if f.status == "warning"]

    def to_dict(self) -> dict:
        return {
            "overall_drift_detected": self.overall_drift_detected,
            "threshold_psi": self.threshold_psi,
            "threshold_ks_pvalue": self.threshold_ks_pvalue,
            "n_reference": self.n_reference,
            "n_production": self.n_production,
            "timestamp": self.timestamp,
            "features": [
                {
                    "feature": r.feature,
                    "psi": r.psi,
                    "ks_statistic": r.ks_statistic,
                    "ks_pvalue": r.ks_pvalue,
                    "status": r.status,
                }
                for r in self.features
            ],
        }


def psi_score(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute the Population Stability Index (PSI) between two distributions.

    PSI = Σ (actual_% - expected_%) × ln(actual_% / expected_%)

    Args:
        expected: Reference (training) distribution, 1-D array.
        actual:   Production distribution, 1-D array.
        bins:     Number of equal-width bins computed from *expected*.

    Returns:
        PSI score (non-negative float).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Build bin edges from the reference distribution
    breakpoints = np.nanpercentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # deduplicate flat regions

    if len(breakpoints) < 2:
        # All values identical — no drift possible
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    # Convert to proportions, guard against zero bins
    eps = 1e-4
    expected_pct = (expected_counts / expected_counts.sum()).clip(eps)
    actual_pct = (actual_counts / actual_counts.sum()).clip(eps)

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return round(abs(psi), 6)


def ks_test(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Run a two-sample Kolmogorov-Smirnov test.

    Returns:
        (statistic, pvalue) — lower p-value → more likely distributions differ.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    result = stats.ks_2samp(expected, actual)
    return round(float(result.statistic), 6), round(float(result.pvalue), 6)


def _classify_feature(psi: float, ks_pvalue: float) -> str:
    """Assign a status string from PSI + KS evidence."""
    if psi >= PSI_THRESHOLD_DRIFT or ks_pvalue < KS_PVALUE_THRESHOLD:
        return "drift"
    if psi >= PSI_THRESHOLD_WARNING:
        return "warning"
    return "stable"


def detect_drift(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    threshold: float = PSI_THRESHOLD_DRIFT,
    feature_columns: list[str] | None = None,
) -> DriftReport:
    """Run per-feature drift detection between reference and production data.

    Args:
        reference_df:   DataFrame of training / reference samples.
        production_df:  DataFrame of live / production samples.
        threshold:      PSI threshold above which drift is flagged (default 0.2).
        feature_columns: Subset of columns to analyse. Defaults to all numeric columns.

    Returns:
        DriftReport with per-feature results and an overall verdict.
    """
    import datetime

    if feature_columns is None:
        feature_columns = [
            c
            for c in reference_df.select_dtypes(include="number").columns
            if c not in {"churn", "label", "target"}
        ]

    results: list[FeatureDriftResult] = []
    for col in feature_columns:
        if col not in production_df.columns:
            continue
        ref_vals = reference_df[col].dropna().values
        prod_vals = production_df[col].dropna().values

        psi = psi_score(ref_vals, prod_vals)
        ks_stat, ks_pval = ks_test(ref_vals, prod_vals)
        status = _classify_feature(psi, ks_pval)

        results.append(
            FeatureDriftResult(
                feature=col,
                psi=psi,
                ks_statistic=ks_stat,
                ks_pvalue=ks_pval,
                status=status,
            )
        )

    overall_drift = any(r.status == "drift" for r in results)

    return DriftReport(
        features=results,
        overall_drift_detected=overall_drift,
        threshold_psi=threshold,
        threshold_ks_pvalue=KS_PVALUE_THRESHOLD,
        n_reference=len(reference_df),
        n_production=len(production_df),
        timestamp=datetime.datetime.utcnow().isoformat(),
    )
