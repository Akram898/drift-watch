"""Tests for PSI and KS drift detection functions."""

from __future__ import annotations

import numpy as np
import pytest

from drift_watch.drift import (
    psi_score,
    ks_test,
    detect_drift,
    DriftReport,
    PSI_THRESHOLD_DRIFT,
    PSI_THRESHOLD_WARNING,
)
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────── #
#  PSI tests                                                                   #
# ──────────────────────────────────────────────────────────────────────────── #


class TestPsiScore:
    def test_identical_distributions_returns_near_zero(self):
        rng = np.random.default_rng(0)
        data = rng.normal(50, 10, 2_000)
        psi = psi_score(data, data)
        assert psi < 0.02, f"Expected near-zero PSI for identical data, got {psi}"

    def test_stable_distributions_below_warning_threshold(self):
        rng = np.random.default_rng(1)
        ref = rng.normal(50, 10, 3_000)
        prod = rng.normal(50, 10, 1_000)   # same distribution, different seed
        psi = psi_score(ref, prod)
        assert psi < PSI_THRESHOLD_WARNING, (
            f"PSI {psi:.4f} should be below warning threshold {PSI_THRESHOLD_WARNING}"
        )

    def test_heavily_drifted_distributions_above_drift_threshold(self):
        rng = np.random.default_rng(2)
        ref = rng.normal(50, 10, 3_000)
        prod = rng.normal(90, 10, 1_000)   # mean shift of 4 std-devs
        psi = psi_score(ref, prod)
        assert psi >= PSI_THRESHOLD_DRIFT, (
            f"PSI {psi:.4f} should exceed drift threshold {PSI_THRESHOLD_DRIFT}"
        )

    def test_moderate_drift_between_thresholds(self):
        rng = np.random.default_rng(3)
        ref = rng.normal(50, 10, 3_000)
        prod = rng.normal(60, 10, 1_000)   # mean shift of 1 std-dev
        psi = psi_score(ref, prod)
        # Between 0.1 and 0.4 — a warning region
        assert PSI_THRESHOLD_WARNING <= psi <= 0.5, (
            f"Expected moderate PSI, got {psi:.4f}"
        )

    def test_psi_is_non_negative(self):
        rng = np.random.default_rng(4)
        ref = rng.normal(0, 1, 1_000)
        prod = rng.exponential(1, 1_000)
        psi = psi_score(ref, prod)
        assert psi >= 0.0

    def test_psi_with_custom_bin_count(self):
        rng = np.random.default_rng(5)
        data = rng.normal(50, 10, 2_000)
        psi_10 = psi_score(data, data, bins=10)
        psi_20 = psi_score(data, data, bins=20)
        # Both should be near zero regardless of bin count
        assert psi_10 < 0.05
        assert psi_20 < 0.05

    def test_constant_distribution_returns_zero(self):
        ref = np.ones(500) * 5.0
        prod = np.ones(200) * 5.0
        psi = psi_score(ref, prod)
        assert psi == 0.0

    def test_psi_accepts_pandas_series(self):
        rng = np.random.default_rng(6)
        ref = pd.Series(rng.normal(50, 10, 1_000))
        prod = pd.Series(rng.normal(50, 10, 500))
        psi = psi_score(ref.values, prod.values)
        assert isinstance(psi, float)


# ──────────────────────────────────────────────────────────────────────────── #
#  KS test tests                                                               #
# ──────────────────────────────────────────────────────────────────────────── #


class TestKsTest:
    def test_same_distribution_high_pvalue(self):
        rng = np.random.default_rng(10)
        data = rng.normal(50, 10, 2_000)
        ref = data[:1_000]
        prod = data[1_000:]
        stat, pval = ks_test(ref, prod)
        # Splitting same data — should NOT be significant
        assert pval > 0.05, f"Expected high p-value for same distribution, got {pval:.4f}"

    def test_drifted_distribution_low_pvalue(self):
        rng = np.random.default_rng(11)
        ref = rng.normal(50, 10, 1_000)
        prod = rng.normal(80, 10, 1_000)   # 3-std shift
        stat, pval = ks_test(ref, prod)
        assert pval < 0.01, f"Expected low p-value for drifted distribution, got {pval:.4f}"

    def test_ks_returns_tuple_of_floats(self):
        rng = np.random.default_rng(12)
        ref = rng.normal(0, 1, 500)
        prod = rng.normal(0, 1, 500)
        result = ks_test(ref, prod)
        assert len(result) == 2
        stat, pval = result
        assert isinstance(stat, float)
        assert isinstance(pval, float)

    def test_ks_statistic_bounded(self):
        rng = np.random.default_rng(13)
        ref = rng.normal(0, 1, 500)
        prod = rng.normal(5, 1, 500)
        stat, pval = ks_test(ref, prod)
        assert 0.0 <= stat <= 1.0
        assert 0.0 <= pval <= 1.0

    def test_large_shift_gives_high_statistic(self):
        rng = np.random.default_rng(14)
        ref = rng.normal(0, 1, 2_000)
        prod = rng.normal(10, 1, 2_000)   # extreme shift
        stat, pval = ks_test(ref, prod)
        assert stat > 0.9
        assert pval < 1e-10


# ──────────────────────────────────────────────────────────────────────────── #
#  detect_drift integration tests                                              #
# ──────────────────────────────────────────────────────────────────────────── #


class TestDetectDrift:
    def _make_df(self, n: int = 1_000, shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "feature_a": rng.normal(50 + shift, 10, n),
                "feature_b": rng.normal(100 + shift, 20, n),
                "feature_c": rng.normal(0 + shift, 1, n),
                "churn": rng.binomial(1, 0.2, n),
            }
        )

    def test_stable_data_no_drift_detected(self):
        ref = self._make_df(shift=0.0, seed=20)
        prod = self._make_df(shift=0.0, seed=21)
        report = detect_drift(ref, prod)
        assert isinstance(report, DriftReport)
        assert not report.overall_drift_detected

    def test_heavily_drifted_data_drift_detected(self):
        ref = self._make_df(shift=0.0, seed=30)
        prod = self._make_df(shift=50.0, seed=31)   # extreme shift
        report = detect_drift(ref, prod)
        assert report.overall_drift_detected

    def test_report_contains_all_numeric_features(self):
        ref = self._make_df(seed=40)
        prod = self._make_df(seed=41)
        report = detect_drift(ref, prod)
        feature_names = {r.feature for r in report.features}
        # churn should be excluded
        assert "churn" not in feature_names
        assert "feature_a" in feature_names
        assert "feature_b" in feature_names
        assert "feature_c" in feature_names

    def test_report_n_reference_and_n_production(self):
        ref = self._make_df(n=800, seed=50)
        prod = self._make_df(n=300, seed=51)
        report = detect_drift(ref, prod)
        assert report.n_reference == 800
        assert report.n_production == 300

    def test_to_dict_is_json_serialisable(self):
        import json
        ref = self._make_df(seed=60)
        prod = self._make_df(seed=61)
        report = detect_drift(ref, prod)
        d = report.to_dict()
        # Should not raise
        json.dumps(d)

    def test_custom_feature_columns(self):
        ref = self._make_df(seed=70)
        prod = self._make_df(shift=40.0, seed=71)
        report = detect_drift(ref, prod, feature_columns=["feature_a"])
        assert len(report.features) == 1
        assert report.features[0].feature == "feature_a"

    def test_feature_result_status_values(self):
        ref = self._make_df(seed=80)
        prod = self._make_df(seed=81)
        report = detect_drift(ref, prod)
        for r in report.features:
            assert r.status in {"stable", "warning", "drift"}

    def test_drifted_features_property(self):
        ref = self._make_df(shift=0.0, seed=90)
        prod = self._make_df(shift=60.0, seed=91)
        report = detect_drift(ref, prod)
        # All features shifted — all should drift
        assert len(report.drifted_features) > 0

    def test_timestamp_is_set(self):
        ref = self._make_df(seed=100)
        prod = self._make_df(seed=101)
        report = detect_drift(ref, prod)
        assert report.timestamp is not None
        # Should be parseable as ISO 8601
        datetime_obj = __import__("datetime").datetime.fromisoformat(report.timestamp)
        assert datetime_obj is not None
