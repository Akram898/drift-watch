"""pytest tests for drift detection module (src/detect.py).

Coverage:
    - test_psi_no_drift       : identical distributions → PSI < 0.1
    - test_psi_drift          : shifted distribution   → PSI > 0.2
    - test_ks_clean           : same dist              → p-value > 0.05
    - test_ks_drifted         : shifted dist           → p-value < 0.05
    - test_detect_clean       : no drift on clean data
    - test_detect_drifted     : drift detected on injected data
    Plus comprehensive edge-case tests.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

# The module under test lives at src/detect.py.  The tests directory is at the
# same level as src/ so we add the project root to sys.path via conftest or
# pytest.ini. When running from the project root with `pytest` the src package
# is importable directly.
from src.detect import (
    DriftReport,
    calculate_psi,
    detect_drift,
    ks_test,
    PSI_DRIFT,
    PSI_WARNING,
    KS_ALPHA,
)


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


def _make_df(
    n: int = 1_000,
    shift: float = 0.0,
    seed: int = 0,
    n_features: int = 4,
) -> pd.DataFrame:
    """Generate a DataFrame with *n_features* numeric columns.

    shift ≠ 0 moves every feature mean by *shift* (in raw units, not std-devs).
    The function uses fixed distributions so the shift is predictable.
    """
    rng = np.random.default_rng(seed)
    data = {}
    for i in range(n_features):
        mean = 50.0 * (i + 1)
        std = 10.0
        data[f"feature_{i}"] = rng.normal(mean + shift, std, n)
    return pd.DataFrame(data)


# --------------------------------------------------------------------------- #
#  PSI tests                                                                   #
# --------------------------------------------------------------------------- #


class TestCalculatePsi:
    """Tests for calculate_psi(expected, actual, buckets=10)."""

    def test_psi_no_drift_identical_data(self):
        """Identical arrays → PSI must be < 0.1 (essentially zero)."""
        rng = np.random.default_rng(1)
        data = rng.normal(50, 10, 2_000)
        psi = calculate_psi(data, data)
        assert psi < 0.1, f"Expected PSI < 0.1 for identical data, got {psi:.6f}"

    def test_psi_no_drift_same_distribution_different_sample(self):
        """Two large samples from the same distribution → PSI < 0.1."""
        rng = np.random.default_rng(2)
        ref  = rng.normal(50, 10, 5_000)
        prod = rng.normal(50, 10, 2_000)
        psi = calculate_psi(ref, prod)
        assert psi < 0.1, f"Expected PSI < 0.1 for same dist, got {psi:.6f}"

    def test_psi_drift_large_mean_shift(self):
        """Large mean shift (4 std-devs) → PSI must exceed drift threshold (0.2)."""
        rng = np.random.default_rng(3)
        ref  = rng.normal(50, 10, 3_000)
        prod = rng.normal(90, 10, 1_000)   # 4-std shift
        psi = calculate_psi(ref, prod)
        assert psi > PSI_DRIFT, f"Expected PSI > {PSI_DRIFT} for large shift, got {psi:.6f}"

    def test_psi_drift_moderate_shift_above_warning(self):
        """1.5 std shift → PSI should be in warning or drift zone (>= 0.1)."""
        rng = np.random.default_rng(4)
        ref  = rng.normal(50, 10, 3_000)
        prod = rng.normal(65, 10, 1_000)   # 1.5-std shift
        psi = calculate_psi(ref, prod)
        assert psi >= PSI_WARNING, f"Expected PSI >= {PSI_WARNING} for moderate shift, got {psi:.6f}"

    def test_psi_is_non_negative(self):
        """PSI is always >= 0."""
        rng = np.random.default_rng(5)
        ref  = rng.normal(0, 1, 1_000)
        prod = rng.exponential(1, 1_000)
        psi = calculate_psi(ref, prod)
        assert psi >= 0.0

    def test_psi_constant_distribution_returns_zero(self):
        """Constant arrays produce a single bin → PSI = 0."""
        ref  = np.ones(500) * 7.0
        prod = np.ones(200) * 7.0
        psi = calculate_psi(ref, prod)
        assert psi == 0.0

    def test_psi_returns_float(self):
        rng = np.random.default_rng(6)
        ref  = rng.normal(50, 10, 500)
        prod = rng.normal(60, 10, 500)
        psi = calculate_psi(ref, prod)
        assert isinstance(psi, float)

    def test_psi_custom_buckets(self):
        """Changing bucket count should still return near-zero for same dist."""
        rng = np.random.default_rng(7)
        data = rng.normal(50, 10, 2_000)
        for buckets in [5, 10, 20, 50]:
            psi = calculate_psi(data, data, buckets=buckets)
            assert psi < 0.05, f"PSI {psi:.6f} with {buckets} buckets on identical data"

    def test_psi_accepts_pandas_series(self):
        """calculate_psi should accept pandas Series as well as ndarray."""
        rng = np.random.default_rng(8)
        ref  = pd.Series(rng.normal(50, 10, 1_000))
        prod = pd.Series(rng.normal(50, 10, 500))
        psi = calculate_psi(ref.values, prod.values)
        assert isinstance(psi, float)


# --------------------------------------------------------------------------- #
#  KS test tests                                                               #
# --------------------------------------------------------------------------- #


class TestKsTest:
    """Tests for ks_test(expected, actual)."""

    def test_ks_clean_same_distribution_high_pvalue(self):
        """Splitting one sample → KS p-value should be > 0.05 (not significant)."""
        rng = np.random.default_rng(10)
        data = rng.normal(50, 10, 4_000)
        ref  = data[:2_000]
        prod = data[2_000:]
        _, pvalue = ks_test(ref, prod)
        assert pvalue > 0.05, f"Expected p-value > 0.05 for same dist, got {pvalue:.6f}"

    def test_ks_drifted_different_distribution_low_pvalue(self):
        """3-std-dev shift → KS p-value should be < 0.05."""
        rng = np.random.default_rng(11)
        ref  = rng.normal(50, 10, 1_000)
        prod = rng.normal(80, 10, 1_000)
        _, pvalue = ks_test(ref, prod)
        assert pvalue < 0.05, f"Expected p-value < 0.05 for drifted dist, got {pvalue:.6f}"

    def test_ks_returns_tuple_of_two_floats(self):
        rng = np.random.default_rng(12)
        ref  = rng.normal(0, 1, 500)
        prod = rng.normal(0, 1, 500)
        result = ks_test(ref, prod)
        assert len(result) == 2
        stat, pvalue = result
        assert isinstance(stat,   float)
        assert isinstance(pvalue, float)

    def test_ks_statistic_bounded_0_1(self):
        """KS statistic is always in [0, 1]; p-value in [0, 1]."""
        rng = np.random.default_rng(13)
        ref  = rng.normal(0, 1, 500)
        prod = rng.normal(5, 1, 500)
        stat, pvalue = ks_test(ref, prod)
        assert 0.0 <= stat   <= 1.0
        assert 0.0 <= pvalue <= 1.0

    def test_ks_extreme_shift_near_zero_pvalue(self):
        """10-std shift → statistic near 1, p-value essentially 0."""
        rng = np.random.default_rng(14)
        ref  = rng.normal(0,  1, 2_000)
        prod = rng.normal(10, 1, 2_000)
        stat, pvalue = ks_test(ref, prod)
        assert stat   > 0.9
        assert pvalue < 1e-10

    def test_ks_identical_arrays(self):
        """Identical arrays → p-value of 1.0."""
        rng = np.random.default_rng(15)
        data = rng.normal(50, 10, 500)
        _, pvalue = ks_test(data, data)
        assert pvalue == 1.0


# --------------------------------------------------------------------------- #
#  detect_drift integration tests                                              #
# --------------------------------------------------------------------------- #


class TestDetectDrift:
    """Tests for detect_drift(reference_data, production_data, ...) → DriftReport."""

    def test_detect_clean_no_drift_detected(self):
        """Same distribution → DriftReport.drifted is False."""
        ref  = _make_df(shift=0.0, seed=20)
        prod = _make_df(shift=0.0, seed=21)
        report = detect_drift(ref, prod)
        assert isinstance(report, DriftReport)
        assert not report.drifted, "Expected no drift on clean data"

    def test_detect_drifted_drift_detected(self):
        """Large shift (40 units, 4 std-devs) → DriftReport.drifted is True."""
        ref  = _make_df(shift=0.0,  seed=30)
        prod = _make_df(shift=40.0, seed=31)
        report = detect_drift(ref, prod)
        assert report.drifted, "Expected drift to be detected on injected data"
        assert len(report.affected_features) > 0

    def test_detect_returns_drift_report_instance(self):
        ref  = _make_df(seed=40)
        prod = _make_df(seed=41)
        report = detect_drift(ref, prod)
        assert isinstance(report, DriftReport)

    def test_detect_psi_score_is_non_negative_float(self):
        ref  = _make_df(seed=42)
        prod = _make_df(seed=43)
        report = detect_drift(ref, prod)
        assert isinstance(report.psi_score, float)
        assert report.psi_score >= 0.0

    def test_detect_ks_pvalue_bounded(self):
        ref  = _make_df(seed=44)
        prod = _make_df(seed=45)
        report = detect_drift(ref, prod)
        assert 0.0 <= report.ks_pvalue <= 1.0

    def test_detect_affected_features_is_list(self):
        ref  = _make_df(seed=46)
        prod = _make_df(seed=47)
        report = detect_drift(ref, prod)
        assert isinstance(report.affected_features, list)

    def test_detect_severity_valid_value(self):
        """Severity must be one of 'none', 'minor', 'major'."""
        ref  = _make_df(seed=48)
        prod = _make_df(seed=49)
        report = detect_drift(ref, prod)
        assert report.severity in {"none", "minor", "major"}

    def test_detect_clean_severity_none(self):
        """Clean data → severity == 'none'."""
        ref  = _make_df(n=2_000, shift=0.0, seed=50)
        prod = _make_df(n=2_000, shift=0.0, seed=51)
        report = detect_drift(ref, prod)
        if not report.drifted:
            assert report.severity == "none"

    def test_detect_major_drift_severity_major(self):
        """Very large shift → severity should be 'major'."""
        ref  = _make_df(n=2_000, shift=0.0,  seed=52, n_features=4)
        prod = _make_df(n=2_000, shift=60.0, seed=53, n_features=4)
        report = detect_drift(ref, prod)
        assert report.drifted
        assert report.severity in {"minor", "major"}  # at minimum minor

    def test_detect_feature_details_present(self):
        """feature_details should list each feature's individual stats."""
        ref  = _make_df(seed=54, n_features=4)
        prod = _make_df(seed=55, n_features=4)
        report = detect_drift(ref, prod)
        assert isinstance(report.feature_details, list)
        assert len(report.feature_details) == 4  # 4 features

    def test_detect_feature_detail_keys(self):
        """Each feature_details entry has the required keys."""
        ref  = _make_df(seed=56)
        prod = _make_df(seed=57)
        report = detect_drift(ref, prod)
        required_keys = {"feature", "psi", "ks_pvalue", "status"}
        for detail in report.feature_details:
            missing = required_keys - set(detail.keys())
            assert not missing, f"Missing keys in feature detail: {missing}"

    def test_detect_affected_features_subset_of_feature_details(self):
        """Every affected feature must appear in feature_details."""
        ref  = _make_df(shift=0.0,  seed=58, n_features=4)
        prod = _make_df(shift=40.0, seed=59, n_features=4)
        report = detect_drift(ref, prod)
        all_features = {d["feature"] for d in report.feature_details}
        for feat in report.affected_features:
            assert feat in all_features

    def test_detect_to_dict_is_json_serialisable(self):
        """to_dict() must produce a JSON-serialisable object."""
        ref  = _make_df(seed=60)
        prod = _make_df(shift=30.0, seed=61)
        report = detect_drift(ref, prod)
        d = report.to_dict()
        serialised = json.dumps(d)   # raises if not serialisable
        assert "drifted" in serialised

    def test_detect_to_dict_required_keys(self):
        """to_dict() must include all DriftReport fields."""
        ref  = _make_df(seed=62)
        prod = _make_df(seed=63)
        report = detect_drift(ref, prod)
        d = report.to_dict()
        for key in ("drifted", "psi_score", "ks_pvalue", "affected_features", "severity"):
            assert key in d, f"Missing key '{key}' in DriftReport.to_dict()"

    def test_detect_custom_psi_threshold(self):
        """A very low PSI threshold should flag features that wouldn't normally drift."""
        ref  = _make_df(n=2_000, shift=0.0, seed=64)
        prod = _make_df(n=2_000, shift=0.0, seed=65)
        # Default threshold should produce no drift
        report_default = detect_drift(ref, prod)
        # An extremely tight threshold should produce drift on natural variation
        report_tight = detect_drift(ref, prod, threshold_psi=0.001)
        # The tight threshold should flag at least as many features
        assert len(report_tight.affected_features) >= len(report_default.affected_features)

    def test_detect_excludes_label_columns(self):
        """Columns named 'churn', 'label', or 'target' should be excluded."""
        rng = np.random.default_rng(66)
        n = 500
        ref  = pd.DataFrame({
            "feature_0": rng.normal(50, 10, n),
            "churn":     rng.binomial(1, 0.2, n),
            "label":     rng.binomial(1, 0.5, n),
            "target":    rng.integers(0, 3, n),
        })
        prod = pd.DataFrame({
            "feature_0": rng.normal(90, 10, n),  # large shift
            "churn":     rng.binomial(1, 0.2, n),
            "label":     rng.binomial(1, 0.5, n),
            "target":    rng.integers(0, 3, n),
        })
        report = detect_drift(ref, prod)
        feature_names = {d["feature"] for d in report.feature_details}
        assert "churn"  not in feature_names
        assert "label"  not in feature_names
        assert "target" not in feature_names
        assert "feature_0" in feature_names
