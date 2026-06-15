"""Tests for data generation, model training, and save/load roundtrip."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drift_watch.model import (
    generate_training_data,
    train_model,
    save_model,
    load_model,
    FEATURE_NAMES,
)


# ──────────────────────────────────────────────────────────────────────────── #
#  generate_training_data                                                      #
# ──────────────────────────────────────────────────────────────────────────── #


class TestGenerateTrainingData:
    def test_returns_dataframe(self):
        df = generate_training_data(n_samples=100)
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self):
        df = generate_training_data(n_samples=500)
        assert len(df) == 500

    def test_has_all_feature_columns(self):
        df = generate_training_data(n_samples=100)
        for col in FEATURE_NAMES:
            assert col in df.columns, f"Missing column: {col}"

    def test_has_churn_column(self):
        df = generate_training_data(n_samples=100)
        assert "churn" in df.columns

    def test_churn_is_binary(self):
        df = generate_training_data(n_samples=500)
        assert set(df["churn"].unique()).issubset({0, 1})

    def test_churn_rate_in_expected_range(self):
        df = generate_training_data(n_samples=5_000)
        rate = df["churn"].mean()
        # Expect ~15–35% churn given the logistic rule
        assert 0.10 <= rate <= 0.45, f"Churn rate {rate:.2%} outside expected range"

    def test_no_null_values(self):
        df = generate_training_data(n_samples=500)
        assert df.isnull().sum().sum() == 0

    def test_credit_score_bounds(self):
        df = generate_training_data(n_samples=2_000)
        assert df["credit_score"].min() >= 300
        assert df["credit_score"].max() <= 900

    def test_age_bounds(self):
        df = generate_training_data(n_samples=2_000)
        assert df["age"].min() >= 18
        assert df["age"].max() <= 80

    def test_balance_non_negative(self):
        df = generate_training_data(n_samples=2_000)
        assert (df["balance"] >= 0).all()

    def test_binary_features_in_zero_one(self):
        df = generate_training_data(n_samples=2_000)
        for col in ["has_credit_card", "is_active_member", "geography_germany", "geography_spain"]:
            assert set(df[col].unique()).issubset({0.0, 1.0}), (
                f"{col} has values outside {{0, 1}}"
            )

    def test_reproducibility_with_same_seed(self):
        df1 = generate_training_data(n_samples=100, seed=42)
        df2 = generate_training_data(n_samples=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        df1 = generate_training_data(n_samples=100, seed=1)
        df2 = generate_training_data(n_samples=100, seed=2)
        assert not df1["age"].equals(df2["age"])

    def test_default_sample_count_is_5000(self):
        df = generate_training_data()
        assert len(df) == 5_000


# ──────────────────────────────────────────────────────────────────────────── #
#  train_model                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #


class TestTrainModel:
    @pytest.fixture(scope="class")
    def trained(self):
        df = generate_training_data(n_samples=1_000, seed=42)
        X = df.drop(columns=["churn"])
        y = df["churn"]
        clf, scaler, metrics = train_model(X, y, n_estimators=50, seed=42)
        return clf, scaler, metrics

    def test_returns_three_items(self, trained):
        assert len(trained) == 3

    def test_model_has_predict_method(self, trained):
        clf, _, _ = trained
        assert hasattr(clf, "predict")
        assert hasattr(clf, "predict_proba")

    def test_metrics_contains_accuracy(self, trained):
        _, _, metrics = trained
        assert "accuracy" in metrics
        assert isinstance(metrics["accuracy"], float)

    def test_metrics_contains_roc_auc(self, trained):
        _, _, metrics = trained
        assert "roc_auc" in metrics
        assert isinstance(metrics["roc_auc"], float)

    def test_accuracy_above_random(self, trained):
        _, _, metrics = trained
        # Should easily beat 0.5 on this dataset
        assert metrics["accuracy"] > 0.55, (
            f"Accuracy {metrics['accuracy']:.4f} barely above random"
        )

    def test_roc_auc_above_random(self, trained):
        _, _, metrics = trained
        assert metrics["roc_auc"] > 0.6, (
            f"ROC-AUC {metrics['roc_auc']:.4f} too low"
        )

    def test_metrics_n_train_and_n_val(self, trained):
        _, _, metrics = trained
        assert "n_train" in metrics
        assert "n_val" in metrics
        assert metrics["n_train"] + metrics["n_val"] == 1_000

    def test_scaler_has_mean_and_scale(self, trained):
        _, scaler, _ = trained
        assert hasattr(scaler, "mean_")
        assert hasattr(scaler, "scale_")

    def test_model_predicts_binary(self, trained):
        clf, scaler, _ = trained
        df = generate_training_data(n_samples=10, seed=99)
        X = df.drop(columns=["churn"])
        X_scaled = scaler.transform(X)
        preds = clf.predict(X_scaled)
        assert set(preds).issubset({0, 1})

    def test_model_outputs_probabilities_between_zero_and_one(self, trained):
        clf, scaler, _ = trained
        df = generate_training_data(n_samples=10, seed=88)
        X = df.drop(columns=["churn"])
        X_scaled = scaler.transform(X)
        proba = clf.predict_proba(X_scaled)
        assert proba.shape[1] == 2
        assert (proba >= 0).all()
        assert (proba <= 1).all()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ──────────────────────────────────────────────────────────────────────────── #
#  save_model / load_model roundtrip                                           #
# ──────────────────────────────────────────────────────────────────────────── #


class TestSaveLoadRoundtrip:
    @pytest.fixture(scope="class")
    def model_and_path(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("models")
        df = generate_training_data(n_samples=500, seed=7)
        X = df.drop(columns=["churn"])
        y = df["churn"]
        clf, scaler, _ = train_model(X, y, n_estimators=20, seed=7)
        path = tmp / "test_model.pkl"
        save_model(clf, scaler, path)
        return clf, scaler, path

    def test_file_is_created(self, model_and_path):
        _, _, path = model_and_path
        assert path.exists()

    def test_file_is_non_empty(self, model_and_path):
        _, _, path = model_and_path
        assert path.stat().st_size > 0

    def test_loaded_model_is_same_type(self, model_and_path):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        _, _, path = model_and_path
        clf_loaded, scaler_loaded = load_model(path)
        assert isinstance(clf_loaded, RandomForestClassifier)
        assert isinstance(scaler_loaded, StandardScaler)

    def test_loaded_model_same_predictions(self, model_and_path):
        clf_orig, scaler_orig, path = model_and_path
        clf_loaded, scaler_loaded = load_model(path)

        df = generate_training_data(n_samples=50, seed=123)
        X = df.drop(columns=["churn"]).values
        X_s_orig = scaler_orig.transform(X)
        X_s_load = scaler_loaded.transform(X)

        preds_orig = clf_orig.predict(X_s_orig)
        preds_load = clf_loaded.predict(X_s_load)
        np.testing.assert_array_equal(preds_orig, preds_load)

    def test_load_raises_for_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "does_not_exist.pkl")

    def test_save_creates_parent_directories(self, tmp_path):
        df = generate_training_data(n_samples=200, seed=5)
        X = df.drop(columns=["churn"])
        y = df["churn"]
        clf, scaler, _ = train_model(X, y, n_estimators=10, seed=5)
        nested = tmp_path / "a" / "b" / "c" / "model.pkl"
        save_model(clf, scaler, nested)   # should not raise
        assert nested.exists()
