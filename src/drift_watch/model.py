"""Model training, serialisation, and data-generation helpers."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
#  Feature schema — keep in sync with simulator.py                            #
# --------------------------------------------------------------------------- #

FEATURE_NAMES = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "num_products",
    "has_credit_card",
    "is_active_member",
    "estimated_salary",
    "geography_germany",
    "geography_spain",
]

FEATURE_MEANS = {
    "credit_score": 650.0,
    "age": 38.0,
    "tenure": 5.0,
    "balance": 76_000.0,
    "num_products": 1.5,
    "has_credit_card": 0.7,
    "is_active_member": 0.5,
    "estimated_salary": 100_000.0,
    "geography_germany": 0.25,
    "geography_spain": 0.25,
}

FEATURE_STDS = {
    "credit_score": 100.0,
    "age": 10.0,
    "tenure": 2.5,
    "balance": 62_000.0,
    "num_products": 0.6,
    "has_credit_card": 0.46,
    "is_active_member": 0.5,
    "estimated_salary": 57_000.0,
    "geography_germany": 0.43,
    "geography_spain": 0.43,
}


def generate_training_data(n_samples: int = 5_000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic bank-churn dataset.

    Returns a DataFrame with FEATURE_NAMES columns plus a binary ``churn`` target.
    Churn probability depends on age, balance, and num_products — realistic enough
    to train a useful classifier.
    """
    rng = np.random.default_rng(seed)

    credit_score = rng.normal(650, 100, n_samples).clip(300, 850)
    age = rng.normal(38, 10, n_samples).clip(18, 80)
    tenure = rng.integers(0, 11, n_samples).astype(float)
    balance = rng.normal(76_000, 62_000, n_samples).clip(0)
    num_products = rng.choice([1, 2, 3, 4], n_samples, p=[0.5, 0.35, 0.1, 0.05]).astype(float)
    has_credit_card = rng.binomial(1, 0.7, n_samples).astype(float)
    is_active_member = rng.binomial(1, 0.5, n_samples).astype(float)
    estimated_salary = rng.normal(100_000, 57_000, n_samples).clip(0)
    geography = rng.choice(["france", "germany", "spain"], n_samples, p=[0.5, 0.25, 0.25])
    geography_germany = (geography == "germany").astype(float)
    geography_spain = (geography == "spain").astype(float)

    # Logistic churn rule — weighted combination of features
    log_odds = (
        -3.0
        + 0.02 * (age - 38)
        + 0.8 * (num_products - 1)
        + -0.01 * (balance / 10_000)
        + -0.5 * is_active_member
        + 0.3 * geography_germany
    )
    churn_prob = 1 / (1 + np.exp(-log_odds))
    churn = rng.binomial(1, churn_prob, n_samples)

    df = pd.DataFrame(
        {
            "credit_score": credit_score,
            "age": age,
            "tenure": tenure,
            "balance": balance,
            "num_products": num_products,
            "has_credit_card": has_credit_card,
            "is_active_member": is_active_member,
            "estimated_salary": estimated_salary,
            "geography_germany": geography_germany,
            "geography_spain": geography_spain,
            "churn": churn,
        }
    )
    return df


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_estimators: int = 200,
    max_depth: int = 8,
    seed: int = 42,
) -> tuple[RandomForestClassifier, StandardScaler, dict]:
    """Train a RandomForestClassifier and return (model, scaler, metrics)."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_val_scaled)
    y_proba = clf.predict_proba(X_val_scaled)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_val, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_val, y_proba)), 4),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "churn_rate": round(float(y.mean()), 4),
    }
    return clf, scaler, metrics


def save_model(clf: RandomForestClassifier, scaler: StandardScaler, path: str | Path) -> None:
    """Pickle (model, scaler) bundle to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump({"model": clf, "scaler": scaler}, fh, protocol=5)


def load_model(path: str | Path) -> tuple[RandomForestClassifier, StandardScaler]:
    """Load a (model, scaler) bundle from *path*."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as fh:
        bundle = pickle.load(fh)
    return bundle["model"], bundle["scaler"]
