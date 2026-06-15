"""Train a RandomForestClassifier on the Iris dataset and register it.

Usage:
    python src/train.py             # trains version 1
    python src/train.py --version 2 # trains and registers as version 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Paths are relative to project root; resolve from this file's location.
_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
MODELS_DIR = _ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"


# --------------------------------------------------------------------------- #
#  Training                                                                    #
# --------------------------------------------------------------------------- #


def train(version: int = 1, random_state: int = 42) -> dict:
    """Train RandomForestClassifier on Iris, persist model, update registry.

    Args:
        version:      Integer version tag (used in the file name).
        random_state: Seed for reproducibility.

    Returns:
        Registry entry dict: { version, path, metrics, timestamp }.
    """
    # 1. Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # 2. Train
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # 3. Evaluate
    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_names": list(iris.feature_names),
        "target_names": list(iris.target_names),
    }

    # 4. Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"model_v{version}.pkl"
    joblib.dump(clf, model_path)

    # 5. Register
    from src.registry import register_model  # local import avoids circular deps at module level

    entry = register_model(version=version, metrics=metrics, path=str(model_path))

    print(f"[drift-watch] Trained model v{version}")
    print(f"  accuracy   : {metrics['accuracy']:.4f}")
    print(f"  f1 weighted: {metrics['f1_weighted']:.4f}")
    print(f"  saved to   : {model_path}")

    return entry


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train",
        description="Train drift-watch Iris classifier and register the model.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Integer version tag for the model file (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    train(version=args.version, random_state=args.seed)


if __name__ == "__main__":
    main()
