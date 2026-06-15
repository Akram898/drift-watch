"""Drift simulation — generates Iris-like data and injects configurable drift.

Public API:
    generate_reference_data(n=1000) -> pd.DataFrame
    inject_drift(data, severity=0.3) -> pd.DataFrame
    run_simulation(n_rounds=5) -> list[dict]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent

# Feature names used throughout the simulation
IRIS_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


# --------------------------------------------------------------------------- #
#  Reference data generation                                                   #
# --------------------------------------------------------------------------- #


def generate_reference_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate clean Iris-like reference data.

    Samples are drawn from Gaussian distributions whose mean and std are
    estimated from the real Iris dataset, then scaled to *n* samples.

    Args:
        n:    Number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns sepal_length, sepal_width, petal_length,
        petal_width (the 4 Iris numeric features).
    """
    rng = np.random.default_rng(seed)

    # Real Iris statistics
    iris = load_iris()
    X_real = iris.data  # shape (150, 4)
    means = X_real.mean(axis=0)
    stds = X_real.std(axis=0)

    data = {}
    for i, name in enumerate(IRIS_FEATURES):
        data[name] = rng.normal(loc=means[i], scale=stds[i], size=n)

    df = pd.DataFrame(data)
    # Clip to realistic bounds
    df["sepal_length"] = df["sepal_length"].clip(4.0, 8.0)
    df["sepal_width"] = df["sepal_width"].clip(1.5, 5.0)
    df["petal_length"] = df["petal_length"].clip(0.5, 7.5)
    df["petal_width"] = df["petal_width"].clip(0.0, 3.0)

    return df


# --------------------------------------------------------------------------- #
#  Drift injection                                                              #
# --------------------------------------------------------------------------- #


def inject_drift(
    data: pd.DataFrame,
    severity: float = 0.3,
    seed: int = 99,
) -> pd.DataFrame:
    """Inject Gaussian noise drift into a feature DataFrame.

    Drift is simulated by adding noise scaled by *severity* times the
    feature's own standard deviation, which keeps the shift proportional
    to the natural spread of each feature.

    Args:
        data:     Input DataFrame (numeric features only).
        severity: Drift magnitude as a fraction of feature std.
                  0.0 = no drift, 1.0 = shift by 1 full std-dev.
                  Values above 0.2 are generally detectable by PSI/KS.
        seed:     Random seed.

    Returns:
        New DataFrame with drifted feature values.
    """
    rng = np.random.default_rng(seed)
    drifted = data.copy()

    for col in drifted.select_dtypes(include="number").columns:
        col_std = float(drifted[col].std())
        noise = rng.normal(loc=0.0, scale=severity * col_std, size=len(drifted))
        drifted[col] = drifted[col] + noise

    return drifted


# --------------------------------------------------------------------------- #
#  Multi-round simulation                                                       #
# --------------------------------------------------------------------------- #


def run_simulation(n_rounds: int = 5, samples_per_round: int = 500) -> list[dict]:
    """Simulate production traffic for *n_rounds* with escalating drift.

    Each successive round injects a higher drift severity so the detection
    pipeline can be tested across a gradient of distribution shift.

    Args:
        n_rounds:         Number of rounds to simulate (default: 5).
        samples_per_round: Samples per production round.

    Returns:
        List of round result dicts, each containing:
          { round, severity, drifted, psi_score, ks_pvalue, affected_features }
    """
    from src.detect import detect_drift  # deferred to avoid circular at import

    reference = generate_reference_data(n=1000, seed=42)

    results: list[dict] = []

    print(f"[simulate] Running {n_rounds} simulation rounds...")

    for i in range(n_rounds):
        # Severity increases linearly from 0.0 to 0.8 across rounds
        severity = round(i * (0.8 / max(n_rounds - 1, 1)), 3)
        seed = 100 + i

        production = inject_drift(reference, severity=severity, seed=seed)

        # Run statistical drift detection
        report = detect_drift(reference, production)

        result = {
            "round": i + 1,
            "severity": severity,
            "drifted": report.drifted,
            "psi_score": report.psi_score,
            "ks_pvalue": report.ks_pvalue,
            "affected_features": report.affected_features,
            "severity_label": report.severity,
        }
        results.append(result)

        status_icon = "DRIFT" if report.drifted else "STABLE"
        print(
            f"  Round {i + 1:>2}/{n_rounds} | "
            f"severity={severity:.3f} | "
            f"PSI={report.psi_score:.4f} | "
            f"KS_pval={report.ks_pvalue:.4f} | "
            f"[{status_icon}]"
        )

    print(f"[simulate] Done. {sum(r['drifted'] for r in results)}/{n_rounds} rounds drifted.")
    return results


# --------------------------------------------------------------------------- #
#  CLI entry point                                                              #
# --------------------------------------------------------------------------- #


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="simulate",
        description="Simulate production traffic with escalating drift.",
    )
    parser.add_argument("--rounds", type=int, default=5, help="Number of simulation rounds.")
    parser.add_argument("--samples", type=int, default=500, help="Samples per round.")
    args = parser.parse_args()

    results = run_simulation(n_rounds=args.rounds, samples_per_round=args.samples)

    # Pretty print summary
    print("\n--- Simulation Summary ---")
    for r in results:
        icon = "DRIFT" if r["drifted"] else "STABLE"
        print(
            f"  Round {r['round']}: severity={r['severity']:.3f}  "
            f"psi={r['psi_score']:.4f}  ks={r['ks_pvalue']:.4f}  [{icon}]"
        )


if __name__ == "__main__":
    main()
