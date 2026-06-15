"""Click CLI — entry point for all drift-watch commands."""

from __future__ import annotations

import sys
import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

# Canonical paths (relative to cwd, resolved at runtime)
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
REGISTRY_FILE = Path("registry") / "models.json"


# --------------------------------------------------------------------------- #
#  CLI group                                                                   #
# --------------------------------------------------------------------------- #


@click.group()
@click.version_option("0.1.0", prog_name="drift-watch")
def main() -> None:
    """drift-watch — MLOps drift detection toolkit.

    \b
    Workflow:
      1. drift-watch train      — train baseline model
      2. drift-watch simulate   — generate production traffic with drift
      3. drift-watch detect     — detect drift between train/prod distributions
      4. drift-watch retrain    — retrain on combined data, register new model
      5. drift-watch registry   — inspect model registry
    """


# --------------------------------------------------------------------------- #
#  train                                                                       #
# --------------------------------------------------------------------------- #


@main.command()
@click.option("--n-samples", default=5_000, show_default=True, help="Training set size.")
@click.option("--n-estimators", default=200, show_default=True, help="RF trees.")
@click.option("--seed", default=42, show_default=True, help="Random seed.")
def train(n_samples: int, n_estimators: int, seed: int) -> None:
    """Train a RandomForestClassifier on synthetic bank-churn data."""
    from drift_watch.model import generate_training_data, train_model, save_model
    from drift_watch.registry import register_model

    console.print(Panel("[bold cyan]drift-watch train[/bold cyan]", expand=False))

    with console.status("Generating synthetic training data…"):
        df = generate_training_data(n_samples=n_samples, seed=seed)

    X = df.drop(columns=["churn"])
    y = df["churn"]

    console.print(
        f"  Dataset: [green]{len(df):,}[/green] rows, "
        f"churn rate [yellow]{y.mean():.1%}[/yellow]"
    )

    with console.status("Training RandomForestClassifier…"):
        clf, scaler, metrics = train_model(X, y, n_estimators=n_estimators, seed=seed)

    console.print(
        f"  Accuracy [green]{metrics['accuracy']:.4f}[/green]  "
        f"ROC-AUC [green]{metrics['roc_auc']:.4f}[/green]"
    )

    # Persist training data for later drift comparison
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    training_csv = DATA_DIR / "training.csv"
    df.to_csv(training_csv, index=False)
    console.print(f"  Training data saved → [dim]{training_csv}[/dim]")

    # Save model with timestamp and as latest
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    versioned_path = MODELS_DIR / f"model_v{ts}.pkl"
    latest_path = MODELS_DIR / "latest.pkl"

    save_model(clf, scaler, versioned_path)
    save_model(clf, scaler, latest_path)

    console.print(f"  Model saved → [dim]{versioned_path}[/dim]")
    console.print(f"  Symlink    → [dim]{latest_path}[/dim]")

    # Register
    entry = register_model(versioned_path, metrics, registry_file=REGISTRY_FILE)
    console.print(
        f"\n[bold green]✅  Model registered[/bold green] (id={entry['id']}, "
        f"accuracy={metrics['accuracy']})"
    )


# --------------------------------------------------------------------------- #
#  simulate                                                                    #
# --------------------------------------------------------------------------- #


@main.command()
@click.option("--n-samples", default=1_000, show_default=True, help="Production sample count.")
@click.option(
    "--drift-factor",
    default=2.0,
    show_default=True,
    help="Mean shift in units of std-dev for drifted features.",
)
@click.option(
    "--drift-fraction",
    default=0.3,
    show_default=True,
    help="Fraction of features to drift (0.0–1.0).",
)
@click.option("--seed", default=99, show_default=True, help="Random seed.")
@click.option("--no-drift", is_flag=True, default=False, help="Simulate stable traffic (no drift).")
def simulate(
    n_samples: int,
    drift_factor: float,
    drift_fraction: float,
    seed: int,
    no_drift: bool,
) -> None:
    """Generate production traffic and save to data/production.csv."""
    from drift_watch.simulator import simulate_production_traffic, simulate_stable_traffic

    console.print(Panel("[bold cyan]drift-watch simulate[/bold cyan]", expand=False))

    with console.status("Simulating production traffic…"):
        if no_drift:
            df = simulate_stable_traffic(n_samples=n_samples, seed=seed)
            console.print("  Mode: [green]STABLE[/green] (no drift injected)")
        else:
            df = simulate_production_traffic(
                n_samples=n_samples,
                drift_factor=drift_factor,
                drift_fraction=drift_fraction,
                seed=seed,
            )
            console.print(
                f"  Mode: [red]DRIFTED[/red] — "
                f"shift={drift_factor}σ on {drift_fraction:.0%} of features"
            )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "production.csv"
    df.to_csv(out_path, index=False)

    console.print(f"  Samples: [green]{len(df):,}[/green]")
    console.print(f"\n[bold green]✅  Production data saved[/bold green] → [dim]{out_path}[/dim]")


# --------------------------------------------------------------------------- #
#  detect                                                                      #
# --------------------------------------------------------------------------- #


@main.command()
@click.option(
    "--threshold",
    default=0.2,
    show_default=True,
    help="PSI threshold above which drift is flagged.",
)
@click.option(
    "--reference",
    default=str(DATA_DIR / "training.csv"),
    show_default=True,
    help="Path to reference CSV.",
)
@click.option(
    "--production",
    default=str(DATA_DIR / "production.csv"),
    show_default=True,
    help="Path to production CSV.",
)
@click.option("--save-report", is_flag=True, default=False, help="Save JSON report to data/report.json.")
def detect(threshold: float, reference: str, production: str, save_report: bool) -> None:
    """Run PSI + KS drift detection and print a colour-coded report."""
    import pandas as pd
    from drift_watch.drift import detect_drift

    console.print(Panel("[bold cyan]drift-watch detect[/bold cyan]", expand=False))

    ref_path = Path(reference)
    prod_path = Path(production)

    if not ref_path.exists():
        console.print(f"[red]Reference file not found:[/red] {ref_path}")
        console.print("Run [bold]drift-watch train[/bold] first.")
        sys.exit(1)

    if not prod_path.exists():
        console.print(f"[red]Production file not found:[/red] {prod_path}")
        console.print("Run [bold]drift-watch simulate[/bold] first.")
        sys.exit(1)

    with console.status("Loading data…"):
        ref_df = pd.read_csv(ref_path)
        prod_df = pd.read_csv(prod_path)

    console.print(
        f"  Reference:  [dim]{ref_path}[/dim] "
        f"([green]{len(ref_df):,}[/green] rows)"
    )
    console.print(
        f"  Production: [dim]{prod_path}[/dim] "
        f"([green]{len(prod_df):,}[/green] rows)"
    )

    with console.status("Running drift detection…"):
        report = detect_drift(ref_df, prod_df, threshold=threshold)

    # ── Rich table ──────────────────────────────────────────────────────────
    table = Table(
        title="Drift Detection Report",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Feature", style="bold", min_width=18)
    table.add_column("PSI", justify="right", min_width=8)
    table.add_column("KS stat", justify="right", min_width=8)
    table.add_column("KS p-val", justify="right", min_width=10)
    table.add_column("Status", min_width=20)

    for r in report.features:
        psi_color = "red" if r.psi >= threshold else ("yellow" if r.psi >= 0.1 else "green")
        pval_color = "red" if r.ks_pvalue < 0.05 else "green"
        table.add_row(
            r.feature,
            f"[{psi_color}]{r.psi:.4f}[/{psi_color}]",
            f"{r.ks_statistic:.4f}",
            f"[{pval_color}]{r.ks_pvalue:.4f}[/{pval_color}]",
            f"{r.emoji} {r.label}",
        )

    console.print()
    console.print(table)
    console.print()

    if report.overall_drift_detected:
        n_drift = len(report.drifted_features)
        n_warn = len(report.warning_features)
        console.print(
            f"[bold red]🚨  DRIFT DETECTED[/bold red] — "
            f"{n_drift} feature(s) drifted, {n_warn} warning(s). "
            f"Run [bold]drift-watch retrain[/bold] to update the model."
        )
        if save_report:
            _save_json_report(report)
        sys.exit(1)
    else:
        n_warn = len(report.warning_features)
        if n_warn:
            console.print(
                f"[bold yellow]⚠️   {n_warn} WARNING(s)[/bold yellow] — "
                f"monitor closely but no retraining required yet."
            )
        else:
            console.print("[bold green]✅  All features STABLE — no action required.[/bold green]")

    if save_report:
        _save_json_report(report)


def _save_json_report(report) -> None:
    import json
    out = DATA_DIR / "report.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2))
    console.print(f"  Report saved → [dim]{out}[/dim]")


# --------------------------------------------------------------------------- #
#  retrain                                                                     #
# --------------------------------------------------------------------------- #


@main.command()
@click.option("--seed", default=7, show_default=True, help="Random seed for retrain.")
def retrain(seed: int) -> None:
    """Retrain model on combined reference + production data, update registry."""
    import pandas as pd
    from drift_watch.model import train_model, save_model
    from drift_watch.registry import register_model

    console.print(Panel("[bold cyan]drift-watch retrain[/bold cyan]", expand=False))

    ref_path = DATA_DIR / "training.csv"
    prod_path = DATA_DIR / "production.csv"

    if not ref_path.exists():
        console.print(f"[red]Reference data not found:[/red] {ref_path}")
        sys.exit(1)

    with console.status("Loading datasets…"):
        ref_df = pd.read_csv(ref_path)
        if prod_path.exists():
            prod_df = pd.read_csv(prod_path)
            # Production data has no churn column — we need to generate pseudo-labels
            # In practice you'd have delayed labels; here we use the baseline model
            from drift_watch.model import load_model, FEATURE_NAMES
            latest_pkl = MODELS_DIR / "latest.pkl"
            if latest_pkl.exists():
                clf, scaler = load_model(latest_pkl)
                import numpy as np
                X_prod = prod_df[FEATURE_NAMES].fillna(0).values
                X_prod_scaled = scaler.transform(X_prod)
                pseudo_labels = clf.predict(X_prod_scaled)
                prod_df = prod_df.copy()
                prod_df["churn"] = pseudo_labels
            else:
                console.print("[yellow]No trained model found — using reference data only.[/yellow]")
                prod_df = None
        else:
            prod_df = None

    if prod_df is not None:
        combined = pd.concat([ref_df, prod_df], ignore_index=True)
        console.print(
            f"  Combined dataset: [green]{len(combined):,}[/green] rows "
            f"(ref={len(ref_df):,} + prod={len(prod_df):,})"
        )
    else:
        combined = ref_df
        console.print(f"  Using reference only: [green]{len(combined):,}[/green] rows")

    X = combined.drop(columns=["churn"])
    y = combined["churn"]

    with console.status("Retraining RandomForestClassifier…"):
        clf, scaler, metrics = train_model(X, y, seed=seed)

    console.print(
        f"  Accuracy [green]{metrics['accuracy']:.4f}[/green]  "
        f"ROC-AUC [green]{metrics['roc_auc']:.4f}[/green]"
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    versioned_path = MODELS_DIR / f"model_v{ts}.pkl"
    latest_path = MODELS_DIR / "latest.pkl"

    save_model(clf, scaler, versioned_path)
    save_model(clf, scaler, latest_path)

    # Save new combined dataset as the new training reference
    combined.to_csv(ref_path, index=False)

    entry = register_model(
        versioned_path,
        metrics,
        registry_file=REGISTRY_FILE,
        notes="retrain after drift detection",
    )
    console.print(
        f"\n[bold green]✅  Retrain complete[/bold green] "
        f"(id={entry['id']}, accuracy={metrics['accuracy']})"
    )
    console.print(f"  Model → [dim]{versioned_path}[/dim]")


# --------------------------------------------------------------------------- #
#  registry                                                                    #
# --------------------------------------------------------------------------- #


@main.command(name="registry")
def show_registry() -> None:
    """Display the model registry in a formatted table."""
    from drift_watch.registry import list_models, get_latest

    console.print(Panel("[bold cyan]drift-watch registry[/bold cyan]", expand=False))

    models = list_models(registry_file=REGISTRY_FILE)
    latest = get_latest(registry_file=REGISTRY_FILE)

    if not models:
        console.print("[yellow]Registry is empty. Run[/yellow] [bold]drift-watch train[/bold] [yellow]first.[/yellow]")
        return

    table = Table(
        title=f"Model Registry  ([dim]{REGISTRY_FILE}[/dim])",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("ID", justify="right", style="dim", width=4)
    table.add_column("Registered", min_width=20)
    table.add_column("Accuracy", justify="right")
    table.add_column("ROC-AUC", justify="right")
    table.add_column("N Train", justify="right")
    table.add_column("Notes")
    table.add_column("", width=8)

    for m in models:
        metrics = m.get("metrics", {})
        is_latest = latest and m["path"] == latest["path"]
        table.add_row(
            str(m["id"]),
            m["registered_at"][:19],
            f"[green]{metrics.get('accuracy', '—'):.4f}[/green]"
            if isinstance(metrics.get("accuracy"), float)
            else "—",
            f"{metrics.get('roc_auc', '—'):.4f}"
            if isinstance(metrics.get("roc_auc"), float)
            else "—",
            str(metrics.get("n_train", "—")),
            m.get("notes", ""),
            "[bold cyan]LATEST[/bold cyan]" if is_latest else "",
        )

    console.print(table)
    console.print(f"\nTotal: [green]{len(models)}[/green] model(s) registered.")
