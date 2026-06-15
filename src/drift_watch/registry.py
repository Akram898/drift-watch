"""JSON-based model registry for versioned model tracking."""

from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Any

# Default registry location (relative to project root, but resolved at runtime)
_DEFAULT_REGISTRY_PATH = Path("registry") / "models.json"


def _registry_path(registry_file: Path | str | None = None) -> Path:
    if registry_file is not None:
        return Path(registry_file)
    return _DEFAULT_REGISTRY_PATH


def _load_registry(registry_file: Path | str | None = None) -> dict:
    path = _registry_path(registry_file)
    if not path.exists():
        return {"models": [], "latest": None}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_registry(data: dict, registry_file: Path | str | None = None) -> None:
    path = _registry_path(registry_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def register_model(
    model_path: str | Path,
    metrics: dict[str, Any],
    registry_file: Path | str | None = None,
    notes: str = "",
) -> dict:
    """Register a trained model and its evaluation metrics.

    Args:
        model_path:     Path to the saved model file (*.pkl).
        metrics:        Dict of evaluation metrics (accuracy, roc_auc, etc.).
        registry_file:  Override the default registry JSON path.
        notes:          Optional human-readable notes.

    Returns:
        The registry entry that was added.
    """
    registry = _load_registry(registry_file)

    entry = {
        "id": len(registry["models"]) + 1,
        "path": str(model_path),
        "registered_at": datetime.datetime.utcnow().isoformat(),
        "metrics": metrics,
        "notes": notes,
        "status": "active",
    }

    registry["models"].append(entry)
    registry["latest"] = entry
    _save_registry(registry, registry_file)
    return entry


def list_models(registry_file: Path | str | None = None) -> list[dict]:
    """Return all entries in the model registry (newest first)."""
    registry = _load_registry(registry_file)
    return list(reversed(registry["models"]))


def get_latest(registry_file: Path | str | None = None) -> dict | None:
    """Return the most recently registered model entry, or None."""
    registry = _load_registry(registry_file)
    return registry.get("latest")


def get_registry_path(registry_file: Path | str | None = None) -> Path:
    """Return the resolved path to the registry JSON file."""
    return _registry_path(registry_file)
