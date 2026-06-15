"""JSON-based model registry for versioned model tracking.

Public API:
    register_model(version, metrics, path) -> dict
    get_latest_model()                     -> str | None
    list_models()                          -> list[dict]
    compare_models(v1, v2)                 -> dict
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
_DEFAULT_REGISTRY = _ROOT / "models" / "registry.json"


# --------------------------------------------------------------------------- #
#  Internal helpers                                                             #
# --------------------------------------------------------------------------- #


def _load(path: Path) -> dict:
    if not path.exists():
        return {"models": [], "latest_version": None}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _registry_path(override: Path | str | None = None) -> Path:
    return Path(override) if override is not None else _DEFAULT_REGISTRY


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #


def register_model(
    version: int,
    metrics: dict[str, Any],
    path: str,
    registry_file: Path | str | None = None,
    notes: str = "",
) -> dict:
    """Add a newly trained model to the JSON registry.

    Args:
        version:       Integer version number (e.g. 1, 2, 3).
        metrics:       Evaluation metrics dict (accuracy, f1, …).
        path:          File-system path to the saved model artefact.
        registry_file: Override the default registry.json location.
        notes:         Optional human-readable annotation.

    Returns:
        The registry entry that was appended.
    """
    reg_path = _registry_path(registry_file)
    registry = _load(reg_path)

    entry = {
        "version": version,
        "path": str(path),
        "registered_at": datetime.datetime.utcnow().isoformat(),
        "metrics": metrics,
        "notes": notes,
        "status": "active",
    }

    # Replace if same version exists, otherwise append
    existing_versions = [m["version"] for m in registry["models"]]
    if version in existing_versions:
        idx = existing_versions.index(version)
        registry["models"][idx] = entry
    else:
        registry["models"].append(entry)

    registry["latest_version"] = version
    _save(registry, reg_path)
    return entry


def get_latest_model(registry_file: Path | str | None = None) -> str | None:
    """Return the file path of the most recently registered model, or None.

    Args:
        registry_file: Override the default registry.json location.

    Returns:
        Path string to the latest model .pkl file, or None if empty.
    """
    reg_path = _registry_path(registry_file)
    registry = _load(reg_path)

    latest_version = registry.get("latest_version")
    if latest_version is None:
        return None

    for entry in reversed(registry["models"]):
        if entry["version"] == latest_version:
            return entry["path"]

    # Fallback to last entry
    if registry["models"]:
        return registry["models"][-1]["path"]

    return None


def list_models(registry_file: Path | str | None = None) -> list[dict]:
    """Return all registered models, newest version first.

    Args:
        registry_file: Override the default registry.json location.

    Returns:
        List of model entry dicts sorted by version descending.
    """
    reg_path = _registry_path(registry_file)
    registry = _load(reg_path)
    return sorted(registry["models"], key=lambda m: m["version"], reverse=True)


def compare_models(
    v1: int,
    v2: int,
    registry_file: Path | str | None = None,
) -> dict:
    """Side-by-side metric comparison between two model versions.

    Args:
        v1:            First version number.
        v2:            Second version number.
        registry_file: Override the default registry.json location.

    Returns:
        Dict with keys 'v1', 'v2', and 'delta' (v2 metric - v1 metric).

    Raises:
        ValueError: If either version is not found in the registry.
    """
    reg_path = _registry_path(registry_file)
    registry = _load(reg_path)

    entries = {m["version"]: m for m in registry["models"]}

    if v1 not in entries:
        raise ValueError(f"Version {v1} not found in registry")
    if v2 not in entries:
        raise ValueError(f"Version {v2} not found in registry")

    m1 = entries[v1]
    m2 = entries[v2]

    # Compute deltas for numeric metrics
    metrics1 = m1.get("metrics", {})
    metrics2 = m2.get("metrics", {})

    delta: dict[str, Any] = {}
    for key in set(metrics1) | set(metrics2):
        val1 = metrics1.get(key)
        val2 = metrics2.get(key)
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            delta[key] = round(val2 - val1, 6)
        else:
            delta[key] = None

    return {
        "v1": {
            "version": v1,
            "registered_at": m1.get("registered_at"),
            "path": m1.get("path"),
            "metrics": metrics1,
        },
        "v2": {
            "version": v2,
            "registered_at": m2.get("registered_at"),
            "path": m2.get("path"),
            "metrics": metrics2,
        },
        "delta": delta,
        "winner": v2 if delta.get("accuracy", 0) >= 0 else v1,
    }


# --------------------------------------------------------------------------- #
#  CLI shim                                                                     #
# --------------------------------------------------------------------------- #


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="registry", description="Inspect drift-watch model registry.")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List all registered models.")

    cmp = sub.add_parser("compare", help="Compare two model versions.")
    cmp.add_argument("v1", type=int)
    cmp.add_argument("v2", type=int)

    sub.add_parser("latest", help="Print path to latest model.")

    args = parser.parse_args()

    if args.cmd == "list" or args.cmd is None:
        models = list_models()
        if not models:
            print("Registry is empty. Run: python src/train.py")
            return
        print(f"{'Version':<10} {'Accuracy':<12} {'F1':<12} {'Registered At'}")
        print("-" * 60)
        for m in models:
            met = m.get("metrics", {})
            print(
                f"  v{m['version']:<8} "
                f"{met.get('accuracy', '-'):<12} "
                f"{met.get('f1_weighted', '-'):<12} "
                f"{m.get('registered_at', '')[:19]}"
            )

    elif args.cmd == "compare":
        result = compare_models(args.v1, args.v2)
        print(json.dumps(result, indent=2))

    elif args.cmd == "latest":
        path = get_latest_model()
        print(path or "No models registered yet.")


if __name__ == "__main__":
    main()
