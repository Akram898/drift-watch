"""FastAPI server for the drift-watch dashboard and REST API.

Endpoints:
    GET  /             — HTML dashboard (reads templates/dashboard.html)
    GET  /api/status   — { models, latest_drift, system }
    GET  /api/models   — model registry list
    POST /api/detect   — run drift detection on body { reference: [[...]], production: [[...]] }
    POST /api/simulate — run simulation; returns JSON stream of round results

Usage:
    python src/serve.py
    uvicorn src.serve:app --reload --port 8000
"""

from __future__ import annotations

import datetime
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
_TEMPLATES = _ROOT / "templates"

app = FastAPI(
    title="drift-watch",
    description="MLOps model monitoring and drift detection API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# --------------------------------------------------------------------------- #
#  Request / response models                                                   #
# --------------------------------------------------------------------------- #


class DetectRequest(BaseModel):
    reference: list[list[float]]
    """2-D list: each inner list is one sample row."""

    production: list[list[float]]
    """2-D list: each inner list is one sample row."""

    feature_names: list[str] | None = None
    """Optional column names. Defaults to feature_0, feature_1, …"""


class SimulateRequest(BaseModel):
    n_rounds: int = 5
    samples_per_round: int = 500


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _load_registry() -> list[dict]:
    from src.registry import list_models
    return list_models()


def _load_latest_drift() -> dict | None:
    report_path = _ROOT / "data" / "latest_drift_report.json"
    if report_path.exists():
        try:
            return json.loads(report_path.read_text())
        except Exception:
            return None
    return None


def _save_drift_report(report_dict: dict) -> None:
    out = _ROOT / "data" / "latest_drift_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report_dict, indent=2))


def _system_info() -> dict:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.system(),
        "server_time": datetime.datetime.utcnow().isoformat(),
        "uptime": "live",
    }


def _read_dashboard_html() -> str:
    dashboard_path = _TEMPLATES / "dashboard.html"
    if not dashboard_path.exists():
        return "<html><body><h1>dashboard.html not found</h1></body></html>"
    return dashboard_path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
#  Routes                                                                      #
# --------------------------------------------------------------------------- #


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root() -> HTMLResponse:
    """Serve the drift-watch HTML dashboard."""
    return HTMLResponse(content=_read_dashboard_html())


@app.get("/api/status")
async def status() -> JSONResponse:
    """Return a JSON snapshot: models list, latest drift report, system info."""
    models = _load_registry()
    latest_drift = _load_latest_drift()
    return JSONResponse(
        {
            "models": models,
            "latest_drift": latest_drift,
            "system": _system_info(),
        }
    )


@app.get("/api/models")
async def get_models() -> JSONResponse:
    """Return all registered model versions."""
    models = _load_registry()
    return JSONResponse({"models": models, "count": len(models)})


@app.post("/api/detect")
async def detect(body: DetectRequest) -> JSONResponse:
    """Run drift detection on provided reference and production arrays.

    Body:
        {
          "reference":   [[f0, f1, f2, f3], ...],
          "production":  [[f0, f1, f2, f3], ...],
          "feature_names": ["sepal_length", ...]  // optional
        }

    Returns DriftReport as JSON.
    """
    from src.detect import detect_drift

    if not body.reference:
        raise HTTPException(status_code=400, detail="'reference' array is empty")
    if not body.production:
        raise HTTPException(status_code=400, detail="'production' array is empty")

    # Build DataFrames
    n_features = len(body.reference[0])
    if body.feature_names:
        if len(body.feature_names) != n_features:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"feature_names length ({len(body.feature_names)}) "
                    f"does not match sample width ({n_features})"
                ),
            )
        cols = body.feature_names
    else:
        cols = [f"feature_{i}" for i in range(n_features)]

    ref_df = pd.DataFrame(body.reference, columns=cols)
    prod_df = pd.DataFrame(body.production, columns=cols)

    report = detect_drift(ref_df, prod_df)
    report_dict = report.to_dict()

    _save_drift_report(report_dict)

    return JSONResponse(report_dict)


@app.post("/api/simulate")
async def simulate(body: SimulateRequest) -> StreamingResponse:
    """Trigger a multi-round drift simulation and stream results as JSON lines."""

    def _stream():
        from src.simulate import generate_reference_data, inject_drift
        from src.detect import detect_drift

        reference = generate_reference_data(n=1000, seed=42)

        yield json.dumps({"event": "start", "n_rounds": body.n_rounds}) + "\n"

        results = []
        for i in range(body.n_rounds):
            severity = round(i * (0.8 / max(body.n_rounds - 1, 1)), 3)
            production = inject_drift(reference, severity=severity, seed=100 + i)
            report = detect_drift(reference, production)

            result = {
                "event": "round",
                "round": i + 1,
                "severity": severity,
                "drifted": report.drifted,
                "psi_score": report.psi_score,
                "ks_pvalue": report.ks_pvalue,
                "affected_features": report.affected_features,
                "severity_label": report.severity,
            }
            results.append(result)
            yield json.dumps(result) + "\n"
            time.sleep(0.05)  # small delay for streaming UX

        # Save final round's report for dashboard
        if results:
            last = results[-1]
            _save_drift_report(last)

        n_drifted = sum(r["drifted"] for r in results)
        yield json.dumps({"event": "done", "n_drifted": n_drifted, "n_rounds": body.n_rounds}) + "\n"

    return StreamingResponse(
        _stream(),
        media_type="application/x-ndjson",
    )


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main() -> None:
    uvicorn.run(
        "src.serve:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
