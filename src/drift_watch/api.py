"""FastAPI dashboard API for drift-watch."""

from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from drift_watch.registry import list_models, get_latest
from drift_watch.drift import detect_drift

app = FastAPI(
    title="drift-watch",
    description="MLOps drift detection dashboard",
    version="0.1.0",
)

DATA_DIR = Path("data")
REGISTRY_FILE = Path("registry") / "models.json"

# ──────────────────────────────────────────────────────────────────────────── #
#  Helpers                                                                     #
# ──────────────────────────────────────────────────────────────────────────── #


def _load_last_report() -> dict | None:
    report_path = DATA_DIR / "report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    return None


def _psi_bar_width(psi: float, max_psi: float = 0.6) -> int:
    """Return a bar width (0–100) clamped to max_psi."""
    return min(100, int((psi / max_psi) * 100))


def _status_color(status: str) -> str:
    return {"stable": "#22c55e", "warning": "#f59e0b", "drift": "#ef4444"}.get(status, "#94a3b8")


def _status_emoji(status: str) -> str:
    return {"stable": "✅", "warning": "⚠️", "drift": "🚨"}.get(status, "❓")


# ──────────────────────────────────────────────────────────────────────────── #
#  Dashboard HTML                                                              #
# ──────────────────────────────────────────────────────────────────────────── #


def _build_feature_rows(report: dict) -> str:
    rows = []
    for feat in report.get("features", []):
        status = feat["status"]
        color = _status_color(status)
        emoji = _status_emoji(status)
        psi = feat["psi"]
        bar_w = _psi_bar_width(psi)
        rows.append(f"""
        <tr>
          <td class="feat-name">{feat['feature']}</td>
          <td>
            <div class="bar-wrap">
              <div class="bar-fill" style="width:{bar_w}%;background:{color}"></div>
              <span class="bar-label">{psi:.4f}</span>
            </div>
          </td>
          <td class="num">{feat['ks_statistic']:.4f}</td>
          <td class="num" style="color:{'#ef4444' if feat['ks_pvalue'] < 0.05 else '#22c55e'}">{feat['ks_pvalue']:.4f}</td>
          <td><span class="badge" style="background:{color}20;color:{color};border:1px solid {color}40">{emoji} {status.upper()}</span></td>
        </tr>""")
    return "\n".join(rows) if rows else "<tr><td colspan='5' class='empty'>No detection data yet — run drift-watch detect</td></tr>"


def _build_registry_rows(models: list[dict]) -> str:
    rows = []
    for i, m in enumerate(models[:6]):  # show last 6
        metrics = m.get("metrics", {})
        acc = metrics.get("accuracy", "—")
        auc = metrics.get("roc_auc", "—")
        is_latest = i == 0
        badge = '<span class="latest-badge">LATEST</span>' if is_latest else ""
        rows.append(f"""
        <tr>
          <td class="num dim">#{m['id']}</td>
          <td class="dim">{m['registered_at'][:19]}</td>
          <td class="num green">{acc:.4f if isinstance(acc, float) else acc}</td>
          <td class="num">{auc:.4f if isinstance(auc, float) else auc}</td>
          <td>{m.get('notes','') or '—'}{badge}</td>
        </tr>""")
    return "\n".join(rows) if rows else "<tr><td colspan='5' class='empty'>No models registered yet</td></tr>"


def _render_dashboard() -> str:
    report = _load_last_report()
    models = list_models(REGISTRY_FILE)
    latest = get_latest(REGISTRY_FILE)

    # Overall status
    if report:
        drift_detected = report.get("overall_drift_detected", False)
        overall_color = "#ef4444" if drift_detected else "#22c55e"
        overall_label = "🚨 DRIFT DETECTED" if drift_detected else "✅ ALL STABLE"
        last_run = report.get("timestamp", "—")[:19] if report.get("timestamp") else "—"
        n_ref = report.get("n_reference", "—")
        n_prod = report.get("n_production", "—")
    else:
        overall_color = "#6366f1"
        overall_label = "— No detection run yet"
        last_run = "—"
        n_ref = "—"
        n_prod = "—"

    # Latest model card
    if latest:
        m = latest
        metrics = m.get("metrics", {})
        model_acc = metrics.get("accuracy", "—")
        model_auc = metrics.get("roc_auc", "—")
        model_ts = m.get("registered_at", "—")[:19]
        model_id = f"#{m['id']}"
    else:
        model_acc = model_auc = model_ts = "—"
        model_id = "—"

    feature_rows = _build_feature_rows(report or {})
    registry_rows = _build_registry_rows(models)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta http-equiv="refresh" content="30" />
  <title>drift-watch dashboard</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg:        #0a0f1e;
      --surface:   #111827;
      --surface2:  #1a2236;
      --border:    #1e2d45;
      --text:      #e2e8f0;
      --muted:     #64748b;
      --accent:    #6366f1;
      --accent2:   #818cf8;
      --green:     #22c55e;
      --yellow:    #f59e0b;
      --red:       #ef4444;
    }}

    html, body {{
      height: 100%;
      background: var(--bg);
      color: var(--text);
      font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
      font-size: 14px;
      line-height: 1.6;
    }}

    /* ── Layout ─────────────────────────────────────────── */
    .shell {{
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 100vh;
    }}

    header {{
      background: linear-gradient(135deg, #0d1424 0%, #111827 100%);
      border-bottom: 1px solid var(--border);
      padding: 0 2rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      height: 64px;
    }}

    .logo {{
      display: flex;
      align-items: center;
      gap: 0.6rem;
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }}
    .logo-icon {{
      width: 32px; height: 32px;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      border-radius: 8px;
      display: flex; align-items: center; justify-content: center;
      font-size: 16px;
    }}
    .logo-sub {{ color: var(--muted); font-weight: 400; font-size: 0.8rem; }}

    .header-right {{
      display: flex;
      align-items: center;
      gap: 1rem;
      font-size: 0.8rem;
      color: var(--muted);
    }}
    .refresh-dot {{
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--green);
      animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
      0%,100% {{ opacity: 1; }}
      50%      {{ opacity: 0.3; }}
    }}

    main {{
      padding: 2rem;
      display: grid;
      gap: 1.5rem;
      max-width: 1400px;
      width: 100%;
      margin: 0 auto;
    }}

    /* ── Cards ──────────────────────────────────────────── */
    .cards-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
    }}

    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem 1.5rem;
      position: relative;
      overflow: hidden;
    }}
    .card::before {{
      content: '';
      position: absolute;
      inset: 0;
      background: linear-gradient(135deg, rgba(99,102,241,0.04) 0%, transparent 60%);
      pointer-events: none;
    }}

    .card-label {{
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 0.5rem;
    }}
    .card-value {{
      font-size: 1.6rem;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .card-sub {{
      font-size: 0.75rem;
      color: var(--muted);
      margin-top: 0.25rem;
    }}

    /* status card accent strip */
    .card-status {{
      border-left: 4px solid {overall_color};
    }}
    .card-model  {{ border-left: 4px solid var(--accent); }}
    .card-acc    {{ border-left: 4px solid var(--green); }}
    .card-refs   {{ border-left: 4px solid #06b6d4; }}

    /* ── Section panels ─────────────────────────────────── */
    .panel {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }}

    .panel-header {{
      padding: 1rem 1.5rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: var(--surface2);
    }}
    .panel-title {{
      font-weight: 600;
      font-size: 0.9rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}
    .panel-title .dot {{
      width: 8px; height: 8px;
      border-radius: 50%;
    }}
    .panel-hint {{
      font-size: 0.75rem;
      color: var(--muted);
    }}

    /* ── Tables ─────────────────────────────────────────── */
    .tbl-wrap {{ overflow-x: auto; }}

    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    thead tr {{
      background: rgba(255,255,255,0.02);
    }}
    th, td {{
      padding: 0.75rem 1rem;
      text-align: left;
      border-bottom: 1px solid rgba(255,255,255,0.04);
    }}
    th {{
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      color: var(--muted);
      font-weight: 600;
    }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: rgba(255,255,255,0.025); }}

    .feat-name {{ font-weight: 500; font-family: 'Fira Code', monospace; font-size: 0.85rem; }}
    .num {{ font-family: 'Fira Code', monospace; font-size: 0.85rem; }}
    .dim {{ color: var(--muted); }}
    .green {{ color: var(--green); }}
    .empty {{ color: var(--muted); text-align: center; padding: 2rem; font-style: italic; }}

    /* PSI bar */
    .bar-wrap {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }}
    .bar-fill {{
      height: 6px;
      border-radius: 3px;
      min-width: 2px;
      flex-shrink: 0;
      transition: width 0.4s ease;
    }}
    .bar-label {{
      font-family: 'Fira Code', monospace;
      font-size: 0.8rem;
      white-space: nowrap;
    }}

    /* Badges */
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.2rem 0.6rem;
      border-radius: 20px;
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.04em;
    }}

    .latest-badge {{
      margin-left: 0.5rem;
      display: inline-flex;
      align-items: center;
      padding: 0.15rem 0.5rem;
      border-radius: 20px;
      font-size: 0.68rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      background: rgba(99,102,241,0.15);
      color: var(--accent2);
      border: 1px solid rgba(99,102,241,0.3);
    }}

    /* ── API quick ref ──────────────────────────────────── */
    .api-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 0.75rem;
      padding: 1rem 1.5rem;
    }}
    .api-item {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
      padding: 0.75rem 1rem;
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    .method {{
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-size: 0.7rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      min-width: 44px;
      text-align: center;
    }}
    .method.get  {{ background: rgba(34,197,94,0.15); color: var(--green); }}
    .method.post {{ background: rgba(99,102,241,0.15); color: var(--accent2); }}
    .api-path {{
      font-family: 'Fira Code', monospace;
      font-size: 0.8rem;
      color: var(--text);
    }}
    .api-desc {{ font-size: 0.75rem; color: var(--muted); }}

    /* ── Footer ─────────────────────────────────────────── */
    footer {{
      text-align: center;
      padding: 1.5rem;
      color: var(--muted);
      font-size: 0.75rem;
      border-top: 1px solid var(--border);
      margin-top: 1rem;
    }}
    footer a {{ color: var(--accent2); text-decoration: none; }}
    footer a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
<div class="shell">

  <!-- ── Header ──────────────────────────────────────────────────────── -->
  <header>
    <div class="logo">
      <div class="logo-icon">🔬</div>
      <div>
        drift-watch
        <div class="logo-sub">MLOps Drift Detection Dashboard</div>
      </div>
    </div>
    <div class="header-right">
      <div class="refresh-dot"></div>
      <span>Auto-refresh every 30s</span>
      <span>·</span>
      <span>v0.1.0</span>
    </div>
  </header>

  <!-- ── Main ────────────────────────────────────────────────────────── -->
  <main>

    <!-- KPI cards -->
    <div class="cards-row">
      <div class="card card-status">
        <div class="card-label">Overall Status</div>
        <div class="card-value" style="font-size:1.1rem;color:{overall_color}">{overall_label}</div>
        <div class="card-sub">Last run: {last_run}</div>
      </div>

      <div class="card card-model">
        <div class="card-label">Current Model</div>
        <div class="card-value" style="font-size:1.3rem;color:var(--accent2)">{model_id}</div>
        <div class="card-sub">Registered {model_ts}</div>
      </div>

      <div class="card card-acc">
        <div class="card-label">Model Accuracy</div>
        <div class="card-value" style="color:var(--green)">{f'{model_acc:.4f}' if isinstance(model_acc, float) else model_acc}</div>
        <div class="card-sub">ROC-AUC: {f'{model_auc:.4f}' if isinstance(model_auc, float) else model_auc}</div>
      </div>

      <div class="card card-refs">
        <div class="card-label">Sample Sizes</div>
        <div class="card-value" style="font-size:1.2rem;color:#06b6d4">{n_ref} / {n_prod}</div>
        <div class="card-sub">Reference / Production</div>
      </div>
    </div>

    <!-- Feature drift table -->
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">
          <div class="dot" style="background:{overall_color}"></div>
          Per-Feature Drift Analysis
        </div>
        <span class="panel-hint">PSI threshold 0.2 · KS α 0.05</span>
      </div>
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr>
              <th>Feature</th>
              <th>PSI Score</th>
              <th>KS Statistic</th>
              <th>KS p-value</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {feature_rows}
          </tbody>
        </table>
      </div>
    </div>

    <!-- Model registry table -->
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">
          <div class="dot" style="background:var(--accent)"></div>
          Model Registry
        </div>
        <span class="panel-hint">Showing last 6 models · {len(models)} total</span>
      </div>
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Registered At</th>
              <th>Accuracy</th>
              <th>ROC-AUC</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>
            {registry_rows}
          </tbody>
        </table>
      </div>
    </div>

    <!-- API reference -->
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">
          <div class="dot" style="background:var(--yellow)"></div>
          REST API
        </div>
        <a href="/docs" style="font-size:0.75rem;color:var(--accent2);text-decoration:none;">OpenAPI docs →</a>
      </div>
      <div class="api-grid">
        <div class="api-item">
          <div class="method get">GET</div>
          <div>
            <div class="api-path">/api/status</div>
            <div class="api-desc">Current model + drift status</div>
          </div>
        </div>
        <div class="api-item">
          <div class="method get">GET</div>
          <div>
            <div class="api-path">/api/registry</div>
            <div class="api-desc">Full model version registry</div>
          </div>
        </div>
        <div class="api-item">
          <div class="method post">POST</div>
          <div>
            <div class="api-path">/api/detect</div>
            <div class="api-desc">Trigger live drift detection</div>
          </div>
        </div>
        <div class="api-item">
          <div class="method get">GET</div>
          <div>
            <div class="api-path">/docs</div>
            <div class="api-desc">Interactive Swagger UI</div>
          </div>
        </div>
      </div>
    </div>

  </main>

  <footer>
    drift-watch v0.1.0 &mdash; built with
    <a href="https://fastapi.tiangolo.com">FastAPI</a> &amp;
    <a href="https://scikit-learn.org">scikit-learn</a>
    &mdash; <a href="https://github.com">View on GitHub</a>
  </footer>

</div>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────────────── #
#  Routes                                                                      #
# ──────────────────────────────────────────────────────────────────────────── #


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    """Serve the dark-mode drift monitoring dashboard."""
    return HTMLResponse(content=_render_dashboard())


@app.get("/api/status")
async def status() -> JSONResponse:
    """Return a JSON snapshot of current model info and drift status."""
    latest = get_latest(REGISTRY_FILE)
    report = _load_last_report()

    return JSONResponse(
        {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "model": latest,
            "drift_detected": report["overall_drift_detected"] if report else None,
            "last_detection": report["timestamp"] if report else None,
        }
    )


@app.get("/api/registry")
async def registry() -> JSONResponse:
    """Return all registered model versions."""
    models = list_models(REGISTRY_FILE)
    return JSONResponse({"models": models, "count": len(models)})


@app.post("/api/detect")
async def run_detect() -> JSONResponse:
    """Trigger drift detection between reference and production CSVs."""
    import pandas as pd

    ref_path = DATA_DIR / "training.csv"
    prod_path = DATA_DIR / "production.csv"

    if not ref_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Reference data not found. Run `drift-watch train` first.",
        )
    if not prod_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Production data not found. Run `drift-watch simulate` first.",
        )

    ref_df = pd.read_csv(ref_path)
    prod_df = pd.read_csv(prod_path)
    report = detect_drift(ref_df, prod_df)

    # Persist for dashboard
    report_dict = report.to_dict()
    (DATA_DIR / "report.json").write_text(json.dumps(report_dict, indent=2))

    return JSONResponse(report_dict)
