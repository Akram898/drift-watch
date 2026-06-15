# drift-watch

<div align="center">

![CI](https://github.com/Akram898/drift-watch/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)
![MLOps](https://img.shields.io/badge/MLOps-Production%20Ready-8a6fff?style=flat-square)

**Production-grade data drift detection for machine learning models.**
Catches distribution shift before it silently degrades your model in production.

<!-- screenshot placeholder -->
<!-- ![Dashboard Screenshot](docs/dashboard.png) -->

</div>

---

## What it does

- **Trains** a scikit-learn RandomForestClassifier on the Iris dataset and persists it to a versioned model registry
- **Simulates** realistic production traffic with configurable Gaussian drift injection
- **Detects** distribution shift automatically using two complementary statistical tests:
  - **PSI** (Population Stability Index) — industry-standard binned comparison (threshold: 0.2)
  - **KS test** (Kolmogorov-Smirnov) — non-parametric two-sample test (alpha: 0.05)
- **Triggers retraining** automatically when drift exceeds configured thresholds
- **Versions every model** in a JSON registry with full metrics history
- **Serves a real-time dashboard** via FastAPI with auto-refresh every 30 seconds
- **CI pipeline** runs on every push — trains, detects, and posts drift report as PR annotation

---

## Architecture

```
                         drift-watch pipeline
+---------------------------------------------------------------------+
|                                                                     |
|   Raw Data          Feature         Production       Drift          |
|  +--------+        Extraction       Simulator       Detector        |
|  | Iris   |------> +--------+ ----> +----------+ ----> +----------+ |
|  | dataset|        | pandas |       | drift inj|       | PSI + KS | |
|  +--------+        +--------+       +----------+       +----+-----+ |
|                                                             |        |
|                                          +------------------v------+ |
|   Model              Auto               |                        | |
|  Registry           Retrain             |   Drift Report         | |
|  +--------+        +--------+           |  drifted: bool         | |
|  |  JSON  |<-------| sklearn|<----------+  psi_score: float      | |
|  | store  |        | retrain|           |  ks_pvalue: float      | |
|  +--------+        +--------+           |  severity: none/minor/ | |
|      |                                  |           major        | |
|  registry/                              +------------------------+ |
|  models.json                                                        |
+---------------------------------------------------------------------+
              |
              v
    FastAPI Dashboard (port 8000)
    - System Status (green/red)
    - Model Registry table
    - PSI gauge per feature
    - Simulation controls
    - Auto-refresh 30s
```

---

## Quickstart

```bash
# 1 — Install
pip install -r requirements.txt
# or
pip install -e .

# 2 — Train a baseline model on Iris data
python src/train.py
# or with explicit version
python src/train.py --version 1

# 3 — Simulate production traffic with drift
python src/simulate.py

# 4 — Start the dashboard and API
python src/serve.py
# open http://localhost:8000

# 5 — Run drift detection via CLI
drift-watch detect

# 6 — Inspect model registry
drift-watch registry
```

### Via Makefile

```bash
make train      # train model v1
make simulate   # simulate drifted production data
make detect     # run PSI + KS detection
make serve      # start FastAPI dashboard on :8000
make test       # run pytest
make lint       # ruff + black check
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | HTML dashboard |
| `GET` | `/api/status` | JSON: models list, latest drift report, system info |
| `GET` | `/api/models` | Full model registry |
| `POST` | `/api/detect` | Detect drift: body `{ reference: [[...]], production: [[...]] }` |
| `POST` | `/api/simulate` | Trigger simulation; streams results |

---

## Drift Detection Report

Example output from `drift-watch detect`:

```
+------------------+----------+----------+--------------+
| Feature          | PSI      | KS p-val | Status       |
+------------------+----------+----------+--------------+
| sepal_length     | 0.018    | 0.412    | STABLE       |
| sepal_width      | 0.312    | 0.001    | DRIFT        |
| petal_length     | 0.087    | 0.091    | WARNING      |
| petal_width      | 0.021    | 0.387    | STABLE       |
+------------------+----------+----------+--------------+
DRIFT DETECTED — retraining triggered
```

---

## Project Structure

```
drift-watch/
+-- src/
|   +-- __init__.py          # package init
|   +-- train.py             # train RandomForest on Iris, save to registry
|   +-- detect.py            # PSI + KS drift detection, DriftReport dataclass
|   +-- simulate.py          # production traffic simulator with drift injection
|   +-- registry.py          # JSON model registry: register, list, compare
|   +-- serve.py             # FastAPI app: dashboard + REST API
|   +-- drift_watch/         # importable package (same logic, class-based)
|       +-- __init__.py
|       +-- drift.py
|       +-- model.py
|       +-- registry.py
|       +-- simulator.py
|       +-- api.py
|       +-- cli.py
+-- templates/
|   +-- dashboard.html       # dark-theme single-page dashboard
+-- tests/
|   +-- test_detect.py       # pytest: PSI, KS, drift detection
+-- registry/
|   +-- models.json          # committed model metadata (no binaries)
+-- .github/workflows/
|   +-- ci.yml               # lint -> test -> train -> detect-drift
+-- Makefile
+-- vercel.json
+-- pyproject.toml
+-- requirements.txt
```

---

## GitHub Actions CI

Every push to `main` runs the full pipeline:

```
push -> [lint] -> [test] -> [train] -> [detect-drift]
          |          |         |              |
          v          v         v              v
        ruff      pytest    model_v1.pkl   drift report
        black     coverage  registry.json  PR annotation
```

See `.github/workflows/ci.yml` for the full workflow definition.

---

## License

MIT (c) drift-watch contributors
