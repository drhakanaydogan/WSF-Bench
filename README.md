# WSF-Bench Python Core

This repository contains the curated data tables, Python scripts, diagnostic outputs, and figure/table generation utilities supporting the WSF-Bench analysis of wood-related production and trade monitoring.

The package supports inspection and reproduction of the reported benchmark outputs, including leakage-safe model comparisons, execution-validity diagnostics, recurrent challenger results, regime-robustness indicators, grouped LightGBM interpretability summaries, supplementary tables, and publication figures.

## Repository structure

```text
woodsciforecast_repo/
├── data/
│   ├── raw/                 # Master input workbook
│   ├── processed/           # Curated analysis workbooks and supporting outputs
│   └── metadata/            # Split definitions and variable dictionary
├── docs/                    # Data and code availability notes
├── outputs/
│   ├── figures/             # Generated figure outputs
│   └── tables/              # Generated supplementary table outputs
├── src/                     # Reproducible Python modules and scripts
└── requirements.txt
```

## Scope of reproduction

The repository is designed to reproduce the reported benchmark outputs from the curated analysis files included in `data/processed/`. It is not presented as a raw-data harvesting pipeline from external statistical portals. The official-data preparation steps are documented in the manuscript and supporting notes, and the curated benchmark workbooks are provided for transparent inspection.

## Main design choices

- Forecasts are evaluated under fixed chronological split rules.
- Pooled LightGBM uses only information available before the forecast block.
- Fallback-contingent LightGBM outputs are retained for auditability but are not interpreted as valid pooled-learning wins.
- MASE is the primary metric; MAE, RMSE, and sMAPE are reported as supporting metrics.
- Regime-robustness indicators quantify changes in monitoring performance across pre-shock, shock, and post-shock windows.
- Grouped SHAP summaries report feature-attribution patterns for execution-valid LightGBM runs.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

The focused recurrent challenger requires TensorFlow. Install `requirements-recurrent.txt` only if rerunning that component.

```bash
pip install -r requirements-recurrent.txt
```

## Main entry points

Regenerate all manuscript and supplementary figures:

```bash
python src/run_all_figures.py
```

Export supplementary tables from the diagnostics workbook:

```bash
python src/build_supplementary_tables.py
```

Rebuild diagnostics from the main benchmark workbook:

```bash
python src/build_diagnostics.py
```

The benchmark and recurrent-challenger scripts are provided in `src/` for transparent inspection and rerunning when the expected input workbooks are available.
