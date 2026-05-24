# WSF-Bench repository scaffold

This repository contains a cleaned Python implementation of the benchmark workflow used in the manuscript.

## Included components

- fixed benchmark splits for production and trade panels
- country-level baseline evaluation (Seasonal Naive and ETS)
- pooled LightGBM benchmark with execution-validity logging and fallback handling
- trade-only LSTM deep challenger
- manuscript-facing summary tables and reproducible Excel outputs

## Expected input files

Place the following files in `data/processed/` before running the scripts:

- `woodsciforecast_benchmark_protocol_v1.xlsx`
- `woodsciforecast_first_baseline_results_v1.xlsx`
- `woodsciforecast_second_round_results_v2.xlsx` (optional if re-running round 2)
- `woodsciforecast_round3_trade_deep_results_v1.xlsx` (optional if re-running round 3)

## Run scripts

From the project root:

```bash
python -m src.run_round2_benchmark
python -m src.run_round3_trade_deep
```

Generated outputs are written to `data/processed/` using the file names defined in `src/config.py`.


## Figure scripts

After placing the processed Excel files in `data/processed/`, the manuscript figures can be regenerated with:

```bash
python -m src.make_figure2_timeline
python -m src.make_figure3_monthly_profiles
python -m src.make_figure4_main_comparison
python -m src.make_figure5_winner_map
python -m src.make_figure6_relative_trade_advantage
```

Figures are written to `outputs/figures/` in PNG, TIFF, and PDF formats.

## Adding Excel files

Place your final Excel files manually in `data/processed/` using these exact names:

- `woodsciforecast_benchmark_protocol_v1.xlsx`
- `woodsciforecast_first_baseline_results_v1.xlsx`
- `woodsciforecast_second_round_results_v2.xlsx`
- `woodsciforecast_round3_trade_deep_results_v1.xlsx`


## Metadata and documentation

Additional benchmark metadata are provided in `data/metadata/`, and manuscript-facing availability notes are provided in `docs/`.

Repository archive release updated for the synchronization.
