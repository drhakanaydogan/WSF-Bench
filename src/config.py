from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

MAIN_RESULTS_FILE = PROCESSED_DIR / "woodsciforecast_leaksafe_main_results_v1.xlsx"
TRADE_RECURRENT_RESULTS_FILE = PROCESSED_DIR / "woodsciforecast_leaksafe_trade_recurrent_results_v1.xlsx"
DIAGNOSTICS_FILE = PROCESSED_DIR / "woodsciforecast_diagnostics_v1.xlsx"
INTERPRETABILITY_FILE = PROCESSED_DIR / "woodsciforecast_leaksafe_lgbm_interpretability_v1.xlsx"
DESCRIPTIVE_FILE = PROCESSED_DIR / "woodsciforecast_descriptive_package_v1.xlsx"
PROTOCOL_FILE = PROCESSED_DIR / "woodsciforecast_benchmark_protocol_v1.xlsx"

PRODUCTION_SPLITS = [
    ("S1", "2015-01-01", "2019-12-01"),
    ("S2", "2020-01-01", "2021-12-01"),
    ("S3", "2022-01-01", "2024-12-01"),
]

TRADE_SPLITS = [
    ("T1", "2014-01-01", "2019-12-01"),
    ("T2", "2020-01-01", "2021-12-01"),
    ("T3", "2022-01-01", "2024-12-01"),
]

PRODUCTION_TARGETS = {
    "ln_sts_c16_sa_idx2021_100": "Production C16",
    "ln_sts_c31_sa_idx2021_100": "Production C31",
}

TRADE_TARGETS = {
    "ln_trade_world_export_eur_hs4sum": "Trade export",
    "ln_trade_world_import_eur_hs4sum": "Trade import",
}

LIGHTGBM_PARAMS = {
    "n_estimators": 120,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "objective": "regression",
    "random_state": 42,
    "verbose": -1,
}
