from __future__ import annotations

import pandas as pd

from config import DIAGNOSTICS_FILE, TABLE_DIR

SHEETS = [
    "Table_A1_execution_validity",
    "Table_A2_robust_metrics",
    "Table_A3_no_fallback_winners",
    "Table_A4_no_fallback_means",
    "Table_A5_stat_tests",
    "Table_A6_regime_robustness",
]


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for sheet in SHEETS:
        data = pd.read_excel(DIAGNOSTICS_FILE, sheet_name=sheet)
        data.to_csv(TABLE_DIR / f"{sheet}.csv", index=False)


if __name__ == "__main__":
    main()
