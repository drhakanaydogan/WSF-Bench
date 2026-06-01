from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_excel_sheet(path: str | Path, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(Path(path), sheet_name=sheet_name)


def write_excel_sheets(path: str | Path, sheets: dict[str, pd.DataFrame]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)
