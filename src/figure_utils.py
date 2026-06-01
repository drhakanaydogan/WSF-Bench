from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def save_figure(fig, output_path: str | Path, dpi: int = 600, extra_formats: Iterable[str] = ("tiff", "pdf")) -> None:
    """Save a Matplotlib figure in PNG plus optional archival formats.

    Figure scripts are responsible for layout control before calling this helper.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    for fmt in extra_formats:
        fig.savefig(output_path.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
