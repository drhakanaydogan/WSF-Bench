from __future__ import annotations

from make_figure3_monthly_profiles import main as figure3
from make_figure4_main_comparison import main as figure4
from make_figure5_winner_map import main as figure5
from make_figure6_relative_trade_advantage import main as figure6
from make_figure_s1_execution_validity import main as figure_s1
from make_figure_s2_grouped_interpretability import main as figure_s2


def main() -> None:
    figure3()
    figure4()
    figure5()
    figure6()
    figure_s1()
    figure_s2()


if __name__ == "__main__":
    main()
