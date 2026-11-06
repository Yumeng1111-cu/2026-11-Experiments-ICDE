
"""Plot experiments A, B, and C using saved CSV data only.

No Monte Carlo simulation is performed.
"""

from __future__ import annotations

import plot_settings as S
from plot_saved import draw


def main():
    print(f"Plot data mode: {S.PLOT_DATA_MODE}")

    for experiment in ("A", "B", "C"):
        print(f"\nPlotting experiment {experiment}...")
        draw(experiment, mode=S.PLOT_DATA_MODE)

    print("\nAll three figures generated successfully.")


if __name__ == "__main__":
    main()
