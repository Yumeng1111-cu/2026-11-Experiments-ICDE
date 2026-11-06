"""Recompute experiment CSVs from source, then draw all three figures.

Examples:
    python reproduce.py --quick --reps 2
    python reproduce.py                 # formal: 5000 test / 5000 pilot repetitions
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run low-repetition smoke test")
    parser.add_argument("--reps", type=int, default=None,
                        help="Repetitions per stream in quick mode (default from settings.py)")
    parser.add_argument("--force", action="store_true", help="Ignore existing simulation caches")
    args = parser.parse_args()
    if args.reps is not None and (not args.quick or args.reps < 1):
        parser.error("--reps requires --quick and a positive integer")

    # Change settings before importing experiment/core modules, which import constants.
    import settings
    import plot_settings
    settings.QUICK_MODE = args.quick
    if args.reps is not None:
        settings.QUICK_REPS = args.reps
    if args.force:
        settings.FORCE_RECOMPUTE = True
    plot_settings.PLOT_DATA_MODE = "quick" if args.quick else "formal"

    from experiment1_mismatch import main as gen1
    from experiment2_tradeoff import main as gen2
    from experiment3_comparative import main as gen3
    from plot_saved import draw

    for generate in (gen1, gen2, gen3):
        generate()
    for experiment in ("1", "2", "3"):
        draw(experiment, mode=plot_settings.PLOT_DATA_MODE)
    print("Finished: outputs/data/%s/*.csv and outputs/figures/*" % plot_settings.PLOT_DATA_MODE)


if __name__ == "__main__":
    main()
