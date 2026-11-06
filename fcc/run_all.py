#!/usr/bin/env python3
"""Run the FCC Auction 97 block-G valuation model-selection workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "src" / "fcc_g_model_selection.py"
RAW_ZIP = ROOT / "data" / "raw" / "97_341_all_files.zip"
PROCESSED_CSV = (
    ROOT / "data" / "processed" / "fcc_auction97_block_g_exit_bids.csv"
)
OUTPUT_DIR = ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce candidate-model comparison and the selected G-block model."
    )
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="Re-extract the 4,286 exit bids from the downloaded FCC ZIP before fitting.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the official FCC ZIP if the packaged copy is absent.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download and not RAW_ZIP.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "download_data.py")],
            check=True,
            cwd=ROOT,
        )

    command = [
        sys.executable,
        str(SCRIPT),
        "--output-dir",
        str(OUTPUT_DIR),
        "--seed",
        str(args.seed),
    ]
    if args.from_raw:
        if not RAW_ZIP.exists():
            raise FileNotFoundError(
                f"Raw FCC ZIP not found: {RAW_ZIP}. "
                "Run: python run_all.py --from-raw --download"
            )
        command.extend(["--zip", str(RAW_ZIP)])
    else:
        if not PROCESSED_CSV.exists():
            raise FileNotFoundError(
                f"Processed CSV not found: {PROCESSED_CSV}. Use --from-raw instead."
            )
        command.extend(["--input-csv", str(PROCESSED_CSV)])

    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
