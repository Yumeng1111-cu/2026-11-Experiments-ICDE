#!/usr/bin/env python3
"""Download the official FCC Auction 97 cumulative-results ZIP."""

from __future__ import annotations

import argparse
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "97_341_all_files.zip"
URLS = [
    "https://auctionresults.fcc.gov/Auction_97/Results/full/341/97_341_all_files.zip",
    "https://web.archive.org/web/20240913164133id_/https://auctionresults.fcc.gov/Auction_97/Results/full/341/97_341_all_files.zip",
]


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as target:
            shutil.copyfileobj(response, target, length=1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.force:
        print(f"Already present: {args.output}")
        return

    partial = args.output.with_suffix(args.output.suffix + ".part")
    errors: list[str] = []
    for url in URLS:
        try:
            print(f"Downloading: {url}")
            download(url, partial)
            if not zipfile.is_zipfile(partial):
                raise RuntimeError("Downloaded file is not a valid ZIP archive")
            partial.replace(args.output)
            print(f"Saved: {args.output}")
            return
        except (OSError, urllib.error.URLError) as exc:
            errors.append(f"{url}: {exc}")
            if partial.exists():
                partial.unlink()

    raise RuntimeError("All download sources failed:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
