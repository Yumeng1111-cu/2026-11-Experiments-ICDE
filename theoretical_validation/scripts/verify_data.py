"""Validate CSV summaries produced by this repository (after running generation)."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "experiment1_mismatch.csv": ("sigma_theta", "historical_distance_d", "population_mismatch"),
    "experiment2_tradeoff.csv": ("sigma_theta", "method", "memory_parameter", "relative_revenue_loss_percent"),
    "experiment3_heterogeneity.csv": ("sigma_theta", "n", "h_revenue", "h_bound", "lambda_revenue", "lambda_bound"),
    "experiment3_sample_size.csv": ("sigma_theta", "n", "h_revenue", "h_bound", "lambda_revenue", "lambda_bound"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("formal", "quick"), default="formal")
    mode = parser.parse_args().mode
    for name, fields in FILES.items():
        path = ROOT / "outputs" / "data" / mode / name
        if not path.is_file():
            raise SystemExit(f"Missing {path.relative_to(ROOT)}; first run python reproduce.py "
                             + ("--quick" if mode == "quick" else ""))
        df = pd.read_csv(path)
        assert not df.empty, f"Empty: {name}"
        assert set(fields).issubset(df.columns), f"Missing columns: {name}"
        for col in fields:
            if col == "method":
                assert df[col].notna().all()
            else:
                assert np.isfinite(pd.to_numeric(df[col])).all(), f"Non-finite {col} in {name}"
        if name == "experiment2_tradeoff.csv":
            assert set(df.method.unique()) == {"SW", "EW"}, "Missing SW/EW"
        print(f"PASS {name}: {len(df)} rows")
    print(f"All {mode} CSV summaries are valid.")


if __name__ == "__main__":
    main()
