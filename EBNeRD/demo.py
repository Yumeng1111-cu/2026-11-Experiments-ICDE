"""Entirely synthetic offline demonstration; does NOT contain EB-NeRD data."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from src.analysis import fit_model, save_csv, dump_json
from src.figures import create_figures

ROOT = Path(__file__).resolve().parent


def main():
    rng = np.random.default_rng(20261106)
    h = np.arange(48)
    # Explicitly fabricated exposure and click processes, for software testing only.
    exposure = np.rint(18000 * np.exp(-h / 15) + 1500).astype(int)
    early = 0.085 + 0.045 * (1 - np.exp(-h / 2.5))
    later = 0.12 * np.exp(-0.042 * (h - 8))
    click_probability = np.where(h < 8, early, later)
    clicks = rng.binomial(exposure, click_probability)
    fits = {
        'exp_0_48': fit_model(exposure, clicks, start=0),
        'exp_8_48': fit_model(exposure, clicks, start=8),
    }
    output = ROOT / 'results' / 'demo'
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output, exposure, clicks, fits)
    dump_json(output / 'fit_results.json', {
        'metadata': {'data_name': 'SYNTHETIC DEMO',
                     'warning': 'Entirely synthetic illustration; NOT EB-NeRD observations'},
        'fits': fits,
        'bootstrap': {'n_requested': 0, 'fits': {}},
    })
    create_figures(output, exposure, clicks, fits)
    print('SYNTHETIC demo created in:', output)
    print('Then run: python plot_results.py --input results/demo')


if __name__ == '__main__':
    main()
