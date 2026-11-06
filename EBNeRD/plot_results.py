"""Plot observed hourly click-through rate and the 8--48h exponential fit.

Uses only locally computed aggregates and fit summaries. No Parquet needed.
The real-data results directory is ignored by Git; do not publish its contents
without confirming the dataset owner's permission.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent

# Publication-style presentation settings (change these only to alter visuals)
FIG_SIZE = (7.16, 3.0)
FONT_FAMILY = ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif']
FONT_SIZE = 9
AXIS_LABEL_SIZE = 10
TICK_SIZE = 9
LEGEND_SIZE = 9
OBS_COLOR = '#667B95'
FIT_COLOR = '#C95763'
GRID_COLOR = '#D0D5DB'
OBS_LINEWIDTH = 1.6
FIT_LINEWIDTH = 1.8
MARKER_SIZE = 3
MARK_EVERY = 2
FIT_LINESTYLE = '--'
DPI = 600


def load_results(folder: Path):
    p = folder / 'hourly_ctr_48h.csv'
    j = folder / 'fit_results.json'
    if not p.exists() or not j.exists():
        raise FileNotFoundError(
            f'Missing {p} or {j}. Run python run.py with licensed data, '
            'or run python demo.py for synthetic input.'
        )
    with p.open('r', newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 48 or [int(row['age_bin_start_h']) for row in rows] != list(range(48)):
        raise ValueError('Expected 48 hourly rows, age_bin_start_h = 0..47')
    with j.open('r', encoding='utf-8') as f:
        summary = json.load(f)
    lamfit = summary['fits']['exp_8_48']
    h = np.asarray([float(row['age_bin_start_h']) for row in rows])
    observed = np.asarray([float(row['ctr']) if row['ctr'] else np.nan for row in rows])
    return h, observed, lamfit, summary


def make_plot(folder: Path, output_dir: Path):
    h, observed, fit, summary = load_results(folder)
    is_demo = summary.get('metadata', {}).get('data_name') == 'SYNTHETIC DEMO'
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': FONT_FAMILY, 'font.size': FONT_SIZE,
        'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.unicode_minus': False,
    })
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    centers = h + 0.5
    keep = h >= 8
    fitted = fit['c'] * np.exp(-fit['lambda_per_hour'] * (centers[keep] - fit['start_h']))
    ax.plot(centers, observed, color=OBS_COLOR, linewidth=OBS_LINEWIDTH,
            marker='o', markersize=MARKER_SIZE, markevery=MARK_EVERY,
            label='Observed', zorder=5)
    ax.plot(centers[keep], fitted, color=FIT_COLOR, linewidth=FIT_LINEWIDTH,
            linestyle=FIT_LINESTYLE, label='Exponential fit', zorder=4)
    ax.set_xlim(0, 48)
    ax.set_xticks([0, 8, 16, 24, 32, 40, 48])
    ax.set_xlabel('News age (h)', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('Click-through rate (%)', fontsize=AXIS_LABEL_SIZE)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0, symbol=''))
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color=GRID_COLOR, linewidth=0.5, alpha=0.55)
    ax.legend(loc='upper right', frameon=False, fontsize=LEGEND_SIZE,
              handlelength=1.5, handletextpad=0.35, labelspacing=0.25)
    if is_demo:
        ax.set_title('SYNTHETIC DEMONSTRATION — not EB-NeRD measurements', fontsize=10)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ['pdf', 'png', 'svg']:
        fig.savefig(output_dir / f'ctr_decay_publication.{suffix}', dpi=DPI)
    plt.close(fig)
    print(f'Wrote publication-style figures to {output_dir} (synthetic={is_demo})')


def main():
    parser = argparse.ArgumentParser(description='Plot 8–48h fit from saved aggregates (no dataset required).')
    parser.add_argument('--input', type=Path, default=ROOT / 'results')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    make_plot(args.input, args.output if args.output else args.input)


if __name__ == '__main__':
    main()
