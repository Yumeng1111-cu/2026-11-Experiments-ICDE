from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from .analysis import predict


def _axes_style(ax, end):
    ax.set_xlim(-0.4, end + 0.3)
    ax.set_xlabel('Hours since article publication')
    ax.set_ylabel('Click-through rate (%)')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0, symbol=''))
    ax.grid(color='#d8e0e8', alpha=0.55, linewidth=0.8)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)


def create_figures(output: Path, exposure, clicks, fits):
    hours = len(exposure)
    h = np.arange(hours)
    x = h + .5
    obs = np.divide(clicks, exposure, out=np.full(hours, np.nan, dtype=float), where=exposure > 0)
    all_fit = fits['exp_0_48']
    late_fit = fits['exp_8_48']
    fig, axs = plt.subplots(1, 2, figsize=(13.3, 5), sharey=True)
    for ax in axs:
        _axes_style(ax, hours)
        ax.plot(x, obs, color='#194d6d', linewidth=2.0, marker='o', markersize=3.8, label='Observed hourly CTR')
    axs[0].plot(x, predict(all_fit, h), '--', lw=2.5, color='#d16b4c', label='Exponential fit')
    axs[0].set_title('(a) Full publication-age window: 0–48 h', fontsize=11)
    axs[0].text(.97, .98, f"λ = {all_fit['lambda_per_hour']:.4f} /h\nWeighted R² = {all_fit['weighted_r2']:.3f}",
                transform=axs[0].transAxes, va='top', ha='right', fontsize=9, bbox={'facecolor': 'white', 'alpha':.90, 'edgecolor':'#e0e0e0'})
    axs[1].axvspan(0, 8, color='#c3c9ce', alpha=.22)
    axs[1].axvline(8, color='#777f86', lw=1.0, ls=':')
    valid = h >= 8
    axs[1].plot(x[valid], predict(late_fit, h[valid]), '--', lw=2.5, color='#d16b4c', label='Exponential fit, age ≥ 8 h')
    axs[1].set_title('(b) Later publication-age window: 8–48 h', fontsize=11)
    axs[1].text(.97, .98, f"λ = {late_fit['lambda_per_hour']:.4f} /h\nWeighted R² = {late_fit['weighted_r2']:.3f}",
                transform=axs[1].transAxes, va='top', ha='right', fontsize=9, bbox={'facecolor': 'white', 'alpha':.90, 'edgecolor':'#e0e0e0'})
    for ax in axs:
        ax.legend(frameon=False, fontsize=9, loc='upper right', bbox_to_anchor=(.98, .83))
    fig.suptitle('EB-NeRD Small | Pooled CTR by age since publication', fontsize=14, y=1.02, fontweight='bold')
    fig.tight_layout(w_pad=2.3)
    fig.savefig(output / 'ctr_decay_actual_vs_exponential.png', dpi=220, bbox_inches='tight')
    fig.savefig(output / 'ctr_decay_actual_vs_exponential.svg', bbox_inches='tight')
    plt.close(fig)

