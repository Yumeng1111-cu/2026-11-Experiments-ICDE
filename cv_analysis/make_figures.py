#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render Figures 1–3 independently from CSV; never generate Figure 4 or Word."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
D=ROOT/'results/summary'; FIG=ROOT/'figures'
STYLE=json.loads((ROOT/'config/plot_style.json').read_text())
NAMES=list(STYLE['algorithms'].keys())
plt.rcParams.update({'font.family':STYLE.get('font_family','DejaVu Sans'),
 'font.size':9,'axes.labelsize':10,'legend.fontsize':8,
 'xtick.labelsize':9,'ytick.labelsize':9,'figure.dpi':120,
 'axes.spines.top':False,'axes.spines.right':False})

def finish(fig,slug):
    FIG.mkdir(exist_ok=True,parents=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(FIG/f'{slug}.{ext}',bbox_inches='tight',dpi=int(STYLE['figure_dpi']))
    plt.close(fig)

def prettify(ax):
    ax.grid(axis='y',color='#E4E8ED',lw=.6)
    ax.set_axisbelow(True)
    ax.set_xticks([.2,.3,.4,.5]);ax.set_xlabel('Time-sensitivity heterogeneity (CV)')

def plot_line(data,methods,slug,zoom=False):
    fig,ax=plt.subplots(figsize=(7.2,4.3))
    for name in methods:
        g=data[data.method==name].sort_values('cv')
        s=STYLE['algorithms'][name]
        ax.errorbar(g.cv,g.mean_RRL_pct,yerr=1.96*g.se_RRL_pct,
            label=name,color=s['color'],linestyle=s['linestyle'],marker=s['marker'],
            linewidth=s['linewidth'],markersize=5,capsize=2.8,elinewidth=.9)
    prettify(ax);ax.set_ylabel('Relative revenue loss (%)')
    if zoom: ax.set_ylim(bottom=0)
    ax.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.23))
    fig.tight_layout()
    finish(fig,slug)

def main():
    r=pd.read_csv(D/'summary_rrl.csv')
    pairs=pd.read_csv(D/'paired_differences.csv')
    wins=pd.read_csv(D/'window_summary.csv')
    plot_line(r,NAMES,'Fig1_six_methods_RRL')
    plot_line(r,[n for n in NAMES if n!='SW-FI-UCB-FV'],'Fig1b_detail_excluding_UCB')
    fig,ax=plt.subplots(figsize=(7.3,4.2))
    for name in ['SW-FV','SW-FI-UCB-FV','Fixed-Share']:
        g=pairs[pairs.comparator==name].sort_values('cv');s=STYLE['algorithms'][name]
        ax.errorbar(g.cv,g.mean_ACRP_minus_comp_pct,
          yerr=[g.mean_ACRP_minus_comp_pct-g.ci95_lower,g.ci95_upper-g.mean_ACRP_minus_comp_pct],
          label=f'ACRP-FV − {name}',color=s['color'],marker=s['marker'],
          linestyle=s['linestyle'],linewidth=s['linewidth'],capsize=3)
    ax.axhline(0,color='#444444',linestyle=':',lw=1)
    prettify(ax);ax.set_ylabel('Paired RRL difference (percentage points)')
    ax.legend(frameon=False,ncol=1,loc='best');fig.tight_layout();finish(fig,'Fig2_paired_RRL_difference')
    fig,ax=plt.subplots(figsize=(7.2,4.1))
    for name in ['ACRP-FV','SW-FV','SW-FI-UCB-FV','Oracle-SW']:
        g=wins[wins.method==name].sort_values('cv');s=STYLE['algorithms'][name]
        ax.plot(g.cv,g.mean_window, label=name, color=s['color'],marker=s['marker'],
                linestyle=s['linestyle'],linewidth=s['linewidth'],markersize=6)
    prettify(ax);ax.set_ylabel('Selected historical window length')
    ax.legend(frameon=False,ncol=2);fig.tight_layout();finish(fig,'Fig3_window_selection')
    print('Images exported to',FIG)
if __name__=='__main__':main()
