#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Figures 1–3 ONLY from saved CSV results. No simulation, Word, or Figure 4."""
from pathlib import Path
import csv,json,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'
CSV=ROOT/'results'/'summary'
STYLE=json.loads((ROOT/'config'/'plot_style.json').read_text(encoding='utf-8'))
METHODS=['ACRP-FV','SW-FV','SW-FI-UCB-FV','Fixed-Share','Calibrated Full-history','Oracle-SW']
RHOS=[]  # Populated from experiment_config.json for the selected --data-root.

def load(name):
    with (CSV/name).open('r',newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def fmt_axis(ax, ylabel):
    ax.set_xticks(RHOS)
    ax.set_xticklabels([('0' if abs(v)<1e-14 else f'{v:+g}').replace('-', '−') for v in RHOS])
    ax.set_xlabel('Gaussian-copula dependence (latent ρ)')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y',color='#E4E8ED',linewidth=.65)
    ax.set_axisbelow(True)


def finish(fig,name):
    OUT.mkdir(exist_ok=True,parents=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=600,bbox_inches='tight')
    plt.close(fig)


def main():
    global OUT,CSV,RHOS
    pa=argparse.ArgumentParser()
    pa.add_argument('--data-root', type=str, default='results')
    a=pa.parse_args()
    CSV=ROOT/a.data_root/'summary'
    OUT=ROOT/'figures' if a.data_root=='results' else ROOT/a.data_root/'figures'
    meta_path=ROOT/a.data_root/'raw'/'experiment_config.json'
    meta=json.loads(meta_path.read_text(encoding='utf-8'))
    RHOS=sorted(float(r) for r in meta['rho_values'])
    plt.rcParams.update({'font.family':STYLE.get('font_family','DejaVu Sans'),
        'font.size':9.5,'axes.spines.top':False,'axes.spines.right':False,
        'figure.dpi':120,'legend.fontsize':8,'axes.labelsize':10.5})
    data=load('summary_rrl.csv')
    pairs=load('paired_differences.csv')
    wins=load('window_summary.csv')
    # Figure 1: all six methods, standard error-based 95% intervals.
    fig,ax=plt.subplots(figsize=(8.3,4.9))
    for m in METHODS:
        rows=sorted([r for r in data if r['method']==m],key=lambda x:float(x['rho']))
        s=STYLE['algorithms'][m]
        ax.errorbar([float(x['rho']) for x in rows],[float(x['mean_RRL_pct']) for x in rows],
            yerr=[1.96*float(x['se_RRL_pct']) for x in rows],
            color=s['color'],marker=s['marker'],linestyle=s['linestyle'],
            linewidth=s['linewidth'],markersize=5.3,capsize=2.8,elinewidth=.85,label=m)
    fmt_axis(ax,'Relative revenue loss (%)')
    ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.23))
    fig.tight_layout();finish(fig,'Fig1_six_methods_RRL_by_rho')
    # Figure 2: pairing preserves same test histories within each rho.
    fig,ax=plt.subplots(figsize=(7.6,4.5))
    for m in ['SW-FV','SW-FI-UCB-FV','Fixed-Share']:
        rows=sorted([r for r in pairs if r['comparator']==m],key=lambda x:float(x['rho']))
        s=STYLE['algorithms'][m]
        ax.errorbar([float(x['rho']) for x in rows],
            [float(x['mean_ACRP_minus_comp_pct']) for x in rows],
            yerr=[1.96*float(x['se_paired_pct']) for x in rows],
            color=s['color'],marker=s['marker'],linestyle=s['linestyle'],
            linewidth=s['linewidth'],capsize=3,markersize=5,label=f'ACRP-FV − {m}')
    ax.axhline(0,color='#666666',linewidth=1,linestyle=':')
    fmt_axis(ax,'Paired RRL difference (percentage points)')
    ax.legend(frameon=False,loc='best',fontsize=8)
    fig.tight_layout();finish(fig,'Fig2_paired_RRL_difference_by_rho')
    # Figure 3: window sizes; Oracle can pick h in [1,39].
    fig,ax=plt.subplots(figsize=(7.6,4.5))
    for m in ['ACRP-FV','SW-FV','SW-FI-UCB-FV','Oracle-SW']:
        rows=sorted([r for r in wins if r['method']==m],key=lambda x:float(x['rho']))
        s=STYLE['algorithms'][m]
        ax.plot([float(x['rho']) for x in rows],
            [float(x['mean_window']) for x in rows],
            color=s['color'],marker=s['marker'],linestyle=s['linestyle'],
            linewidth=s['linewidth'],markersize=5.3,label=m)
    fmt_axis(ax,'Historical window length (periods)')
    ax.set_ylim(0,42)
    ax.legend(frameon=False,ncol=2,loc='best')
    fig.tight_layout();finish(fig,'Fig3_window_length_by_rho')
    print('Finished: 3 figures x 3 formats (png/pdf/svg)')

if __name__=='__main__':main()
