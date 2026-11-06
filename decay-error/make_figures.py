#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Only reads summary CSVs; never runs simulations or creates Word."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
STYLE={
'ACRP-FV':('#C84C63','D','-'),
'SW-FV':('#E6B84F','^','-'),
'SW-UCB-FV':('#3977AE','o','-'),
'Fixed-Share':('#956943','v','-'),
'Calibrated Full-history':('#2C9D80','s','-'),
'Oracle-SW':('#765AA5','P','--')}

def plot(df,out,selected,zoom=False):
    fig,ax=plt.subplots(figsize=(7.2,4.05))
    for m in selected:
        g=df[df.method==m].sort_values('relative_error')
        col,mark,ls=STYLE[m]
        x=g.relative_error.to_numpy()*100
        y=g.mean_rrl_pct.to_numpy()
        ci=1.96*g.se_pct.to_numpy()
        ax.errorbar(x,y,yerr=ci,color=col,marker=mark,linestyle=ls,
            linewidth=2.6 if m=='ACRP-FV' else 1.8,
            markersize=6 if m=='ACRP-FV' else 5,elinewidth=.8,
            capsize=2.1,label=m)
    ax.axvline(0,color='#999999',linestyle=':',linewidth=1,zorder=0)
    ax.set_xlabel('Relative error in mean time sensitivity (%)',weight='bold')
    ax.set_ylabel('Relative revenue loss (%)',weight='bold')
    ax.grid(axis='y',alpha=.23,color='#BBBBBB',linewidth=.6)
    for side in ['top','right']:ax.spines[side].set_visible(False)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=9)
    ax.legend(ncol=3,loc='lower center',bbox_to_anchor=(.5,1.01),
            fontsize=8,frameon=False,handlelength=2.5,columnspacing=1.1)
    fig.subplots_adjust(left=.115,right=.98,bottom=.18,top=.80)
    for ext in ('png','pdf','svg'):
        fig.savefig(out.with_suffix('.'+ext),dpi=600,facecolor='white',bbox_inches='tight')
    plt.close(fig)

def windows(df,out):
    fig,ax=plt.subplots(figsize=(6.7,3.7))
    g=df.sort_values('relative_error')
    ax.plot(g.relative_error*100,g.mean_ACRP_window,marker='D',color='#C84C63',lw=2.5)
    ax.set_xlabel('Relative error in mean time sensitivity (%)',weight='bold')
    ax.set_ylabel('Mean selected ACRP window',weight='bold')
    ax.grid(axis='y',alpha=.23)
    for s in ['top','right']:ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ('png','pdf','svg'):
        fig.savefig(out.with_suffix('.'+ext),dpi=600,facecolor='white',bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',default='runs/validation_cv04_500mc');ap.add_argument('--methods',nargs='+',default=list(STYLE))
    args=ap.parse_args()
    bad=set(args.methods)-set(STYLE)
    if bad:raise ValueError('Unknown methods: '+str(bad))
    rt=Path(args.data_root);rt=rt if rt.is_absolute() else ROOT/rt
    d=pd.read_csv(rt/'summary/summary_rrl.csv')
    f=rt/'figures';f.mkdir(exist_ok=True)
    plot(d,f/'Fig1_all_methods',args.methods)
    keep=[m for m in args.methods if m!='Calibrated Full-history']
    if keep:plot(d,f/'Fig1b_without_full_history',keep,True)
    w=pd.read_csv(rt/'summary/window_summary.csv')
    windows(w,f/'Fig2_ACRP_window')
    print('Saved images under',f)
