"""Independent plotting from saved summary CSV; Figures 1-3 only. Supports --methods."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
STYLE=json.loads((ROOT/'config/plot_style.json').read_text(encoding='utf-8'))
METHODS=['ACRP-FV','SW-FV','SW-FI-UCB-FV','Fixed-Share','Calibrated Full-history','Oracle-SW']
PRETTY={'gamma':'Gamma','lognormal':'Lognormal','two-point':'Two-point'}

def load(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','pdf','svg'):
        fig.savefig(str(path)+'.'+suffix,dpi=600,bbox_inches='tight')
    plt.close(fig)

def plot_main(summary,dists,method_select,out,with_ci=True):
    fig,ax=plt.subplots(figsize=(8,4.8));xx=np.arange(len(dists))
    for method in method_select:
        data=[next(r for r in summary if r['method']==method and r['distribution']==d) for d in dists]
        s=STYLE['algorithms'][method]
        y=[float(r['mean_RRL_pct']) for r in data]
        se=[1.96*float(r['se_RRL_pct']) for r in data]
        if with_ci:
            ax.errorbar(xx,y,yerr=se,capsize=3,elinewidth=.85,color=s['color'],marker=s['marker'],
                linestyle=s['linestyle'],linewidth=s['linewidth'],markersize=6,label=method)
        else:
            ax.plot(xx,y,color=s['color'],marker=s['marker'],linestyle=s['linestyle'],
                linewidth=s['linewidth'],markersize=6,label=method)
    ax.set_xticks(xx,[PRETTY[d] for d in dists]);ax.set_ylabel('Relative revenue loss (%)')
    ax.set_xlabel('Time-sensitivity distribution');ax.grid(axis='y',color='#E4E8ED',linewidth=.65)
    ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.24))
    ax.set_axisbelow(True);fig.tight_layout();save(fig,out)

def plot_pair(paired,dists,out):
    fig,ax=plt.subplots(figsize=(7.8,4.5));xx=np.arange(len(dists))
    for method in ['SW-FV','SW-FI-UCB-FV','Fixed-Share']:
        data=[next(r for r in paired if r['distribution']==d and r['comparator']==method) for d in dists]
        s=STYLE['algorithms'][method]
        y=[float(r['ACRP_minus_comparator_RRL_pp']) for r in data]
        err=[1.96*float(r['paired_se']) for r in data]
        ax.errorbar(xx,y,yerr=err,capsize=3,color=s['color'],marker=s['marker'],
            linestyle=s['linestyle'],linewidth=s['linewidth'],label='ACRP-FV − '+method)
    ax.axhline(0,color='#777777',lw=1,ls=':')
    ax.set_xticks(xx,[PRETTY[d] for d in dists]);ax.set_xlabel('Time-sensitivity distribution')
    ax.set_ylabel('Paired RRL difference (percentage points)')
    ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',color='#E4E8ED',linewidth=.65)
    fig.tight_layout();save(fig,out)

def plot_window(rows,dists,out):
    fig,ax=plt.subplots(figsize=(7.8,4.5));xx=np.arange(len(dists))
    for method in ['ACRP-FV','SW-FV','SW-FI-UCB-FV','Oracle-SW']:
        ys=[float(next(x for x in rows if x['distribution']==d and x['method']==method)['mean_h']) for d in dists]
        s=STYLE['algorithms'][method]
        ax.plot(xx,ys,color=s['color'],marker=s['marker'],linestyle=s['linestyle'],
                linewidth=s['linewidth'],markersize=6,label=method)
    ax.set_xticks(xx,[PRETTY[d] for d in dists]);ax.set_xlabel('Time-sensitivity distribution')
    ax.set_ylabel('Historical window length (periods)');ax.set_ylim(0,42)
    ax.grid(axis='y',color='#E4E8ED',linewidth=.65);ax.legend(frameon=False,ncol=2)
    fig.tight_layout();save(fig,out)

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--data-root',default='runs/distribution_cv04_pilot')
    pa.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    pa.add_argument('--no-ci',action='store_true')
    pa.add_argument('--only-selected',action='store_true',help='Produce only figure of chosen methods (without overwriting original)')
    a=pa.parse_args()
    root=ROOT/a.data_root;meta=json.loads((root/'raw/experiment_config.json').read_text(encoding='utf-8'))
    dists=meta['distributions'];summary=load(root/'summary/RRL_Statistics.csv')
    out=root/'figures'
    plt.rcParams.update({'font.family':STYLE.get('font_family','DejaVu Sans'),'font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':10.5})
    if a.only_selected:
        slug='_'.join(x.replace(' ','').replace('-','') for x in a.methods)
        plot_main(summary,dists,a.methods,out/'selected'/('RRL_'+slug),not a.no_ci)
        print('Saved selected-only Figure 1. Original files untouched.')
        return
    plot_main(summary,dists,a.methods,out/'Fig1_RRL_by_theta_distribution',not a.no_ci)
    # A zoom of Figure 1 (not a new Figure 4): huge full-history loss can hide curves.
    if 'Calibrated Full-history' in a.methods and len(a.methods)>2:
        plot_main(summary,dists,[m for m in a.methods if m!='Calibrated Full-history'],
                  out/'Fig1_zoom_without_full_history',not a.no_ci)
    plot_pair(load(root/'summary/Paired_Difference.csv'),dists,out/'Fig2_paired_RRL_difference')
    plot_window(load(root/'summary/Window_Statistics.csv'),dists,out/'Fig3_window_length')
    print('Created Figures 1-3 in PNG, PDF, SVG.',flush=True)
if __name__=='__main__':main()
