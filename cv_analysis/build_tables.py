#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build analysis CSVs & optional Excel workbook from saved simulation data.
NO Monte Carlo work and NO figure generation. Safe to rerun at any time.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT=Path(__file__).resolve().parent
CVS=[.2,.3,.4,.5]
ORDER=['ACRP-FV','SW-FV','SW-FI-UCB-FV','Fixed-Share','Calibrated Full-history','Oracle-SW']

def make_workbook(output, tables):
    try:
        from artifact_tool import Workbook,SpreadsheetFile
    except ImportError:
        print('WARNING: artifact_tool not available; CSV tables remain usable. Excel is already supplied in precomputed results.')
        return False
    book=Workbook.create()
    for name,df in tables.items():
        sheet=book.worksheets.add(name[:31])
        matrix=[list(df.columns)]
        for tup in df.itertuples(index=False,name=None):
            matrix.append([float(x) if isinstance(x,(np.float32,np.float64)) and np.isfinite(x) else (int(x) if isinstance(x,(np.integer,)) else (None if pd.isna(x) else x)) for x in tup])
        nrows=len(matrix);ncols=len(df.columns)
        if nrows==0:continue
        rr=sheet.get_range_by_indexes(0,0,nrows,ncols)
        rr.values=matrix
        head=sheet.get_range_by_indexes(0,0,1,ncols)
        head.format={'fill':'#253858','font':{'bold':True,'color':'#FFFFFF'},'row_height':29,'wrap_text':True}
        sheet.freeze_panes.freeze_rows(1)
        for i,c in enumerate(df.columns):
            col=chr(65+i) if i<26 else chr(64+i//26)+chr(65+i%26)
            width=25 if c in ('method','comparator','label','indicator') else 18
            sheet.get_range(f'{col}:{col}').format.column_width=width
        if nrows>1:
            sheet.get_range_by_indexes(1,0,nrows-1,ncols).format.row_height=21
    output.parent.mkdir(parents=True,exist_ok=True)
    SpreadsheetFile.export_xlsx(book).save(str(output))
    print('EXCEL',output)
    return True

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='results/raw')
    ap.add_argument('--out',default='results/summary')
    args=ap.parse_args()
    indir=ROOT/args.data; outdir=ROOT/args.out
    outdir.mkdir(parents=True,exist_ok=True)
    rows=[];params=[];oracles=[]
    for cv in CVS:
        prefix=f'cv_{cv:.1f}'
        rr_file=indir/f'{prefix}_raw_results.csv'; param_file=indir/f'{prefix}_selected_parameters.csv'
        if not rr_file.exists() or not param_file.exists():
            raise FileNotFoundError(f'Required MC outputs missing for {prefix}: {rr_file}')
        rows.append(pd.read_csv(rr_file));params.append(pd.read_csv(param_file))
        oracles.append(json.loads((indir/f'{prefix}_metadata.json').read_text(encoding='utf-8')))
    rr=pd.concat(rows,ignore_index=True)
    pr=pd.concat(params,ignore_index=True)
    assert not rr.duplicated(['cv','mc_id','method']).any()
    for cv in CVS:
        assert rr[rr.cv==cv].groupby('mc_id').method.nunique().eq(6).all()
    rr.to_csv(indir/'all_5000MC_six_method_results.csv',index=False,float_format='%.12g')
    pr.to_csv(indir/'all_5000MC_selected_parameters.csv',index=False,float_format='%.12g')
    total=[]
    for (cv,m),d in rr.groupby(['cv','method'],sort=False):
        val=d.rrl_pct.to_numpy(); n=len(val)
        std=float(val.std(ddof=1)); se=std/np.sqrt(n)
        total.append(dict(cv=cv,method=m,MC=n,mean_RRL_pct=float(val.mean()),sd_RRL_pct=std,
          se_RRL_pct=se,ci95_lower_pct=float(val.mean()-1.96*se),ci95_upper_pct=float(val.mean()+1.96*se),
          mean_expected_revenue=float(d.expected_revenue.mean()),
          mean_price_or_expected_price=float(d.price_or_mean_price.mean())))
    summary=pd.DataFrame(total).sort_values(['cv','method'],key=lambda x:x.map({m:i for i,m in enumerate(ORDER)}) if x.name=='method' else x)
    summary.to_csv(outdir/'summary_rrl.csv',index=False,float_format='%.9g')
    wide=summary.pivot(index='method',columns='cv',values='mean_RRL_pct').reindex(ORDER).reset_index()
    wide.columns=['method']+[f'CV={c:.1f}' for c in CVS]
    wide.to_csv(outdir/'summary_rrl_wide.csv',index=False,float_format='%.9g')
    pairs=[]
    for cv in CVS:
        block=rr[rr.cv==cv].pivot(index='mc_id',columns='method',values='rrl_pct')
        for b in ORDER[1:]:
            v=(block['ACRP-FV']-block[b]).to_numpy(); n=len(v);mean=float(v.mean());se=float(v.std(ddof=1)/np.sqrt(n))
            z=mean/se if se>0 else 0.
            p=2*norm.sf(abs(z)) if se>0 else (0. if mean else 1.)
            pairs.append(dict(cv=cv,comparator=b,MC=n,mean_ACRP_minus_comp_pct=mean,
              se_diff_pct=se,ci95_lower=mean-1.96*se,ci95_upper=mean+1.96*se,
              two_sided_p_normal=p,CI_excludes_zero=bool(mean-1.96*se>0 or mean+1.96*se<0)))
    pairs=pd.DataFrame(pairs)
    # Holm correction within each CV across the three implementable, external/ablation comparators.
    pairs['holm_p_3']=np.nan
    for cv in CVS:
        eligible=pairs[(pairs.cv==cv)&(pairs.comparator.isin(['SW-FV','SW-FI-UCB-FV','Fixed-Share']))]
        indexes=eligible.sort_values('two_sided_p_normal').index.to_list()
        prev=0.
        for rank,i in enumerate(indexes):
            prev=max(prev, min(1.,(3-rank)*pairs.at[i,'two_sided_p_normal']))
            pairs.at[i,'holm_p_3']=prev
    pairs.to_csv(outdir/'paired_differences.csv',index=False,float_format='%.9g')
    windows=[]
    for cv,g in pr.groupby('cv'):
        for c,m in [('ACRP_window','ACRP-FV'),('SW_window','SW-FV'),('UCB_window','SW-FI-UCB-FV'),('Oracle_SW_fixed_window','Oracle-SW')]:
            data=g[c].to_numpy(dtype=float)
            windows.append(dict(cv=cv,method=m,MC=len(data),mean_window=float(data.mean()),sd_window=float(data.std(ddof=1)),
             median_window=float(np.median(data)),p10=float(np.quantile(data,.10)),p90=float(np.quantile(data,.90))))
    wins=pd.DataFrame(windows)
    wins.to_csv(outdir/'window_summary.csv',index=False,float_format='%.9g')
    par=[]; freq=[]
    for cv,g in pr.groupby('cv'):
        par.append(dict(cv=cv,MC=len(g),UCB_beta_zero_fraction=float((g.UCB_beta==0).mean()),
          UCB_beta_mean=float(g.UCB_beta.mean()),FS_alpha_zero_fraction=float((g.FS_alpha==0).mean()),
          FS_eta_max_fraction=float((g.FS_eta==1048576.).mean()),FS_eta_median=float(g.FS_eta.median()),
          FS_alpha_median=float(g.FS_alpha.median())))
        for col in ['UCB_beta','FS_eta','FS_alpha']:
            vals=g[col].value_counts().sort_index()
            for v,ct in vals.items():freq.append(dict(cv=cv,parameter=col,value=float(v),count=int(ct),fraction=float(ct/len(g))))
    par=pd.DataFrame(par); freq=pd.DataFrame(freq)
    par.to_csv(outdir/'parameter_summary.csv',index=False,float_format='%.9g')
    freq.to_csv(outdir/'parameter_frequencies.csv',index=False,float_format='%.9g')
    oms=pd.DataFrame([{'cv':x['cv'],'oracle_SW_window':x['oracle_SW_fixed_window'],
        'pstar_continuous':x['p_star'],'rstar_continuous':x['oracle_revenue'],
        'test_mc':x['mc_runs'],'independent_oracle_mc':x['oracle_training_runs'],
        'mean_theta':x['mean_theta'],'test_seed':x['seed'],'oracle_seed':x['oracle_seed']} for x in oracles])
    oms.to_csv(outdir/'oracle_summary.csv',index=False,float_format='%.9g')
    print('CSV summary complete',len(rr),'per-method MC results',len(pr),'selected-param rows')
    sheets={'RRL_宽表':wide,'RRL_统计表':summary,'ACRP_配对差异':pairs,
            '窗口统计':wins,'参数统计':par,'参数频率':freq,'Oracle_配置':oms}
    make_workbook(outdir/'formal_results.xlsx',sheets)

if __name__=='__main__':main()
