#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CSV+Excel builder independent of MC simulation. Uses artifact_tool for Excel."""
from pathlib import Path
import csv, argparse, json, numpy as np
from scipy.stats import norm

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'results'/'raw'; OUT=ROOT/'results'/'summary'
METHODS=['ACRP-FV','SW-FV','SW-FI-UCB-FV','Fixed-Share','Calibrated Full-history','Oracle-SW']

def read(name):
    with (RAW/name).open('r',newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write(path, data):
    if not data: raise ValueError('No data for output '+str(path))
    path.parent.mkdir(exist_ok=True,parents=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def rrl_stats(x):
    a=np.asarray(x,dtype=float);n=len(a)
    m=float(a.mean());sd=float(a.std(ddof=1));se=sd/np.sqrt(n)
    return m,sd,se,m-1.96*se,m+1.96*se

def make_excel(data, mc):
    from artifact_tool import Workbook,SpreadsheetFile
    book=Workbook.create()
    def typed(c):
        if c is None or c=='': return None
        if isinstance(c,(float,int,bool)): return c
        try:
            return float(c) if '.' in str(c) or 'e' in str(c).lower() or str(c).startswith('-') else int(c)
        except (ValueError,TypeError):return str(c)
    for name,rows in data.items():
        s=book.worksheets.add(name)
        columns=list(rows[0].keys())
        matrix=[columns]+[[typed(row.get(k)) for k in columns] for row in rows]
        s.get_range_by_indexes(0,0,len(matrix),len(columns)).values=matrix
        top=s.get_range_by_indexes(0,0,1,len(columns))
        top.format={'fill':'#30465E','font':{'bold':True,'color':'#FFFFFF'},'row_height':30,'wrap_text':True}
        s.freeze_panes.freeze_rows(1)
        for i,k in enumerate(columns):
            s.get_range_by_indexes(0,i,len(matrix),1).format.column_width=(28 if k in ('method','comparator') else 20)
        if len(matrix)>1:
            s.get_range_by_indexes(1,0,len(matrix)-1,len(columns)).format.row_height=21
    target=OUT/f'correlation_{mc}mc_results.xlsx'
    SpreadsheetFile.export_xlsx(book).save(str(target))
    print('EXCEL created',target,flush=True)

def main():
    global RAW,OUT
    pa=argparse.ArgumentParser()
    pa.add_argument('--data-root',type=str,default='results')
    a=pa.parse_args()
    RAW=ROOT/a.data_root/'raw';OUT=ROOT/a.data_root/'summary'
    meta_path=RAW/'experiment_config.json'
    if not meta_path.exists():
        raise FileNotFoundError(f'Cannot find {meta_path}. Rerun simulation with the same --data-root.')
    meta=json.loads(meta_path.read_text(encoding='utf-8'))
    RHO=tuple(sorted(float(r) for r in meta['rho_values']))
    if not RHO:
        raise ValueError('experiment_config.json contains an empty rho_values list')
    results=read('all_methods_by_rho.csv');params=read('selected_parameters_by_rho.csv')
    oracle=read('rho_market_details.csv')
    assert len(results)==6*len(params)
    raw_rhos=set(float(r['rho']) for r in results)
    if raw_rhos != set(RHO):
        raise ValueError(f'Inconsistent rho values: config={RHO}, raw CSV={sorted(raw_rhos)}')
    summary=[];pairs=[];win=[];par=[]
    for rho in RHO:
        gr=[r for r in results if float(r['rho'])==rho]
        gp=[p for p in params if float(p['rho'])==rho]
        n=len(gp)
        if n<2 or len(gr)!=6*n:
            raise RuntimeError(f'Insufficient/mismatched MC rows for rho={rho}: parameters={n}, algorithm rows={len(gr)}')
        by_method={name:{int(x['mc_id']):float(x['rrl_pct']) for x in gr if x['method']==name} for name in METHODS}
        assert all(len(by_method[name])==n for name in METHODS)
        for name in METHODS:
            g=[r for r in gr if r['method']==name]
            mean,sd,se,lo,hi=rrl_stats([float(r['rrl_pct']) for r in g])
            summary.append({'rho':rho,'method':name,'MC':n,'mean_RRL_pct':mean,
                'sd_RRL_pct':sd,'se_RRL_pct':se,'ci95_lower_pct':lo,'ci95_upper_pct':hi,
                'mean_expected_revenue':float(np.mean([float(r['expected_revenue']) for r in g])),
                'mean_price_or_expected_price':float(np.mean([float(r['price_or_expected_price']) for r in g]))})
        for name in METHODS[1:]:
            dd=[by_method['ACRP-FV'][mid]-by_method[name][mid] for mid in sorted(by_method[name])]
            m,sd,se,lo,hi=rrl_stats(dd)
            p=2*norm.sf(abs(m/se)) if se>0 else (0. if m else 1.)
            pairs.append({'rho':rho,'comparator':name,'MC':n,'mean_ACRP_minus_comp_pct':m,
                'se_paired_pct':se,'ci95_lower':lo,'ci95_upper':hi,
                'normal_p_two_sided':p,'CI_excludes_zero':int(lo>0 or hi<0)})
        for col,method in [('ACRP_window','ACRP-FV'),('SW_window','SW-FV'),('UCB_window','SW-FI-UCB-FV'),('Oracle_SW_fixed_window','Oracle-SW')]:
            ar=np.asarray([float(x[col]) for x in gp])
            win.append({'rho':rho,'method':method,'mean_window':float(ar.mean()),
                        'sd_window':float(ar.std(ddof=1)), 'median_window':float(np.median(ar)),
                        'p10':float(np.quantile(ar,.1)),'p90':float(np.quantile(ar,.9))})
        par.append({'rho':rho,'UCB_beta0_pct':100*sum(float(x['UCB_beta'])==0 for x in gp)/n,
            'FS_alpha0_pct':100*sum(float(x['FS_alpha'])==0 for x in gp)/n,
            'UCB_beta_mean':float(np.mean([float(x['UCB_beta']) for x in gp])),
            'FS_eta_median':float(np.median([float(x['FS_eta']) for x in gp])),
            'FS_alpha_median':float(np.median([float(x['FS_alpha']) for x in gp]))})
    wide=[]
    for name in METHODS:
        v={'method':name}
        for rho in RHO:
            v[f'rho={rho:+.1f}']=next(row['mean_RRL_pct'] for row in summary if row['rho']==rho and row['method']==name)
        wide.append(v)
    for name,data in [('summary_rrl.csv',summary),('summary_rrl_wide.csv',wide),
        ('paired_differences.csv',pairs),('window_summary.csv',win),
        ('parameter_summary.csv',par),('market_details.csv',oracle)]:
        write(OUT/name,data)
    try:
        make_excel({'RRL_宽表':wide,'RRL_统计':summary,'配对差异':pairs,'窗口统计':win,
                    '参数统计':par,'联合分布诊断':oracle}, len(params)//len(RHO))
    except ImportError:
        print('INFO: artifact_tool not installed; all CSVs created, bundled Excel remains available.')
    print('Summary complete',len(results),'method-level observations')

if __name__=='__main__': main()
