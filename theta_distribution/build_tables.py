"""Generate summary CSV and formatted Excel workbook without rerunning the Monte Carlo."""
import csv,argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import norm
ROOT=Path(__file__).resolve().parent
METHODS=['ACRP-FV','SW-FV','SW-FI-UCB-FV','Fixed-Share','Calibrated Full-history','Oracle-SW']
DISTS=['gamma','lognormal','two-point']

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(path,rows):
    if not rows: return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stat(vals):
    a=np.asarray(vals,dtype=float);mean=a.mean();sd=a.std(ddof=1);se=sd/np.sqrt(len(a))
    return dict(mean=float(mean),sd=float(sd),se=float(se),low=float(mean-1.96*se),high=float(mean+1.96*se))
def export_excel(out, tables,mc):
    from artifact_tool import Workbook,SpreadsheetFile
    wb=Workbook.create()
    def num(x):
        if x is None or x=='':return None
        if isinstance(x,(int,float)):return x
        try:return float(x) if '.' in str(x) or 'e' in str(x).lower() else int(x)
        except (ValueError,TypeError):return str(x)
    for name,rows in tables.items():
        if not rows:continue
        sh=wb.worksheets.add(name)
        cols=list(rows[0]);matrix=[cols]+[[num(r.get(k)) for k in cols] for r in rows]
        sh.get_range_by_indexes(0,0,len(matrix),len(cols)).values=matrix
        sh.get_range_by_indexes(0,0,1,len(cols)).format={
            'fill':'#34475E','font':{'bold':True,'color':'#FFFFFF'},'row_height':29,'wrap_text':True}
        sh.freeze_panes.freeze_rows(1)
        for col,h in enumerate(cols):
            sh.get_range_by_indexes(0,col,len(matrix),1).format.column_width = 29 if h in ('method','comparator','theta_parameters') else 19
        if len(matrix)>1:sh.get_range_by_indexes(1,0,len(matrix)-1,len(cols)).format.row_height=20
    path=out/f'theta_distribution_{mc}MC.xlsx'
    SpreadsheetFile.export_xlsx(wb).save(str(path))
    print('Excel saved:',path,flush=True)

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--data-root',default='runs/distribution_cv04_pilot');args=pa.parse_args()
    root=ROOT/args.data_root;raw=root/'raw';out=root/'summary';out.mkdir(parents=True,exist_ok=True)
    allrows=read(raw/'all_methods_by_distribution.csv');par=read(raw/'selected_parameters_by_distribution.csv')
    info=read(raw/'market_details.csv')
    config=json.loads((raw/'experiment_config.json').read_text(encoding='utf-8'))
    dists=config['distributions'];summary=[];paired=[];windows=[];hyper=[];wide=[]
    for d in dists:
        gr=[r for r in allrows if r['distribution']==d]
        gp=[r for r in par if r['distribution']==d]
        n=len(gp)
        assert len(gr)==6*n
        by_method={m:{int(x['mc_id']):float(x['rrl_pct']) for x in gr if x['method']==m} for m in METHODS}
        for m in METHODS:
            assert len(by_method[m])==n
            values=by_method[m]
            s=stat([values[i] for i in sorted(values)])
            g=[r for r in gr if r['method']==m]
            summary.append({'distribution':d,'CV':config['CV'],'method':m,'MC':n,
                'mean_RRL_pct':s['mean'],'sd_RRL_pct':s['sd'],'se_RRL_pct':s['se'],
                'ci95_lower_pct':s['low'],'ci95_upper_pct':s['high'],
                'mean_revenue':float(np.mean([float(x['expected_revenue']) for x in g])),
                'mean_price':float(np.mean([float(x['selected_price_or_mean_price']) for x in g]))})
        for m in METHODS[1:]:
            x=by_method['ACRP-FV'];y=by_method[m]
            s=stat([x[j]-y[j] for j in sorted(x)])
            p=2*norm.sf(abs(s['mean']/s['se'])) if s['se']>0 else (0. if s['mean'] else 1.)
            paired.append({'distribution':d,'comparator':m,'MC':n,
                'ACRP_minus_comparator_RRL_pp':s['mean'],'paired_se':s['se'],
                'ci95_lower':s['low'],'ci95_upper':s['high'],'p_normal_two_sided':p,
                'ci_excludes_zero':int(s['low']>0 or s['high']<0)})
        for col,name in [('ACRP_window','ACRP-FV'),('SW_window','SW-FV'),('UCB_window','SW-FI-UCB-FV'),('Oracle_SW_fixed_window','Oracle-SW')]:
            a=np.array([float(g[col]) for g in gp]);windows.append({
                'distribution':d,'method':name,'mean_h':float(a.mean()),
                'sd_h':float(a.std(ddof=1)),'median_h':float(np.median(a)),
                'p10_h':float(np.quantile(a,.1)),'p90_h':float(np.quantile(a,.9))})
        hyper.append({'distribution':d,'UCB_beta0_pct':100*sum(float(g['UCB_beta'])==0 for g in gp)/n,
             'FS_alpha0_pct':100*sum(float(g['FS_alpha'])==0 for g in gp)/n,
             'UCB_beta_mean':float(np.mean([float(g['UCB_beta']) for g in gp])),
             'FS_eta_median':float(np.median([float(g['FS_eta']) for g in gp])),
             'FS_alpha_median':float(np.median([float(g['FS_alpha']) for g in gp]))})
    for m in METHODS:
        wide.append({'method':m,**{d:next(s['mean_RRL_pct'] for s in summary if s['distribution']==d and s['method']==m) for d in dists}})
    tables={'RRL_Wide':wide,'RRL_Statistics':summary,'Paired_Difference':paired,
        'Window_Statistics':windows,'Hyperparameter':hyper,'Market_Diagnostics':info}
    for name,rows in tables.items():write(out/(name+'.csv'),rows)
    try:export_excel(out,tables,len(par)//len(dists))
    except (ImportError,ModuleNotFoundError):print('artifact_tool unavailable: summary CSVs saved; Excel optional.',flush=True)
    print('Summary generated:',len(summary),'method×distribution records',flush=True)
if __name__=='__main__':main()
