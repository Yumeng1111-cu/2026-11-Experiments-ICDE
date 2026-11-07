"""Check reproducibility invariants for saved 1000-MC correlation results."""
import csv, json, math, argparse
from pathlib import Path
from collections import Counter

root=Path(__file__).resolve().parent
pa=argparse.ArgumentParser()
pa.add_argument('--data-root',type=str,default='results')
a=pa.parse_args()
raw=root/a.data_root/'raw'

def load(name):
    with (raw/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

meta=json.loads((raw/'experiment_config.json').read_text(encoding='utf-8'))
mc=meta['mc_per_rho']; train=meta['oracle_training_mc_per_rho']
rhos=[str(r) for r in meta['rho_values']]
methods=['ACRP-FV','SW-FV','SW-FI-UCB-FV','Fixed-Share','Calibrated Full-history','Oracle-SW']
rows=load('all_methods_by_rho.csv')
params=load('selected_parameters_by_rho.csv')
orec=load('oracle_training_windows.csv')
market=load('rho_market_details.csv')
assert len(rows)==len(rhos)*mc*len(methods),len(rows)
assert len(params)==len(rhos)*mc,len(params)
assert len(orec)==len(rhos)*39,len(orec)
assert len(market)==len(rhos),len(market)
assert all(v==mc for v in Counter((r['rho'],r['method']) for r in rows).values())
assert all(v==len(methods) for v in Counter((r['rho'],r['mc_id']) for r in rows).values())
for r in rows:
    loss=float(r['rrl_pct']);p=float(r['price_or_expected_price']);rv=float(r['expected_revenue'])
    assert all(map(math.isfinite,[loss,p,rv]))
    assert loss >= -1e-7,(r['rho'],r['method'],loss)
    assert 0.01<=p<=5.0
for r in params:
    assert 1<=int(r['ACRP_window'])<=29
    assert 1<=int(r['SW_window'])<=29
    assert 1<=int(r['UCB_window'])<=29
    assert 1<=int(r['Oracle_SW_fixed_window'])<=39
    assert float(r['UCB_beta']) in (0,.025,.05,.1,.2,.35,.5,.75,1,1.5,2)
assert meta['mean_theta']==.042 and meta['CV']>0
print(f'PASS {mc} test histories/rho, {train} independent Oracle training histories/rho; '
      f'{len(rows)} method rows, {len(params)} per-history param rows; all invariants.')
