"""Quick integrity check of a saved experiment; does not rerun simulations."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--data-root',default='runs/validation_cv04_500mc')
a=p.parse_args()
path=Path(a.data_root);path=path if path.is_absolute() else ROOT/path
raw=pd.read_csv(path/'raw/all_method_runs.csv')
par=pd.read_csv(path/'raw/selected_parameters.csv')
meta=pd.read_json(path/'experiment_config.json',typ='series')
errors=sorted(raw.relative_error.unique())
assert len(raw)==len(errors)*6*int(meta['mc'])
assert len(par)==len(errors)*int(meta['mc'])
assert 0 in errors
assert np.allclose(par.assumed_mean,.042*(1+par.relative_error))
assert np.all((par.ACRP_window>=1)&(par.ACRP_window<=29))
for m in ('SW-FV','SW-UCB-FV','Fixed-Share'):
    pvt=raw[raw.method==m].pivot(index='mc_id',columns='relative_error',values='rrl_pct')
    assert np.allclose(pvt.to_numpy(),pvt.iloc[:,0].to_numpy()[:,None],atol=1e-10),m
assert not raw.isna().any().any()
print('PASS: raw/parameters shape, parameter mapping, no missing values, baseline invariance')
