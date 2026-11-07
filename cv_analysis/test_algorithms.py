"""Run: python tests/test_algorithms.py (no pytest dependency)."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from market import sample_histories, population_revenues, PRICE_GRID
from acrp_core import batch_main, LAMBDAS, ETA_GRID, ALPHA_GRID
from competitors import fixed_share_fast, swfi_ucb_joint_fv, FS_ALPHA, FS_ETA, BETA_GRID
from fixedshare_reference import fixed_share_one

rng=np.random.default_rng(20051020)
for cv in [.2,.4,.5]:
    hist=sample_histories(rng,4,80,cv)
    R,I,C,W,FV=batch_main(hist,PRICE_GRID,.042,10,LAMBDAS,ETA_GRID,ALPHA_GRID)
    pop=population_revenues(PRICE_GRID,cv)
    for j in range(4):
        rr=R[j]
        assert np.all(rr[:-1]>=0)
        assert np.all(rr[:-1]<=PRICE_GRID+1e-10)
        z=1-rr/5
        v1,a,b=fixed_share_one(z,pop,10,FS_ETA,FS_ALPHA)
        v2,c,d,_=fixed_share_fast(z,pop,10,FS_ETA,FS_ALPHA)
        assert (a,b)==(c,d), (cv,j,(a,b),(c,d))
        assert abs(v1-v2)<1e-9,(cv,j,v1,v2)
        k,h,beta,score=swfi_ucb_joint_fv(rr,PRICE_GRID,80,10,BETA_GRID)
        assert 0<=k<81 and 1<=h<=29
        assert np.isfinite(score) and beta in BETA_GRID
print('PASS: Fixed-Share exact parameter equality and revenues; UCB and ACRP sanity')
