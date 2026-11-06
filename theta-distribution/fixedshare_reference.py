#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed-Share (Herbster & Warmuth, 1998): original off-diagonal sharing.
Each price is an expert, complete historical revenue vector is observed.
Full-feedback randomized final policy evaluated using true expected revenue.
History-only FV selects eta/alpha before target period.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def _pmean(prob, value):
    s=0.0
    for j in range(len(prob)): s += prob[j] * value[j]
    return s

@njit(cache=True)
def fixed_share_one(loss, true_reward, M, etas, alphas):
    """HW1998 Fig.1 EXACT off-diagonal share, tune parameters prequentially.
    Outputs expected true last-period revenue, eta index, alpha index.
    Last row in loss is dummy: NEVER used to train.
    """
    T,K=loss.shape
    best=-1e100; bestval=0.; bi=-1; bj=-1
    for i in range(len(etas)):
        for j in range(len(alphas)):
            eta=etas[i]; alpha=alphas[j]
            p=np.full(K,1.0/K)
            val=0.
            for t in range(T-1):
                if t>=T-1-M:
                    val += 5. * (1. - _pmean(p,loss[t])) / M
                zmax=-1e100
                for k in range(K):
                    v=np.log(max(p[k],1e-300)) - eta*loss[t,k]
                    if v>zmax:zmax=v
                q=np.empty(K); total=0.
                for k in range(K):
                    q[k]=np.exp(np.log(max(p[k],1e-300))-eta*loss[t,k]-zmax)
                    total+=q[k]
                for k in range(K):
                    q[k]/=total
                    # HW off-diagonal, NOT alpha/K mixture.
                    p[k]=(1.-alpha)*q[k]+alpha*(1.-q[k])/(K-1)
            if val>best+1e-13:
                best=val; bi=i; bj=j;bestval=_pmean(p,true_reward)
    return bestval,bi,bj

