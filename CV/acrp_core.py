#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACRP/SW-FV and fixed-window prices, shared discrete-grid causal routines.
Extracted without changing the core finite-price argmax/forward-validation logic
from the previous common-grid experiment. No data-generating model is embedded.
"""
import numpy as np
from numba import njit
LAMBDAS = np.arange(0.0,0.40000001,0.005)
ETA_GRID = np.array([1.,4.,16.,64.])
ALPHA_GRID = np.array([0.,.02,.08])
PRICE_GRID = np.geomspace(.01,5.,81)

@njit(cache=True)
def lower_bound(a, x):
    lo=0; hi=len(a)
    while lo<hi:
        m=(lo+hi)//2
        if a[m]<x: lo=m+1
        else: hi=m
    return lo

@njit(cache=True)
def get_period_reward(sorted_logv, logprices, prices):
    T,n=sorted_logv.shape
    K=len(prices)
    r=np.zeros((T,K))
    for s in range(T):
        for k in range(K):
            r[s,k] = prices[k] * (n-lower_bound(sorted_logv[s],logprices[k])) / n
    return r

@njit(cache=True)
def choose_empirical_price(counts, prices):
    bestidx=0; bestval=-1.0
    for k in range(len(prices)):
        cur=prices[k]*counts[k]
        if cur>bestval+1e-14:
            bestval=cur;bestidx=k
    return bestidx

@njit(cache=True)
def solve_fv(sorted_logv, prices, logprices, mean_theta, val_M, lam_grid, period_r):
    """Return grid-price indices for ACRP-FV/SW-FV/calibrated full and all oracle params."""
    T,n=sorted_logv.shape; K=len(prices)
    H=T-1-val_M
    valcal=np.zeros(H); valraw=np.zeros(H)
    current_calidx=np.zeros(T-1,np.int32)
    current_rawidx=np.zeros(T-1,np.int32)
    full_idx=0
    count_current=np.zeros((T-1,K))
    for target in range(T-1-val_M,T):  # 0-based: validation T-1-M..T-2, target T-1
        cumraw=np.zeros(K); cumcal=np.zeros(K)
        for h in range(1,target+1):
            old=target-h
            if old<0: break
            for k in range(K):
                cumraw[k]+=n-lower_bound(sorted_logv[old],logprices[k])
                cumcal[k]+=n-lower_bound(sorted_logv[old],logprices[k]+mean_theta*h)
            if target<T-1:
                if h<=H:
                    ipcal=choose_empirical_price(cumcal,prices)
                    ipraw=choose_empirical_price(cumraw,prices)
                    valcal[h-1]+=period_r[target,ipcal]
                    valraw[h-1]+=period_r[target,ipraw]
            else:
                current_calidx[h-1]=choose_empirical_price(cumcal,prices)
                current_rawidx[h-1]=choose_empirical_price(cumraw,prices)
                count_current[h-1,:]=cumcal
                if h==T-1: full_idx=choose_empirical_price(cumcal,prices)
    hcal=0; hraw=0
    for h in range(1,H):
        if valcal[h]>valcal[hcal]+1e-12: hcal=h
        if valraw[h]>valraw[hraw]+1e-12: hraw=h
    # hcal indexes 1,...,H
    calidx=current_calidx[hcal]; rawidx=current_rawidx[hraw]
    L=len(lam_grid)
    ew_idx=np.zeros(L,np.int32)
    for li in range(L):
        counts=np.zeros(K)
        norm=0.
        for h in range(1,T):
            factor=np.exp(-lam_grid[li]*(h-1))
            norm+=factor
            if h==1:
                for k in range(K): counts[k]+=factor*count_current[h-1,k]
            else:
                for k in range(K): counts[k]+=factor*(count_current[h-1,k]-count_current[h-2,k])
        ew_idx[li]=choose_empirical_price(counts,prices)
    return calidx,rawidx,full_idx,current_calidx,ew_idx,hcal+1,hraw+1

@njit(cache=True)
def batch_main(sorted_vals, price_grid, mean_theta, val_M, lambdas, eta_grid, alpha_grid):
    B,T,n=sorted_vals.shape; logprices=np.log(price_grid)
    H=T-1
    L=len(lambdas)
    out_idx=np.zeros((B,3),np.int32)
    swidx=np.zeros((B,H),np.int32)
    ewidx=np.zeros((B,L),np.int32)
    fs_weights_perf=np.zeros(B)
    selected_params=np.zeros((B,4),np.int32)
    period_rewards=np.zeros((B,T,len(price_grid)))
    for b in range(B):
        reward=get_period_reward(sorted_vals[b],logprices,price_grid)
        period_rewards[b]=reward
        ai,si,fi,wi,ei,ha,hs=solve_fv(sorted_vals[b], price_grid,logprices,mean_theta,val_M,lambdas,reward)
        out_idx[b,0]=ai;out_idx[b,1]=si;out_idx[b,2]=fi
        swidx[b,:]=wi
        ewidx[b,:]=ei
        selected_params[b,0]=ha; selected_params[b,1]=hs
    return period_rewards,out_idx,swidx,ewidx,selected_params

