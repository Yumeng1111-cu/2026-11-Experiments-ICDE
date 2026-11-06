#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mean-decay misspecification experiment. True theta distribution NEVER changes.

Simulation and plotting are separate. All candidate grids are frozen.
Every c_hat scenario uses exactly the same test histories (paired comparison).
Oracle-SW uses separate independent histories to choose a fixed h for each c_hat.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from market import sample_histories, population_revenues, PRICE_GRID, MEAN_THETA
from acrp_core import batch_main, LAMBDAS, ETA_GRID, ALPHA_GRID
from competitors import (swfi_ucb_joint_fv, fixed_share_fast,
                         fixed_share_final_meanprice, BETA_GRID,FS_ETA,FS_ALPHA)

METHODS=['ACRP-FV','SW-FV','SW-UCB-FV','Fixed-Share',
         'Calibrated Full-history','Oracle-SW']
DEFAULT_ERRORS=[-0.5,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.5]


def run(args):
    if not (0 < args.cv <= 1):
        raise ValueError('CV must be in (0, 1].')
    if args.mc<2 or args.oracle_mc<2:
        raise ValueError('mc and oracle-mc must be at least 2')
    errors=sorted(set(float(e) for e in args.errors))
    if any(e<=-1 for e in errors):
        raise ValueError('Relative estimation error must exceed -1')
    if 0.0 not in errors:
        raise ValueError('Include zero error to evaluate the benchmark')
    out=Path(args.data_root)
    if not out.is_absolute():out=ROOT/out
    rawdir=out/'raw';sumdir=out/'summary'
    rawdir.mkdir(parents=True,exist_ok=True);sumdir.mkdir(parents=True,exist_ok=True)
    freeze=json.loads((ROOT/'config/frozen_config.json').read_text(encoding='utf-8'))
    assert float(MEAN_THETA)==.042
    assert np.allclose(BETA_GRID,freeze['SW-FI-UCB-FV']['beta_candidates'])
    assert np.allclose(FS_ETA,freeze['Fixed-Share']['eta_candidates'])
    assert np.allclose(FS_ALPHA,freeze['Fixed-Share']['alpha_candidates'])

    # Price oracle uses the true distribution and is shared across all assumed-decay rates.
    population_grid=population_revenues(PRICE_GRID,args.cv)
    dense=np.linspace(.01,5.,10001)
    dense_revenue=population_revenues(dense,args.cv)
    pstar=float(dense[np.argmax(dense_revenue)])
    rstar=float(np.max(dense_revenue))
    print(f'True mean={MEAN_THETA:.6f} CV={args.cv:g} pstar={pstar:.6f}, oracle revenue={rstar:.8f}',flush=True)

    # Test and independent training histories do NOT depend on estimated mean.
    test_rng=np.random.default_rng(args.seed)
    train_rng=np.random.default_rng(args.seed+97813)
    test_histories=sample_histories(test_rng,args.mc,80,args.cv,t=40)
    oracle_histories=sample_histories(train_rng,args.oracle_mc,80,args.cv,t=40)

    # Non-calibrating baselines do not depend on the assumed mean. Compute once.
    rewards,base_idx,base_sw_idx,_,base_par=batch_main(
        test_histories,PRICE_GRID,MEAN_THETA,10,LAMBDAS,ETA_GRID,ALPHA_GRID)
    test_noncal={}
    print('Computing non-calibrating controls (UCB / Fixed-Share)...',flush=True)
    for j in range(args.mc):
        uidx,uh,beta,_=swfi_ucb_joint_fv(rewards[j],PRICE_GRID,80,10,BETA_GRID,.05)
        loss=1-rewards[j]/5
        fs_rev,ie,ia,_=fixed_share_fast(loss,population_grid,10,FS_ETA,FS_ALPHA)
        fs_price,_=fixed_share_final_meanprice(loss,FS_ETA[ie],FS_ALPHA[ia],PRICE_GRID)
        test_noncal[j]=(int(base_idx[j,1]),int(uidx),float(fs_rev),
                        int(base_par[j,1]),int(uh),float(beta),
                        float(FS_ETA[ie]),float(FS_ALPHA[ia]),float(fs_price))

    rows=[];pars=[];oracle_rows=[]
    for e in errors:
        c=MEAN_THETA*(1+e)
        print(f'Running eps={e:+.0%}, assumed mean={c:.6f} ...',flush=True)
        _,target_idx,all_h_idx,_,selected=batch_main(
            test_histories,PRICE_GRID,c,10,LAMBDAS,ETA_GRID,ALPHA_GRID)
        _,_,train_sw_idx,_,_=batch_main(
            oracle_histories,PRICE_GRID,c,10,LAMBDAS,ETA_GRID,ALPHA_GRID)
        # For each assumed mean, choose fixed oracle h on independent training histories.
        true_scores=population_grid[train_sw_idx].mean(axis=0)
        h_oracle=int(np.argmax(true_scores))+1
        oracle_rows.append({'relative_error':e,'assumed_mean':c,'oracle_h':h_oracle,
            'independent_training_revenue':float(true_scores[h_oracle-1])})
        for j in range(args.mc):
            si,ui,fsrev,hs,hu,beta,eta,alpha,fsprice=test_noncal[j]
            ai=int(target_idx[j,0]); ci=int(target_idx[j,2]);oi=int(all_h_idx[j,h_oracle-1])
            names_idx=[ai,si,ui,None,ci,oi]
            prices=[float(PRICE_GRID[idx]) if idx is not None else fsprice for idx in names_idx]
            rev=[float(population_grid[idx]) if idx is not None else fsrev for idx in names_idx]
            for method, price, r in zip(METHODS,prices,rev):
                rows.append({'relative_error':e,'assumed_mean':c,'cv':args.cv,'mc_id':j+1,
                    'method':method,'price_or_expected_price':price,
                    'revenue':r,'rrl_pct':100*(rstar-r)/rstar})
            pars.append({'relative_error':e,'assumed_mean':c,'cv':args.cv,'mc_id':j+1,
                'ACRP_window':int(selected[j,0]),'SW_window':hs,
                'SW_UCB_window':hu,'SW_UCB_beta':beta,'FS_eta':eta,'FS_alpha':alpha,
                'Oracle_SW_fixed_window':h_oracle})
    df=pd.DataFrame(rows);pr=pd.DataFrame(pars);orf=pd.DataFrame(oracle_rows)
    df.to_csv(rawdir/'all_method_runs.csv',index=False,float_format='%.12g')
    pr.to_csv(rawdir/'selected_parameters.csv',index=False,float_format='%.12g')
    orf.to_csv(sumdir/'oracle_window.csv',index=False,float_format='%.12g')

    summary=[]
    for (e,m),g in df.groupby(['relative_error','method'],sort=False):
        v=g['rrl_pct'].to_numpy()
        se=float(v.std(ddof=1)/np.sqrt(len(v)))
        summary.append({'relative_error':e,'assumed_mean':MEAN_THETA*(1+e),
            'method':m,'mc_count':len(v),'mean_rrl_pct':float(v.mean()),
            'std_rrl_pct':float(v.std(ddof=1)), 'se_pct':se,
            'ci95_lower_pct':float(v.mean()-1.96*se),
            'ci95_upper_pct':float(v.mean()+1.96*se)})
    sf=pd.DataFrame(summary).sort_values(['relative_error','method'])
    sf.to_csv(sumdir/'summary_rrl.csv',index=False,float_format='%.12g')
    paired=[]
    baseline=df[df.method=='ACRP-FV'].pivot(index='mc_id',columns='relative_error',values='rrl_pct')
    for e in errors:
        z=(baseline[e]-baseline[0.0]).to_numpy()
        se=z.std(ddof=1)/np.sqrt(len(z))
        paired.append({'relative_error':e,'mean_rrl_change_vs_zero_pctpt':z.mean(),
            'ci95_lower_pctpt':z.mean()-1.96*se,
            'ci95_upper_pctpt':z.mean()+1.96*se})
    pf=pd.DataFrame(paired)
    pf.to_csv(sumdir/'ACRP_paired_vs_correct_mean.csv',index=False,float_format='%.12g')
    ws=pr.groupby('relative_error',as_index=False).agg(
        mean_ACRP_window=('ACRP_window','mean'),median_ACRP_window=('ACRP_window','median'),
        mean_SW_window=('SW_window','mean'),mean_UCB_window=('SW_UCB_window','mean'))
    ws.to_csv(sumdir/'window_summary.csv',index=False,float_format='%.12g')
    conf={'true_mean_theta':float(MEAN_THETA),'cv':args.cv,'mc':args.mc,
      'oracle_mc':args.oracle_mc,'seed':args.seed,'errors':errors,
      'prices':'geomspace(0.01,5,81)','target_t':40,'n':80,'M':10,
      'pstar_true':pstar,'rstar_true':rstar,
      'oracle_parameter_note':'Oracle-SW selects a fixed window on independent training histories for EACH assumed-mean scenario',
      'all_errors_share_same_market_histories':True,
      'UCB_and_FixedShare_do_not_depend_on_assumed_mean':True}
    (out/'experiment_config.json').write_text(json.dumps(conf,indent=2,ensure_ascii=False),encoding='utf-8')
    print('DONE: ',out,flush=True)
    print(sf.pivot(index='relative_error',columns='method',values='mean_rrl_pct').round(4).to_string(),flush=True)
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mc',type=int,default=500)
    parser.add_argument('--oracle-mc',type=int,default=500)
    parser.add_argument('--cv',type=float,default=.4)
    parser.add_argument('--seed',type=int,default=20821104)
    parser.add_argument('--errors',nargs='+',type=float,default=DEFAULT_ERRORS)
    parser.add_argument('--data-root',default='runs/validation_cv04_500mc')
    run(parser.parse_args())
