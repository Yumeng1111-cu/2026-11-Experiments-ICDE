#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Correlation robustness: configurable CV (default .3), five copula rhos, SIX pricing strategies.

Runs 1000 TEST MC and 1000 INDEPENDENT Oracle-SW training MC by default.
Frozen parameter grids exactly unchanged from the 5000MC official experiment.
Common latent normals across rhos for lower Monte Carlo variance.
No Word or Figure 4 is created.
"""
import sys, csv, json, argparse, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from copula_market import (CV,N,T,M,MEAN_THETA,PRICES,RHO_VALUES,
    generate_latents,sample_histories_from_latents,population_revenues_gaussian,reported_correlations)
from acrp_core import batch_main,LAMBDAS,ETA_GRID,ALPHA_GRID
from competitors import swfi_ucb_joint_fv,fixed_share_fast,fixed_share_final_meanprice,BETA_GRID,FS_ETA,FS_ALPHA

METHODS=['ACRP-FV','SW-FV','SW-FI-UCB-FV','Fixed-Share','Calibrated Full-history','Oracle-SW']

def save_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: raise RuntimeError('No rows to save')
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description='Gaussian copula correlated-value robustness pilot')
    parser.add_argument('--mc',type=int,default=1000,help='Test MC replications per rho')
    parser.add_argument('--oracle-mc',type=int,default=1000,help='Independent Oracle-SW training replications per rho')
    parser.add_argument('--seed',type=int,default=20511120)
    parser.add_argument('--cv',type=float,default=CV)
    parser.add_argument('--data-root',type=str,default='results')
    parser.add_argument('--rhos',type=float,nargs='+',default=None,
                        help='One or more Gaussian-copula correlation coefficients (-1 < rho < 1).')
    args=parser.parse_args()
    assert args.mc>1 and args.oracle_mc>1
    assert 0 < args.cv < 3, 'CV must be positive and less than 3'
    cv=float(args.cv)
    rhos=tuple(RHO_VALUES if args.rhos is None else args.rhos)
    if not rhos or any(not np.isfinite(r) or abs(r)>=1 for r in rhos):
        parser.error('All rho values must satisfy -1 < rho < 1')
    if len(set(rhos)) != len(rhos):
        parser.error('Repeated rho values are not allowed')
    freeze=json.loads((ROOT/'config/frozen_config.json').read_text(encoding='utf-8'))
    assert abs(freeze['market']['mean_theta']-MEAN_THETA)<1e-12
    assert np.allclose(np.asarray(freeze['SW-FI-UCB-FV']['beta_candidates']),BETA_GRID)
    assert np.allclose(np.asarray(freeze['Fixed-Share']['eta_candidates']),FS_ETA)
    assert np.allclose(np.asarray(freeze['Fixed-Share']['alpha_candidates']),FS_ALPHA)

    t0=time.time()
    t_z1,t_z2=generate_latents(np.random.default_rng(args.seed+1000003),args.oracle_mc)
    s_z1,s_z2=generate_latents(np.random.default_rng(args.seed+2000003),args.mc)
    out_results=[];out_parameters=[];out_oracle=[];out_corr=[]
    dense=np.linspace(.01,5.,10001)
    for rho in rhos:
        pop=population_revenues_gaussian(PRICES,rho,cv=cv,quadrature=160)
        dense_pop=population_revenues_gaussian(dense,rho,cv=cv,quadrature=160)
        besti=int(np.argmax(dense_pop)); pstar=float(dense[besti]);rstar=float(dense_pop[besti])
        # Independent Oracle-SW training (no test history used for window selection).
        training=sample_histories_from_latents(t_z1,t_z2,rho,cv=cv)
        _,_,indices,_,_=batch_main(training,PRICES,MEAN_THETA,M,LAMBDAS,ETA_GRID,ALPHA_GRID)
        oracle_scores=pop[indices].mean(axis=0)
        oracle_h=int(np.argmax(oracle_scores))+1
        for h,rv in enumerate(oracle_scores,start=1):
            out_oracle.append({'rho':rho,'h':h,'training_mean_revenue':float(rv),'chosen_h':oracle_h,'oracle_training_mc':args.oracle_mc})

        # Test sample is common across methods; latents also common across the five rho scenarios.
        testing=sample_histories_from_latents(s_z1,s_z2,rho,cv=cv)
        rewards,main_indices,all_sw,_,param_arr=batch_main(testing,PRICES,MEAN_THETA,M,LAMBDAS,ETA_GRID,ALPHA_GRID)
        for j in range(args.mc):
            rw=rewards[j]
            k_u,h_u,beta_u,score_u=swfi_ucb_joint_fv(rw,PRICES,N,M,BETA_GRID,.05)
            loss=1.-rw/5.
            fs_rev,eta_id,alpha_id,score_fs=fixed_share_fast(loss,pop,M,FS_ETA,FS_ALPHA)
            fs_mean_price,_=fixed_share_final_meanprice(loss,FS_ETA[eta_id],FS_ALPHA[alpha_id],PRICES)
            idxs=[int(main_indices[j,0]),int(main_indices[j,1]),int(k_u),None,int(main_indices[j,2]),int(all_sw[j,oracle_h-1])]
            revs=[float(pop[k]) if k is not None else float(fs_rev) for k in idxs]
            prices=[float(PRICES[k]) if k is not None else float(fs_mean_price) for k in idxs]
            for name,k,rv,price in zip(METHODS,idxs,revs,prices):
                out_results.append({'rho':rho,'cv':cv,'mc_id':j+1,'method':name,
                    'price_or_expected_price':price,'expected_revenue':rv,'rrl_pct':100.*(rstar-rv)/rstar,
                    'price_index':-1 if k is None else k,'pstar':pstar,'rstar':rstar})
            out_parameters.append({'rho':rho,'mc_id':j+1,'ACRP_window':int(param_arr[j,0]),
                'SW_window':int(param_arr[j,1]),'UCB_window':int(h_u),'UCB_beta':float(beta_u),
                'UCB_validation_score':float(score_u),'FS_eta':float(FS_ETA[eta_id]),
                'FS_alpha':float(FS_ALPHA[alpha_id]),'FS_validation_score':float(score_fs),
                'Oracle_SW_fixed_window':oracle_h})
        pear,spear=reported_correlations(np.random.default_rng(args.seed+5000000),rho,cv=cv,n_sample=100000)
        out_corr.append({'rho':rho,'pearson_X_theta':pear,'spearman_X_theta':spear,
            'cv':cv,'mean_theta':MEAN_THETA,'test_mc':args.mc,'oracle_mc':args.oracle_mc,
            'pstar':pstar,'rstar':rstar,'oracle_SW_fixed_h':oracle_h})
        print(f'rho={rho:+.1f}: OracleSW h={oracle_h:2d}, optimal price={pstar:.4f}, '
              f'RRL ACRP={np.mean([r["rrl_pct"] for r in out_results if r["rho"]==rho and r["method"]=="ACRP-FV"]):.3f}%, '
              f'Pearson={pear:+.3f}, elapsed={time.time()-t0:.1f}s',flush=True)
    result_dir=ROOT/args.data_root/'raw'
    result_dir.mkdir(parents=True,exist_ok=True)
    save_csv(result_dir/'all_methods_by_rho.csv',out_results)
    save_csv(result_dir/'selected_parameters_by_rho.csv',out_parameters)
    save_csv(result_dir/'oracle_training_windows.csv',out_oracle)
    save_csv(result_dir/'rho_market_details.csv',out_corr)
    meta={'status':'EXPLORATORY PILOT, not confirmatory','mc_per_rho':args.mc,
          'oracle_training_mc_per_rho':args.oracle_mc,'rho_values':list(rhos),
          'seed':args.seed,'CV':cv,'mean_theta':MEAN_THETA,'n':N,'t':T,'M':M,
          'prices':PRICES.tolist(),'feedback':'full historical bids','margin_X':'TruncatedLogistic(5.2,0.79,[1,10])',
          'margin_theta':'Gamma(shape=1/CV^2,scale=.042 CV^2)',
          'copula':'Gaussian; latent-normal rho, not Pearson Corr(X,Theta)',
          'common_random_numbers_across_rho':True,
          'independent_oracle_training':True,
          'population_demand':'conditional Gaussian-copula analytic integral via Gauss-Legendre quadrature',
          'dense_oracle_grid_points':10001,
          'fixedshare':'randomized policy expected revenue',
          'frozen_parameter_grids':True,'elapsed_seconds':time.time()-t0}
    (result_dir/'experiment_config.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'DONE: {len(out_results)} method records, {len(out_parameters)} samples in {time.time()-t0:.2f} seconds',flush=True)

if __name__=='__main__':main()
