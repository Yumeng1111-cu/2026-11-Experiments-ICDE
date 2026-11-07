"""Experiment: independence of X and Theta; compare theta marginal Gamma, Lognormal, Two-point.

Each independent scenario has test histories shared by all six algorithms.
Oracle-SW trained on separate common random numbers.  All hyperparameter grids frozen.
"""
import sys,csv,json,argparse,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from independent_market import (DEFAULT_CV,N,T,M,MEAN_THETA,PRICES,DISTRIBUTIONS,
    generate_latents,sample_histories_from_uniforms,population_revenues,
    theta_parameters,theoretical_moments,theta_from_uniform,check_cv,
    X_LOC,X_SCALE,F_LOW,F_HIGH)
from acrp_core import batch_main,LAMBDAS,ETA_GRID,ALPHA_GRID
from competitors import (swfi_ucb_joint_fv,fixed_share_fast,fixed_share_final_meanprice,
    BETA_GRID,FS_ETA,FS_ALPHA)
METHODS=['ACRP-FV','SW-FV','SW-FI-UCB-FV','Fixed-Share','Calibrated Full-history','Oracle-SW']

def save_csv(filepath,rows):
    filepath.parent.mkdir(parents=True,exist_ok=True)
    with filepath.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--cv',type=float,default=DEFAULT_CV)
    pa.add_argument('--mc',type=int,default=100)
    pa.add_argument('--oracle-mc',type=int,default=100)
    pa.add_argument('--seed',type=int,default=20601121)
    pa.add_argument('--data-root',default='runs/distribution_cv04_pilot')
    pa.add_argument('--distributions',nargs='+',choices=DISTRIBUTIONS,default=list(DISTRIBUTIONS))
    a=pa.parse_args();check_cv(a.cv)
    if a.mc<2 or a.oracle_mc<2:pa.error('Both MC numbers must be at least 2')
    frozen=json.loads((ROOT/'config/frozen_config.json').read_text(encoding='utf-8'))
    assert frozen['market']['mean_theta']==MEAN_THETA
    assert np.allclose(frozen['SW-FI-UCB-FV']['beta_candidates'],BETA_GRID)
    assert np.allclose(frozen['Fixed-Share']['eta_candidates'],FS_ETA)
    assert np.allclose(frozen['Fixed-Share']['alpha_candidates'],FS_ALPHA)
    start=time.time()
    # Shared independent uniform draws across theta families reduce between-scenario MC noise.
    tu1,tu2=generate_latents(np.random.default_rng(a.seed+1000003),a.oracle_mc)
    su1,su2=generate_latents(np.random.default_rng(a.seed+2000003),a.mc)
    records=[];params=[];oracle_rows=[];details=[]
    dense=np.linspace(.01,5.,10001)
    for dist in a.distributions:
        population=population_revenues(PRICES,dist,a.cv)
        fine=population_revenues(dense,dist,a.cv)
        pstar=float(dense[np.argmax(fine)]);rstar=float(np.max(fine))
        training=sample_histories_from_uniforms(tu1,tu2,dist,a.cv)
        _,_,sw_train,_,_=batch_main(training,PRICES,MEAN_THETA,M,LAMBDAS,ETA_GRID,ALPHA_GRID)
        avg_revs=np.mean(population[sw_train],axis=0)
        oracle_h=int(np.argmax(avg_revs)+1)
        for h,rv in enumerate(avg_revs,1):
            oracle_rows.append({'distribution':dist,'cv':a.cv,'h':h,
                'mean_training_revenue':float(rv),'oracle_h':oracle_h,'oracle_train_mc':a.oracle_mc})
        test=sample_histories_from_uniforms(su1,su2,dist,a.cv)
        rewards,inds,sw_indices,_,main_params=batch_main(test,PRICES,MEAN_THETA,M,LAMBDAS,ETA_GRID,ALPHA_GRID)
        for j in range(a.mc):
            rw=rewards[j]
            ku,hu,beta,ucb_val=swfi_ucb_joint_fv(rw,PRICES,N,M,BETA_GRID,.05)
            fs_rev,ei,ai,fs_val=fixed_share_fast(1.-rw/5.,population,M,FS_ETA,FS_ALPHA)
            fs_expected_price,_=fixed_share_final_meanprice(1.-rw/5.,FS_ETA[ei],FS_ALPHA[ai],PRICES)
            indices=(int(inds[j,0]),int(inds[j,1]),int(ku),None,int(inds[j,2]),int(sw_indices[j,oracle_h-1]))
            for method,k in zip(METHODS,indices):
                rv=float(fs_rev) if k is None else float(population[k])
                price=float(fs_expected_price) if k is None else float(PRICES[k])
                records.append({'distribution':dist,'cv':a.cv,'mc_id':j+1,'method':method,
                    'selected_price_or_mean_price':price,'expected_revenue':rv,
                    'rrl_pct':100*(rstar-rv)/rstar,'price_index':-1 if k is None else k,
                    'pstar':pstar,'rstar':rstar})
            params.append({'distribution':dist,'cv':a.cv,'mc_id':j+1,
                'ACRP_window':int(main_params[j,0]),'SW_window':int(main_params[j,1]),
                'UCB_window':int(hu),'UCB_beta':float(beta),'UCB_validation_score':float(ucb_val),
                'FS_eta':float(FS_ETA[ei]),'FS_alpha':float(FS_ALPHA[ai]),
                'FS_validation_score':float(fs_val),'Oracle_SW_fixed_window':oracle_h})
        theo_mu,theo_cv=theoretical_moments(dist,a.cv)
        theta_diag=theta_from_uniform(su2,dist,a.cv)
        from scipy.special import logit
        X_diag=X_LOC+X_SCALE*logit(F_LOW+(F_HIGH-F_LOW)*su1)
        empirical_correlation=float(np.corrcoef(X_diag.ravel(),theta_diag.ravel())[0,1])
        details.append({'distribution':dist,'cv_requested':a.cv,'theta_mean_theory':theo_mu,
            'theta_CV_theory':theo_cv,'theta_mean_sample':float(theta_diag.mean()),
            'theta_CV_sample':float(theta_diag.std()/theta_diag.mean()),
            'theta_p05_sample':float(np.quantile(theta_diag,.05)),
            'theta_median_sample':float(np.median(theta_diag)),
            'theta_p95_sample':float(np.quantile(theta_diag,.95)),
            'theta_skew_sample':float(np.mean(((theta_diag-theta_diag.mean())/theta_diag.std())**3)),
            'sample_corr_X_theta':empirical_correlation,
            'theta_parameters':json.dumps(theta_parameters(dist,a.cv)),
            'pstar':pstar,'rstar':rstar,'oracle_window':oracle_h,'test_mc':a.mc,'oracle_mc':a.oracle_mc})
        current=[r['rrl_pct'] for r in records if r['distribution']==dist and r['method']=='ACRP-FV']
        print(f'{dist}: oracle-h={oracle_h}, ACRP mean RRL={np.mean(current):.4f}%, '
              f'theta theoretical mean={theo_mu:.6f}, CV={theo_cv:.4f}; elapsed={time.time()-start:.1f}s',flush=True)
    out=ROOT/a.data_root/'raw'
    save_csv(out/'all_methods_by_distribution.csv',records)
    save_csv(out/'selected_parameters_by_distribution.csv',params)
    save_csv(out/'oracle_training_windows.csv',oracle_rows)
    save_csv(out/'market_details.csv',details)
    metadata={'experiment':'independent X and Theta, theta distribution-shape robustness',
        'CV':a.cv,'mean_theta':MEAN_THETA,'distributions':a.distributions,
        'mc_per_distribution':a.mc,'oracle_training_mc_per_distribution':a.oracle_mc,
        'seed':a.seed,'X':'TruncatedLogistic(5.2,0.79,[1,10])',
        'theta':'gamma/lognormal/symmetric equal-probability two-point; matched mean and CV',
        'two_point':'P(theta=mean*(1-CV))=P(theta=mean*(1+CV))=0.5; requires CV<=1',
        'independence_X_theta':True,'common_random_numbers_across_distributions':True,
        'all_algorithms_full_bid_feedback':True,'fixed_share_expected_randomized_revenue':True,
        'oracle_sw_training_independent':True,'oracle_h_range':[1,39],
        'forward_validation_periods':[30,39],'n':N,'T':T,'price_grid':PRICES.tolist(),
        'continuous_oracle_dense_grid_points':len(dense),'delta_ucb':.05,
        'frozen_grids':True,'elapsed_seconds':time.time()-start}
    (out/'experiment_config.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'COMPLETED: {len(records)} method rows and {len(params)} MC histories; seconds={time.time()-start:.1f}',flush=True)
if __name__=='__main__': main()
