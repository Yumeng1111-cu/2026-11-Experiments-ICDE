"""Independent reproducible experiment. Does not plot or construct Word/Excel.

From project root:
  python run_main_experiment.py --mc 5000 --oracle-mc 5000 --seed 20331129
To split CVs across four processes (safe parallel, same reproducible seeds):
  python run_main_experiment.py --cv 0.2 --mc 5000 --oracle-mc 5000 --seed 20331129
  ... 0.3, 0.4, 0.5 ...
  python build_tables.py
  python make_figures.py
  python create_word_report.py

All algorithms choose action before seeing target-period bids. Test history is shared
among six methods within an MC repetition. The full-feedback UCB is a CUSTOM
adaptation, NOT the original bandit-feedback SW-UCB.
"""
from pathlib import Path
import sys,json,argparse,time,csv,copy
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent/'src'))
from market import sample_histories,population_revenues,PRICE_GRID,MEAN_THETA
from acrp_core import batch_main,LAMBDAS,ETA_GRID,ALPHA_GRID
from competitors import swfi_ucb_joint_fv, fixed_share_fast, fixed_share_final_meanprice, BETA_GRID,FS_ETA,FS_ALPHA

ROOT=Path(__file__).resolve().parent
NAMES=['ACRP-FV','SW-FV','SW-FI-UCB-FV','Fixed-Share','Calibrated Full-history','Oracle-SW']
CVS=(.2,.3,.4,.5)

def output_dir(path):
    path=Path(path)
    return path if path.is_absolute() else ROOT/path

def oracle_continuous(cv):
    dense=np.linspace(.01,5.,10001)
    rr=population_revenues(dense,cv)
    idx=int(np.argmax(rr))
    return float(dense[idx]),float(rr[idx])

def prepare_market(cv,oracle_mc,seed,batch):
    pop=population_revenues(PRICE_GRID,cv)
    pstar,rstar=oracle_continuous(cv)
    rng=np.random.default_rng(seed+1000003)
    scores=np.zeros(39,dtype=np.float64)
    n_done=0
    while n_done<oracle_mc:
        B=min(batch,oracle_mc-n_done)
        logs=sample_histories(rng,B,80,cv,t=40)
        _,_,idx,_,_=batch_main(logs,PRICE_GRID,MEAN_THETA,10,LAMBDAS,ETA_GRID,ALPHA_GRID)
        scores+=pop[idx].sum(axis=0)
        n_done+=B
    scores/=oracle_mc
    best_h=int(np.argmax(scores))+1
    return pop,pstar,rstar,best_h,scores

def mc_per_cv(cv, mc, oracle_mc, seed, batch, path, cv_index):
    path.mkdir(parents=True,exist_ok=True)
    cvname=f'cv_{cv:.1f}'
    rng=np.random.default_rng(seed+10009*cv_index)
    oracle_seed=seed+10009*cv_index
    pop,pstar,rstar,oracle_h,oracle_scores=prepare_market(cv,oracle_mc,oracle_seed,batch)
    rr_records=[]; param_records=[]
    clock=time.perf_counter()
    done=0
    log_path=path/f'{cvname}_progress.log'
    log_path.write_text(f'Start CV={cv} MC={mc} oracle_h={oracle_h} mean_theta={MEAN_THETA}\n',encoding='utf-8')
    while done<mc:
        B=min(batch,mc-done)
        logs=sample_histories(rng,B,80,cv,t=40)
        reward, main_idx, all_sw_idx, _, fv_params=batch_main(logs,PRICE_GRID,MEAN_THETA,10,LAMBDAS,ETA_GRID,ALPHA_GRID)
        for j in range(B):
            mid=done+j+1
            period_reward=reward[j]
            u_idx,u_h,u_beta,u_score=swfi_ucb_joint_fv(period_reward,PRICE_GRID,80,10,BETA_GRID,.05)
            loss=1.-period_reward/5.
            fs_rev,eta_id,alpha_id,fs_score=fixed_share_fast(loss,pop,10,FS_ETA,FS_ALPHA)
            fs_ave_price,fs_peak_price=fixed_share_final_meanprice(loss,FS_ETA[eta_id],FS_ALPHA[alpha_id],PRICE_GRID)
            indexes=[int(main_idx[j,0]),int(main_idx[j,1]),int(u_idx),None,int(main_idx[j,2]),int(all_sw_idx[j,oracle_h-1])]
            revs=[float(pop[ix]) if ix is not None else float(fs_rev) for ix in indexes]
            price_out=[float(PRICE_GRID[ix]) if ix is not None else float(fs_ave_price) for ix in indexes]
            for name, ix,rv,pp in zip(NAMES,indexes,revs,price_out):
                rr_records.append({'cv':cv,'mc_id':mid,'method':name,'price_or_mean_price':pp,
                 'fixed_share_randomized':int(name=='Fixed-Share'),
                 'expected_revenue':rv,'rrl_pct':100.*(rstar-rv)/rstar,
                 'p_star':pstar,'oracle_revenue':rstar,'price_index':-1 if ix is None else ix})
            param_records.append({'cv':cv,'mc_id':mid,'ACRP_window':int(fv_params[j,0]),
                'SW_window':int(fv_params[j,1]),'UCB_window':int(u_h),
                'UCB_beta':float(u_beta),'UCB_validation_score':float(u_score),
                'FS_eta':float(FS_ETA[eta_id]),'FS_alpha':float(FS_ALPHA[alpha_id]),
                'FS_validation_score':float(fs_score),'FS_mean_price':float(fs_ave_price),
                'FS_mode_price':float(fs_peak_price),'Oracle_SW_fixed_window':int(oracle_h)})
        done+=B
        if done%max(batch*4,1)==0 or done==mc:
            sec=time.perf_counter()-clock
            line=f'CV={cv:.1f} completed={done}/{mc} elapsed={sec:.1f}s, rate={done/max(sec,1.e-4):.2f} hist/s'
            with log_path.open('a',encoding='utf-8') as f: f.write(line+'\n')
            print(line,flush=True)
    df=pd.DataFrame(rr_records)
    params=pd.DataFrame(param_records)
    df.to_csv(path/f'{cvname}_raw_results.csv',index=False,float_format='%.12g')
    params.to_csv(path/f'{cvname}_selected_parameters.csv',index=False,float_format='%.12g')
    pd.DataFrame({'h':np.arange(1,40),'independent_training_mean_revenue':oracle_scores}).to_csv(path/f'{cvname}_oracle_training.csv',index=False)
    meta={'cv':cv,'mc_runs':mc,'oracle_training_runs':oracle_mc,
      'seed':seed+10009*cv_index,'oracle_seed':oracle_seed+1000003,
      'oracle_SW_fixed_window':oracle_h,'p_star':pstar,'oracle_revenue':rstar,
      'mean_theta':MEAN_THETA,'prices':PRICE_GRID.tolist(),'N':80,'T':40,'M':10,
      'disclaimer':'Fixed-Share is a randomized price strategy evaluated by expected revenue; adapted full-information UCB is not original SW-UCB.'}
    (path/f'{cvname}_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    return meta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mc',type=int,default=5000)
    ap.add_argument('--oracle-mc',type=int,default=5000)
    ap.add_argument('--seed',type=int,default=20331129)
    ap.add_argument('--cv',type=float,default=None)
    ap.add_argument('--batch',type=int,default=100)
    ap.add_argument('--output',default='results/raw')
    args=ap.parse_args()
    assert args.mc>0 and args.oracle_mc>0
    cvs=CVS if args.cv is None else (args.cv,)
    freeze=json.loads((ROOT/'config/frozen_config.json').read_text())
    assert abs(MEAN_THETA-.042)<1e-12 and freeze['market']['mean_theta']==MEAN_THETA
    assert list(BETA_GRID)==freeze['SW-FI-UCB-FV']['beta_candidates']
    assert np.allclose(FS_ETA,freeze['Fixed-Share']['eta_candidates'])
    assert np.allclose(FS_ALPHA,freeze['Fixed-Share']['alpha_candidates'])
    meta=[]
    for cv in cvs:
        assert any(abs(cv-i)<1e-9 for i in CVS)
        cv_idx=next(i for i,c in enumerate(CVS) if abs(cv-c)<1e-9)
        meta.append(mc_per_cv(cv,args.mc,args.oracle_mc,args.seed,args.batch,output_dir(args.output),cv_idx))
    print('Finished:', [(x['cv'],x['oracle_SW_fixed_window']) for x in meta],flush=True)

if __name__=='__main__':main()
