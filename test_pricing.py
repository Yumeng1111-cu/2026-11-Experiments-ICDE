
import numpy as np
from core import _best_price,_fill_period,_test_history,prep_sort,new_history,population_truth,compute_scenario


def brute_price(q,w,scale,lo,hi):
    prices=np.unique(np.r_[lo,hi,np.asarray(q)*scale])
    prices=prices[(prices>=lo)&(prices<=hi)]
    scores=np.array([p*np.sum(w[np.asarray(q)*scale>=p]) for p in prices])
    return prices[np.argmax(scores)],scores.max()


def test_breakpoints():
    rng=np.random.default_rng(34)
    for case in range(150):
        periods=int(rng.integers(2,8))
        n=int(rng.integers(2,8))
        q=rng.uniform(0.001,6.,size=periods*n)
        if case%5==0:
            q=np.round(q,1)   # repeated bid values & ties
        rankvalues,rank=prep_sort(q)
        weights=np.zeros(periods*n)
        cnt=np.zeros(len(q))
        bc=np.zeros((len(q)+47)//48)
        for p in range(1,periods+1):
            w=float(rng.uniform(.01,1.0))
            _fill_period(rank,cnt,bc,n,p,w)
            weights[(p-1)*n:p*n]=w
        scale=float(rng.uniform(.15,1.0))
        got=_best_price(rankvalues,cnt,bc,scale,.01,5.)
        expected,maxscore=brute_price(q,weights,scale,.01,5.)
        achieved=got*np.sum(weights[q*scale>=got])
        if not np.isclose(achieved,maxscore,rtol=1e-9,atol=1e-8):
            raise AssertionError(f'Case {case}: got {got} score {achieved}, expected {expected} score {maxscore}')
    print('PASS: 150 weighted breakpoint pricing tests (incl. ties and bounded domain).')


def test_all_six_method_prices():
    """独立小样本直接穷举验证：FV 验证窗、全历史、两种 Oracle 价。"""
    s=dict(mu=.02,cv=.4,n=3,t=25)
    raw,cal=new_history(s,303,0)
    n,t,mu=s['n'],s['t'],s['mu']
    hstar=4;lam=.095;M=20;H=t-M-1
    scores_u=np.zeros(H);scores_c=np.zeros(H)
    for v in range(t-M,t):
        idx=(v-1)*n
        current=raw[idx:idx+n]
        for h in range(1,H+1):
            ss=np.arange(v-h,v)
            ids=np.concatenate([np.arange((int(z)-1)*n,int(z)*n) for z in ss])
            qraw=raw[ids];qcal=cal[ids]
            pu,_=brute_price(qraw,np.ones(len(qraw)),1.,.01,5.)
            pc,_=brute_price(qcal,np.ones(len(qcal)),np.exp(-mu*(v-1)),.01,5.)
            scores_u[h-1]+=pu*np.count_nonzero(current>=pu)/n
            scores_c[h-1]+=pc*np.count_nonzero(current>=pc)/n
    hu=int(np.argmax(scores_u)+1);hc=int(np.argmax(scores_c)+1)
    qraw=raw;qcal=cal;sc=np.exp(-mu*(t-1))
    pfullraw,_=brute_price(qraw,np.ones(len(raw)),1.,.01,5.)
    pfullcal,_=brute_price(qcal,np.ones(len(cal)),sc,.01,5.)
    def window_price(h,quote,scale):
        ids=np.arange((t-1-h)*n,(t-1)*n)
        return brute_price(quote[ids],np.ones(len(ids)),scale,.01,5.)[0]
    pew_weights=np.repeat(np.exp(-lam*np.arange(t-1,0,-1)),n)
    pew,_=brute_price(qcal,pew_weights,sc,.01,5.)
    expected=np.array([pfullraw,pfullcal,window_price(hu,raw,1.),
                       window_price(hc,cal,sc),window_price(hstar,cal,sc),pew])
    sr,rr=prep_sort(raw);scc,rc=prep_sort(cal)
    got,gothu,gothc=_test_history(raw,sr,rr,scc,rc,n,t,mu,M,hstar,lam,.01,5.)
    assert (gothu,gothc)==(hu,hc),((gothu,gothc),(hu,hc))
    assert np.allclose(got,expected,atol=1e-10,rtol=1e-10),(got,expected)
    print('PASS: six methods, forward validation and Oracle price vs independent enumeration.')


def test_one_scenario():
    s=dict(mu=.02,cv=.4,n=5,t=28)
    df=compute_scenario(s,test=2,pilot=2)
    assert len(df)==2
    assert df['h_acrp'].between(1,s['t']-21).all()
    assert df['h_unc'].between(1,s['t']-21).all()
    assert df['oracle_h'].between(1,s['t']-1).all()
    assert df['oracle_lambda'].between(0,.4).all()
    assert (df[['Full-history ERM','Calibrated Full-history','Uncalibrated SW-FV',
                'ACRP-FV','Oracle-SW','Oracle-EW']].to_numpy()>=-1e-5).all()
    print('PASS: from-scratch Monte Carlo and FV smoke test.')


if __name__=='__main__':
    test_breakpoints()
    test_all_six_method_prices()
    test_one_scenario()
    print('ALL TESTS PASSED.')
