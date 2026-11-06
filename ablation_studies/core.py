from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from scipy.special import ndtr, roots_genlaguerre, gammaln
import settings as C

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'outputs'/'data'
FIG = ROOT/'outputs'/'figures'
DATA.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

METHODS = ['Full-history ERM', 'Calibrated Full-history', 'Uncalibrated SW-FV',
           'ACRP-FV', 'Oracle-SW', 'Oracle-EW']
COLORS = {'Full-history ERM':'#5F7289', 'Calibrated Full-history':'#43AA8B',
          'Uncalibrated SW-FV':'#E9C46A', 'ACRP-FV':'#C84D62',
          'Oracle-SW':'#7A6BB3', 'Oracle-EW':'#404040'}
MARKERS = {'Full-history ERM':'o', 'Calibrated Full-history':'s',
           'Uncalibrated SW-FV':'^', 'ACRP-FV':'D',
           'Oracle-SW':'v', 'Oracle-EW':'P'}
STYLES = {m: ('--' if m.startswith('Oracle-') else '-') for m in METHODS}
LAMBDAS = np.arange(round(C.LAMBDA_MAX/C.LAMBDA_STEP)+1)*C.LAMBDA_STEP
BLOCK = 48


def scenarios(experiment: str):
    if experiment == 'A':
        return [dict(mu=mu, cv=cv, n=C.A_N, t=C.A_T) for mu in C.A_MU for cv in C.A_CV]
    if experiment == 'B':
        return [dict(mu=C.B_MU, cv=cv, n=n, t=C.B_T) for cv in C.B_CV for n in C.B_N]
    if experiment == 'C':
        return [dict(mu=C.C_MU, cv=cv, n=C.C_N, t=t) for cv in C.C_CV for t in C.C_T]
    raise ValueError('experiment must be A, B or C')


def n_reps():
    return (C.QUICK_REPS, C.QUICK_REPS) if C.QUICK_MODE else (C.TEST_REPS, C.PILOT_REPS)


def scenario_name(s):
    return f"mu{s['mu']:.3f}_cv{s['cv']:.2f}_n{s['n']}_t{s['t']}"


def population_truth(s):
    """直接从 Gamma-normal 数学模型数值积分，获取 r_t(p) 和 r_t(p*)。
    30,001 个点用于插值评价（同先前实验的 0.01..5.0 网格）；最大点用
    局部连续优化补足，独立于任何 Monte Carlo 样本。
    """
    from scipy.optimize import minimize_scalar
    mu, cv, age = s['mu'], s['cv'], s['t']-1
    if cv == 0:
        theta, weights = np.array([mu]), np.array([1.0])
    else:
        shape = 1/(cv*cv)
        nodes, ww = roots_genlaguerre(96, shape-1)
        theta = mu*cv*cv*nodes
        weights = ww/np.exp(gammaln(shape))
        weights = weights/weights.sum()
    grid = np.linspace(C.PRICE_MIN, C.PRICE_MAX, 30001)
    revenue = np.empty_like(grid)
    for k in range(0,len(grid),1000):
        r=min(k+1000,len(grid))
        z=(-np.log(grid[k:r,None])-age*theta[None,:])/C.LOG_X_STD
        revenue[k:r] = grid[k:r]*(ndtr(z) @ weights)
    idx=int(np.argmax(revenue))
    lo=grid[max(idx-2,0)]; hi=grid[min(idx+2,len(grid)-1)]
    obj=lambda p: float(p * (ndtr((-np.log(p)-age*theta)/C.LOG_X_STD) @ weights))
    local=minimize_scalar(lambda p:-obj(p),bounds=(lo,hi),method='bounded',options={'xatol':1e-13})
    return grid, revenue, max(float(revenue[idx]), -float(local.fun))


def new_history(s, stream, rep):
    mu, cv, n, t = s['mu'], s['cv'], s['n'], s['t']
    seed_seq=np.random.SeedSequence([C.SEED, round(mu*1e6), round(cv*1e6), n, t, stream, rep])
    rng=np.random.default_rng(seed_seq)
    age=np.arange(t-1,dtype=np.float64)[:,None]
    z=rng.normal(0.0,C.LOG_X_STD,(t-1,n))
    if cv==0:
        theta=mu
    else:
        theta=rng.gamma(1/(cv*cv),mu*cv*cv,(t-1,n))
    raw=np.exp(z-theta*age)
    cal=raw*np.exp(mu*age)
    return np.ascontiguousarray(raw.ravel()),np.ascontiguousarray(cal.ravel())


def prep_sort(quotes):
    perm=np.argsort(quotes,kind='mergesort')
    rank=np.empty(len(perm),dtype=np.int64)
    rank[perm]=np.arange(len(perm))
    values=np.ascontiguousarray(quotes[perm])
    return values,rank


@njit(cache=True)
def _fill_period(rank, counts, blocks, n, period, weight):
    base=(period-1)*n
    for j in range(n):
        k=rank[base+j]
        counts[k]+=weight
        blocks[k//BLOCK]+=weight


@njit(cache=True)
def _best_price(values, counts, blocks, scale, minp, maxp):
    size=len(values)
    nblocks=len(blocks)
    lo=minp/scale
    hi=maxp/scale
    tails=np.empty(nblocks+1,dtype=np.float64)
    tails[nblocks]=0.0
    for b in range(nblocks-1,-1,-1):
        tails[b]=tails[b+1]+blocks[b]
    # q=lo and q=hi are both legal candidate prices.
    lowerpos=np.searchsorted(values,lo,side='left')
    above_lo=0.0
    if lowerpos<size:
        b=lowerpos//BLOCK
        above_lo=tails[b+1]
        end=min((b+1)*BLOCK,size)
        for k in range(lowerpos,end):
            above_lo+=counts[k]
    bestq=lo
    score=lo*above_lo
    upperpos=np.searchsorted(values,hi,side='left')
    above_hi=0.0
    if upperpos<size:
        b=upperpos//BLOCK
        above_hi=tails[b+1]
        end=min((b+1)*BLOCK,size)
        for k in range(upperpos,end):
            above_hi+=counts[k]
    if hi*above_hi>score:
        score=hi*above_hi
        bestq=hi
    # First feasible q in each block gives a valid attained lower bound.
    for b in range(nblocks):
        st=b*BLOCK; en=min(st+BLOCK,size)
        if values[en-1]<lo or values[st]>hi:
            continue
        k=np.searchsorted(values[st:en],lo,side='left')+st
        if k>=en or values[k]>hi:
            continue
        mass=tails[b+1]
        for j in range(k,en):
            mass+=counts[j]
        candidate=values[k]*mass
        if candidate>score:
            score=candidate
            bestq=values[k]
    # Safe block upper bound; exact enumeration inside surviving blocks.
    for b in range(nblocks):
        st=b*BLOCK; en=min(st+BLOCK,size)
        if values[en-1]<lo or values[st]>hi:
            continue
        max_q=min(hi,values[en-1])
        if max_q*tails[b] + 1e-12*max(1.,score) < score:
            continue
        mass=tails[b+1]
        for j in range(en-1,st-1,-1):
            mass+=counts[j]
            q=values[j]
            if lo<=q<=hi:
                candidate=q*mass
                if candidate>score:
                    score=candidate
                    bestq=q
    return max(minp,min(maxp,bestq*scale))


@njit(cache=True)
def _pilot_sw(values,rank,n,t,scale,minp,maxp):
    cnt=np.zeros(len(values),dtype=np.float64)
    blk=np.zeros((len(values)+BLOCK-1)//BLOCK,dtype=np.float64)
    pred=np.empty(t-1,dtype=np.float64)
    for h in range(1,t):
        _fill_period(rank,cnt,blk,n,t-h,1.0)
        pred[h-1]=_best_price(values,cnt,blk,scale,minp,maxp)
    return pred


@njit(cache=True)
def _pilot_ew(values,rank,n,t,scale,lambdas,minp,maxp):
    cnt=np.zeros(len(values),dtype=np.float64)
    blk=np.zeros((len(values)+BLOCK-1)//BLOCK,dtype=np.float64)
    pred=np.empty(len(lambdas),dtype=np.float64)
    for j in range(len(lambdas)):
        cnt[:]=0.0
        blk[:]=0.0
        lam=lambdas[j]
        for d in range(1,t):
            _fill_period(rank,cnt,blk,n,t-d,np.exp(-lam*d))
        pred[j]=_best_price(values,cnt,blk,scale,minp,maxp)
    return pred


@njit(cache=True)
def _test_history(raw, sorted_raw,rank_raw,sorted_cal,rank_cal,
                  n,t,mu,validation_M,oracle_h,oracle_lambda,minp,maxp):
    """计算同一条 test history 上全部六种方法价格和两个 FV 选中 h。"""
    H=t-validation_M-1
    if H<1:
        raise ValueError('t must be greater than validation_M + 1')
    scores_raw=np.zeros(H,dtype=np.float64)
    scores_cal=np.zeros(H,dtype=np.float64)
    cu=np.zeros(len(sorted_raw),dtype=np.float64)
    bu=np.zeros((len(sorted_raw)+BLOCK-1)//BLOCK,dtype=np.float64)
    cc=np.zeros(len(sorted_cal),dtype=np.float64)
    bc=np.zeros((len(sorted_cal)+BLOCK-1)//BLOCK,dtype=np.float64)
    # v=t-M..t-1; fit windows solely on periods s<v.
    for v in range(t-validation_M,t):
        cu[:]=0.0;bu[:]=0.0;cc[:]=0.0;bc[:]=0.0
        scale_v=np.exp(-mu*(v-1))
        base=(v-1)*n
        for h in range(1,H+1):
            s=v-h
            _fill_period(rank_raw,cu,bu,n,s,1.0)
            _fill_period(rank_cal,cc,bc,n,s,1.0)
            p_unc=_best_price(sorted_raw,cu,bu,1.0,minp,maxp)
            p_cal=_best_price(sorted_cal,cc,bc,scale_v,minp,maxp)
            sales_u=0; sales_c=0
            for i in range(n):
                q=raw[base+i]
                if q>=p_unc:sales_u+=1
                if q>=p_cal:sales_c+=1
            scores_raw[h-1]+=p_unc*sales_u/n
            scores_cal[h-1]+=p_cal*sales_c/n
    # deterministic, leftmost maximizing window length (tie breaking matches manuscript code).
    h_unc=1;h_acrp=1
    for h in range(2,H+1):
        if scores_raw[h-1] > scores_raw[h_unc-1]+1e-13:
            h_unc=h
        if scores_cal[h-1] > scores_cal[h_acrp-1]+1e-13:
            h_acrp=h
    # target date prices for all memory windows; extract full, FV, Oracle-SW.
    cu[:]=0.0;bu[:]=0.0;cc[:]=0.0;bc[:]=0.0
    scale_t=np.exp(-mu*(t-1))
    p_unc_full=0.0;p_cal_full=0.0;p_unc_sel=0.0;p_cal_sel=0.0;p_sw_oracle=0.0
    for h in range(1,t):
        _fill_period(rank_raw,cu,bu,n,t-h,1.0)
        _fill_period(rank_cal,cc,bc,n,t-h,1.0)
        pu=_best_price(sorted_raw,cu,bu,1.0,minp,maxp)
        pc=_best_price(sorted_cal,cc,bc,scale_t,minp,maxp)
        if h==h_unc:p_unc_sel=pu
        if h==h_acrp:p_cal_sel=pc
        if h==oracle_h:p_sw_oracle=pc
        if h==t-1:
            p_unc_full=pu;p_cal_full=pc
    # Fixed Oracle-EW lambda trained *only* on independent pilot histories.
    cc[:]=0.0;bc[:]=0.0
    for d in range(1,t):
        _fill_period(rank_cal,cc,bc,n,t-d,np.exp(-oracle_lambda*d))
    p_ew_oracle=_best_price(sorted_cal,cc,bc,scale_t,minp,maxp)
    return np.array([p_unc_full,p_cal_full,p_unc_sel,p_cal_sel,p_sw_oracle,p_ew_oracle]),h_unc,h_acrp


@njit(cache=True)
def _truth_interp(price, rev, price_step, minp):
    x=(price-minp)/price_step
    if x<=0:
        return rev[0]
    if x>=len(rev)-1:
        return rev[-1]
    j=int(x)
    return rev[j]*(1-(x-j)) + rev[j+1]*(x-j)


@njit(cache=True)
def _vector_true_revenue(prices, rev, step, minp):
    vals=np.empty(len(prices),dtype=np.float64)
    for i in range(len(prices)):
        vals[i]=_truth_interp(prices[i],rev,step,minp)
    return vals


def compute_scenario(s, test, pilot):
    """从头运行：Oracle-SW 训练、Oracle-EW 独立训练、全部策略共同测试。"""
    start=time.monotonic()
    grid,truth,optimal=population_truth(s)
    step=float(grid[1]-grid[0]); minp=C.PRICE_MIN; maxp=C.PRICE_MAX
    n,t=s['n'],s['t']
    scale_t=np.exp(-s['mu']*(t-1))

    sums_sw=np.zeros(t-1,dtype=float)
    for r in range(pilot):
        _,cal=new_history(s,101,r)
        v,ranks=prep_sort(cal)
        prices=_pilot_sw(v,ranks,n,t,scale_t,minp,maxp)
        sums_sw+=_vector_true_revenue(prices,truth,step,minp)
        if pilot>=100 and (r+1)%max(1,pilot//10)==0:
            print(f'  Oracle-SW pilot: {r+1}/{pilot}',flush=True)
    oracle_h=int(np.argmax(sums_sw)+1)

    sums_ew=np.zeros(len(LAMBDAS),dtype=float)
    for r in range(pilot):
        _,cal=new_history(s,202,r)
        v,ranks=prep_sort(cal)
        prices=_pilot_ew(v,ranks,n,t,scale_t,LAMBDAS,minp,maxp)
        sums_ew+=_vector_true_revenue(prices,truth,step,minp)
        if pilot>=100 and (r+1)%max(1,pilot//10)==0:
            print(f'  Oracle-EW pilot: {r+1}/{pilot}',flush=True)
    oracle_lambda=float(LAMBDAS[int(np.argmax(sums_ew))])

    rows=np.empty((test,6),dtype=float)
    unc_h=np.empty(test,dtype=int)
    acrp_h=np.empty(test,dtype=int)
    for r in range(test):
        raw,cal=new_history(s,303,r)
        values_raw,rank_raw=prep_sort(raw)
        values_cal,rank_cal=prep_sort(cal)
        prices,hu,hc=_test_history(raw,values_raw,rank_raw,values_cal,rank_cal,n,t,
                                    s['mu'],C.FV_PERIODS,oracle_h,oracle_lambda,minp,maxp)
        realized=_vector_true_revenue(prices,truth,step,minp)
        rows[r,:]=100*(optimal-realized)/optimal
        unc_h[r]=hu;acrp_h[r]=hc
        if test>=100 and (r+1)%max(1,test//10)==0:
            print(f'  Test histories: {r+1}/{test}',flush=True)
    df=pd.DataFrame(rows,columns=METHODS)
    df.insert(0,'rep',np.arange(test,dtype=int))
    df['h_unc']=unc_h
    df['h_acrp']=acrp_h
    df['oracle_h']=oracle_h
    df['oracle_lambda']=oracle_lambda
    print(f"Completed {scenario_name(s)}: {test} tests, {pilot} pilot histories per Oracle; "
          f"h*={oracle_h}, lambda*={oracle_lambda:.3f}, "
          f"elapsed={time.monotonic()-start:.1f}s",flush=True)
    return df


def run_one(s,test,pilot,force=False):
    """缓存与参数绑定；当 FORCE_RECOMPUTE=True 时从随机抽样重跑。"""
    name=scenario_name(s)
    mode=f'test{test}_pilot{pilot}_seed{C.SEED}_M{C.FV_PERIODS}'
    # QUICK and FORMAL share no file names; avoids mixing preliminary and official results.
    folder=DATA/('quick' if C.QUICK_MODE else 'formal')
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{name}_{mode}.csv'
    if path.exists() and not (force or C.FORCE_RECOMPUTE):
        old=pd.read_csv(path)
        if len(old)==test and all(col in old for col in METHODS):
            print('Cache hit:',path.name,flush=True)
            return old
    df=compute_scenario(s,test,pilot)
    df.to_csv(path,index=False,float_format='%.13g')
    return df


def summarize(df,s,exp,test,pilot):
    row=dict(experiment=exp,**s,tests=test,pilot=pilot,seed=C.SEED)
    for m in METHODS:
        a=df[m].to_numpy(float)
        row[m]=float(a.mean())
        row[f'sd_{m}']=float(a.std(ddof=1)) if test>1 else float('nan')
        row[f'se_{m}']=float(a.std(ddof=1)/np.sqrt(test)) if test>1 else float('nan')
    for baseline in ['Full-history ERM','Calibrated Full-history','Uncalibrated SW-FV',
                     'Oracle-SW','Oracle-EW']:
        diff=(df[baseline]-df['ACRP-FV']).to_numpy(float)
        mean=float(diff.mean())
        se=float(diff.std(ddof=1)/np.sqrt(test)) if test>1 else float('nan')
        row[f'gain_vs_{baseline}']=mean
        row[f'gain_L_{baseline}']=mean-1.96*se
        row[f'gain_U_{baseline}']=mean+1.96*se
    row['mean_h_acrp']=float(df['h_acrp'].mean())
    row['mean_h_unc']=float(df['h_unc'].mean())
    row['oracle_h']=int(df['oracle_h'].iloc[0])
    row['oracle_lambda']=float(df['oracle_lambda'].iloc[0])
    return row


def plot_three_panels(exp, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator,FixedFormatter,NullFormatter,LogLocator
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.titlesize':11.5,'axes.titleweight':'semibold','axes.labelsize':10.5,
        'xtick.labelsize':9.2,'ytick.labelsize':9.2,'savefig.facecolor':'white',
        'pdf.fonttype':42,'ps.fonttype':42})
    if exp=='A':
        panels=[(f'({chr(97+i)})  Mean sensitivity $\\bar{{\\theta}}$ = {mu:.2f}',
                 summary[np.isclose(summary.mu,mu)].sort_values('cv')) for i,mu in enumerate(C.A_MU)]
        xcol='cv';xticks=C.A_CV; xlabel='Time-sensitivity heterogeneity  CV(Θ)'
        title='Experiment A · Mean time sensitivity and heterogeneity'
    elif exp=='B':
        panels=[(f'({chr(97+i)})  Heterogeneity CV(Θ) = {cv:.1f}',
                 summary[np.isclose(summary.cv,cv)].sort_values('n')) for i,cv in enumerate(C.B_CV)]
        xcol='n';xticks=C.B_N;xlabel='Buyers per period  $n$'
        title='Experiment B · Buyers per period and heterogeneity'
    else:
        panels=[(f'({chr(97+i)})  Heterogeneity CV(Θ) = {cv:.1f}',
                 summary[np.isclose(summary.cv,cv)].sort_values('t')) for i,cv in enumerate(C.C_CV)]
        xcol='t';xticks=C.C_T;xlabel='Target period  $t$'
        title='Experiment C · Target period and historical accumulation'
    fig,axs=plt.subplots(1,3,figsize=(17.1,5.35),sharey=True)
    allvals=summary[METHODS].to_numpy(dtype=float)
    positive=allvals[(allvals>0)&np.isfinite(allvals)]
    if len(positive)==0:
        raise ValueError('All RRL values are non-positive or invalid; log axis unavailable.')
    minvalue,maxvalue=float(positive.min()),float(positive.max())
    ymin = 0.09
    ymax = 135.0
    major = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    visible = major
    visible=[x for x in major if ymin<=x<=ymax]
    for ax,(label,df) in zip(axs,panels):
        for m in METHODS:
            y=df[m].to_numpy(dtype=float)
            if np.any(y<=0):
                print(f'WARNING: {m}: non-positive RRL omitted on logarithmic scale')
                y=np.where(y>0,y,np.nan)
            ax.plot(df[xcol],y,color=COLORS[m],linestyle=STYLES[m],linewidth=2.2,
                    marker=MARKERS[m],markersize=4.6,markerfacecolor=COLORS[m],
                    markeredgecolor=COLORS[m],markeredgewidth=.8,label=m)
        ax.set_title(label,pad=13)
        ax.set_xlabel(xlabel,labelpad=8)
        ax.set_yscale('log')
        ax.set_ylim(ymin,ymax)
        ax.set_xticks(xticks)
        ax.yaxis.set_major_locator(FixedLocator(visible))
        ax.yaxis.set_major_formatter(FixedFormatter([f'{x:g}' for x in visible]))
        ax.yaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(1,10)*.1,numticks=100))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(which='major',axis='both',alpha=.34,color='#D9E1E7',linewidth=.8)
        ax.grid(which='minor',axis='y',alpha=.22,color='#E7EBEF',linewidth=.6)
        ax.tick_params(direction='out',length=3)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#64748B')
        ax.set_axisbelow(True)
    axs[0].set_ylabel('RRL (%)')
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,
               bbox_to_anchor=(.5,.015),fontsize=10,columnspacing=2.2,handlelength=3.)
    fig.suptitle(title,fontsize=13,weight='semibold',y=.984)
    if C.QUICK_MODE:
        fig.text(.5,.92,f'QUICK TEST ({C.QUICK_REPS} replications per stream) — NOT paper results',
                 ha='center',fontsize=9,color='#9B3D3D')
    fig.subplots_adjust(left=.06,right=.988,top=.84 if C.QUICK_MODE else .86,bottom=.23,wspace=.12)
    filename=f'Figure_7_2_{exp}' + ('_QUICK_ONLY' if C.QUICK_MODE else '')
    for ext in ('png','pdf','svg'):
        fig.savefig(FIG/f'{filename}.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    return filename


def run_experiment(exp):
    test,pilot=n_reps()
    target=DATA/('quick' if C.QUICK_MODE else 'formal')
    target.mkdir(exist_ok=True,parents=True)
    label='QUICK / not for paper' if C.QUICK_MODE else 'FORMAL'
    print(f'=== Experiment {exp} | {label} | test={test} pilot={pilot} EACH Oracle ===',flush=True)
    rows=[]
    ss=scenarios(exp)
    for idx,s in enumerate(ss,1):
        print(f'[{exp} {idx}/{len(ss)}] {scenario_name(s)}',flush=True)
        df=run_one(s,test,pilot)
        rows.append(summarize(df,s,exp,test,pilot))
        pd.DataFrame(rows).to_csv(target/f'summary_{exp}.csv',index=False)
    summary=pd.DataFrame(rows)
    filename=plot_three_panels(exp,summary)
    config={'experiment':exp,'quick':C.QUICK_MODE,'test':test,'pilot_per_oracle':pilot,
            'seed':C.SEED,'validation_periods':C.FV_PERIODS,'num_scenarios':len(ss),
            'note':'Monte Carlo newly simulated in this project; Python/Numba RNG does not match historical C++ seeds.'}
    (target/f'config_{exp}.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Experiment {exp} completed; figure: outputs/figures/{filename}.png',flush=True)
    return summary
