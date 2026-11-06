"""Full-information UCB-FV and Fixed-Share with the frozen 2026-10-20 grids."""
import numpy as np
from numba import njit

BETA_GRID=np.array([0.,.025,.05,.1,.2,.35,.5,.75,1.,1.5,2.])
FS_ETA=np.array([2.,8.,32.,128.,512.,2048.,4096.,8192.,16384.,32768.,40960.,65536.,98304.,131072.,196608.,262144.,524288.,1048576.])
FS_ALPHA=np.array([0.,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,6e-5,1e-4,2e-4,3e-4,6e-4,.001,.002,.003,.006,.01,.02,.03,.05,.1])

@njit(cache=True)
def swfi_ucb_joint_fv(reward, prices, n, M, betas, delta=.05):
    """Full-feedback adapted UCB, NOT original bandit-feedback SW-UCB.
    Only v<t reward updates; tie-break: smallest h, then smallest beta.
    """
    T,K=reward.shape
    target=T-1
    H=target-M
    num_b=len(betas)
    val=np.zeros((H,num_b))
    target_indices=np.zeros((H,num_b),np.int64)
    logterm=np.log(2.*K/delta)
    for v in range(target-M,target+1):
        cum=np.zeros(K)
        for h in range(1,H+1):
            for k in range(K):
                cum[k]+=reward[v-h,k]
            prefactor=np.sqrt(logterm/(2.*n*h))
            for ib in range(num_b):
                beta=betas[ib]
                best=-1.e100; chosen=0
                for k in range(K):
                    score=cum[k]/h+beta*prices[k]*prefactor
                    if score>best+1e-14:
                        best=score; chosen=k
                if v<target:
                    val[h-1,ib]+=reward[v,chosen]
                else:
                    target_indices[h-1,ib]=chosen
    best=-1.e100; hi=0; bi=0
    for h in range(H):
        for b in range(num_b):
            if val[h,b]>best+1e-12:
                best=val[h,b];hi=h;bi=b
    return target_indices[hi,bi],hi+1,betas[bi],best/M

@njit(cache=True)
def fixed_share_fast(loss, true_revenue, M, etas, alphas):
    """Numerically stable HW off-diagonal share; one log(p) per entry.
    Reproduces fixedshare_reference.fixed_share_one incl. tie breaking.
    """
    T,K=loss.shape
    best=-1.e100;bestrev=0.;bi=0;bj=0;bestval=0.
    for ie in range(len(etas)):
        eta=etas[ie]
        for ia in range(len(alphas)):
            alpha=alphas[ia]
            p=np.full(K,1./K)
            val=0.
            for t in range(T-1):
                if t>=T-1-M:
                    r=0.
                    for k in range(K): r+=p[k]*loss[t,k]
                    val+=5.*(1.-r)/M
                score=np.empty(K)
                zmax=-1.e100
                for k in range(K):
                    score[k]=np.log(max(p[k],1.e-300))-eta*loss[t,k]
                    if score[k]>zmax:zmax=score[k]
                z=np.empty(K);total=0.
                for k in range(K):
                    z[k]=np.exp(score[k]-zmax)
                    total+=z[k]
                for k in range(K):
                    q=z[k]/total
                    p[k]=(1.-alpha)*q+alpha*(1.-q)/(K-1)
            if val>best+1e-13:
                best=val;bi=ie;bj=ia;bestrev=0.
                for k in range(K):bestrev+=p[k]*true_revenue[k]
                bestval=val
    return bestrev,bi,bj,bestval

@njit(cache=True)
def fixed_share_final_meanprice(loss, eta, alpha, prices):
    """Compute expected price & highest-probability price at the target,
    using the same off-diagonal-share recursion as the reference implementation.
    Does NOT inspect target-period losses, and never uses true market demand.
    """
    T,K=loss.shape
    p=np.full(K,1./K)
    for t in range(T-1):
        score=np.empty(K); smax=-1.e300
        for k in range(K):
            score[k]=np.log(max(p[k],1.e-300))-eta*loss[t,k]
            if score[k]>smax:smax=score[k]
        ex=np.exp(score-smax)
        ss=np.sum(ex)
        for k in range(K):
            q=ex[k]/ss
            p[k]=(1.-alpha)*q+alpha*(1.-q)/(K-1)
    meanprice=0.; peak=0; highest=-1.
    for k in range(K):
        meanprice+=p[k]*prices[k]
        if p[k]>highest:
            highest=p[k];peak=k
    return meanprice,prices[peak]
