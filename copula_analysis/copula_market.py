# -*- coding: utf-8 -*-
"""Gaussian-copula market with EXACT original truncated-logistic & Gamma marginals.

Population demand: Gauss-Legendre integration over X and conditional-normal
probability of Theta <= log(x/p)/age given Z_X=Phi^-1(F_X(x)).
The actual Pearson correlation Corr(X,Theta) is NOT equal to copula rho.
"""
import numpy as np
from scipy.special import expit, logit, ndtr, ndtri, gammainc, roots_legendre
from scipy.stats import gamma

LOC=5.2
SCALE=.79
LOW=1.
HIGH=10.
MEAN_THETA=.042
CV=.3
N=80
T=40
M=10
RHO_VALUES=(-.6,-.3,0.,.3,.6)
PRICES=np.geomspace(.01,5.,81)
F_LOW=expit((LOW-LOC)/SCALE)
F_HIGH=expit((HIGH-LOC)/SCALE)


def generate_latents(rng, total, target_period=T, n=N):
    """Common random numbers; train and test MUST use separate streams."""
    shape=(total,target_period-1,n)
    return rng.standard_normal(shape),rng.standard_normal(shape)


def sample_histories_from_latents(z1,z2,rho,cv=CV):
    """(B,40,80) sorted log-bids, last row dummy, never learner-visible."""
    assert -1<rho<1
    assert z1.shape==z2.shape and z1.ndim==3
    ux=ndtr(z1)
    utheta=ndtr(rho*z1+np.sqrt(1.-rho*rho)*z2)
    # Avoid infinite inverse Gamma in the extreme tails, no practical mass change.
    utheta=np.clip(utheta,np.finfo(float).eps,1.-np.finfo(float).eps)
    x=LOC+SCALE*logit(F_LOW+(F_HIGH-F_LOW)*ux)
    theta=gamma.ppf(utheta,a=1./cv**2,scale=MEAN_THETA*cv**2)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(theta))
    b,t1,n=z1.shape
    log_bid=np.log(x)-theta*np.arange(t1)[None,:,None]
    out=np.empty((b,t1+1,n),dtype=np.float64)
    out[:,:-1,:]=np.sort(log_bid,axis=2)
    out[:,-1,:]=-1000.0
    return out


def population_revenues_gaussian(prices,rho,cv=CV,age=T-1,quadrature=128):
    """Joint-distribution population revenue, not independent-demand formula.

    For each X=x, Z_theta|Z_X=z_x ~ N(rho*z_x,1-rho²).
    P(theta<=c | x) = Phi((Phi^-1(F_Gamma(c))-rho*z_x)/sqrt(1-rho²)).
    At rho=0 this reduces to the product-marginal demand formula.
    """
    prices=np.asarray(prices,dtype=np.float64)
    if np.any(prices<=0) or np.any(prices>=HIGH):
        raise ValueError('Prices must be in (0,10)')
    nodes,weights=roots_legendre(quadrature)
    xlow=np.maximum(prices,LOW)
    half=(HIGH-xlow)/2.
    xx=(HIGH+xlow)[:,None]/2.+half[:,None]*nodes[None,:]
    Fx=expit((xx-LOC)/SCALE)
    pdfx=Fx*(1.-Fx)/(SCALE*(F_HIGH-F_LOW))
    thmax=np.maximum(0.,np.log(xx/prices[:,None])/age)
    gamma_cdf=gammainc(1./cv**2,thmax/(MEAN_THETA*cv**2))
    if abs(rho)<1e-14:
        conditional=gamma_cdf
    else:
        zx=ndtri((Fx-F_LOW)/(F_HIGH-F_LOW))
        q=ndtri(gamma_cdf)
        conditional=ndtr((q-rho*zx)/np.sqrt(1.-rho*rho))
    demand=half*np.sum(weights[None,:]*pdfx*conditional,axis=1)
    return prices*demand


def reported_correlations(rng,rho,cv=CV,n_sample=50000):
    """Empirical Pearson and Spearman for reporting, separate diagnostic seed."""
    from scipy.stats import rankdata
    z1=rng.standard_normal(n_sample); z2=rng.standard_normal(n_sample)
    u=ndtr(z1)
    x=LOC+SCALE*logit(F_LOW+(F_HIGH-F_LOW)*u)
    th=gamma.ppf(ndtr(rho*z1+np.sqrt(1-rho**2)*z2),a=1./cv**2,scale=MEAN_THETA*cv**2)
    pearson=np.corrcoef(x,th)[0,1]
    spearman=np.corrcoef(rankdata(x),rankdata(th))[0,1]
    return float(pearson),float(spearman)
