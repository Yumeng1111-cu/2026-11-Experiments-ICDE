"""Independent X and Theta, matched theta mean+CV; Gaussian, Lognormal, equal-weight two-point.

No Gaussian copula is used.  Every (X, Theta) pair is independently drawn.
Theta distribution changes only; truncated-logistic marginal of X stays unchanged.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit, logit, ndtr, gammainc, roots_legendre
from scipy.stats import gamma, lognorm

X_LOC=5.2; X_SCALE=.79; X_LOW=1.; X_HIGH=10.
MEAN_THETA=.042; DEFAULT_CV=.4; N=80; T=40; M=10
PRICES=np.geomspace(.01,5.,81)
DISTRIBUTIONS=('gamma','lognormal','two-point')
F_LOW=expit((X_LOW-X_LOC)/X_SCALE)
F_HIGH=expit((X_HIGH-X_LOC)/X_SCALE)

def check_cv(cv):
    if not 0 < cv <= 1:
        raise ValueError('0 < CV <= 1 required: symmetric equal-probability nonnegative two-point distribution')

def theta_parameters(dist,cv):
    check_cv(cv)
    if dist=='gamma':
        return {'shape':1/cv**2,'scale':MEAN_THETA*cv**2}
    if dist=='lognormal':
        sigma=float(np.sqrt(np.log1p(cv**2)))
        return {'mu_log':float(np.log(MEAN_THETA)-sigma**2/2),'sigma_log':sigma}
    if dist=='two-point':
        return {'theta_low':MEAN_THETA*(1-cv),'theta_high':MEAN_THETA*(1+cv),'prob_each':0.5}
    raise ValueError('Unknown distribution: '+str(dist))

def theta_from_uniform(u,dist,cv):
    pr=theta_parameters(dist,cv)
    u=np.clip(u,np.finfo(float).eps,1.-np.finfo(float).eps)
    if dist=='gamma': return gamma.ppf(u,pr['shape'],scale=pr['scale'])
    if dist=='lognormal': return lognorm.ppf(u,s=pr['sigma_log'],scale=np.exp(pr['mu_log']))
    # Probability theta=low and theta=high are both exactly 1/2
    return np.where(u<.5,pr['theta_low'],pr['theta_high'])

def theta_cdf(x,dist,cv):
    """Analytic theta CDF (nonnegative support), vectorized."""
    pr=theta_parameters(dist,cv)
    x=np.asarray(x,dtype=float)
    if dist=='gamma': return gammainc(pr['shape'],np.maximum(x,0.)/pr['scale'])
    if dist=='lognormal':
        xx=np.maximum(x,np.finfo(float).tiny)
        return np.where(x>0.,ndtr((np.log(xx)-pr['mu_log'])/pr['sigma_log']),0.)
    return np.where(x<pr['theta_low'],0.,np.where(x<pr['theta_high'],.5,1.))

def generate_latents(rng,mc,n=N,periods=T-1):
    """Independent uniforms; common across the three theta distribution scenarios."""
    return rng.random((mc,periods,n)),rng.random((mc,periods,n))

def sample_histories_from_uniforms(ux,ut,dist,cv):
    """Historical sorted log-bids shape (B,T,N); last period dummy, not observable."""
    if ux.shape!=ut.shape or ux.ndim!=3: raise ValueError('uniforms shape mismatch')
    X=X_LOC+X_SCALE*logit(F_LOW+(F_HIGH-F_LOW)*ux)
    theta=theta_from_uniform(ut,dist,cv)
    age=np.arange(ux.shape[1])
    log_bid=np.log(X)-theta*age[None,:,None]
    b,t,n=ux.shape
    result=np.empty((b,t+1,n),dtype=float)
    result[:,:-1,:]=np.sort(log_bid,axis=2)
    result[:,-1,:]=-1000.
    return result

def truncated_logistic_survival(z):
    """P(X >= z) for truncated Logistic X supported on [1,10]."""
    z=np.asarray(z,dtype=float)
    v=(F_HIGH-expit((z-X_LOC)/X_SCALE))/(F_HIGH-F_LOW)
    return np.where(z<=X_LOW,1.,np.where(z>=X_HIGH,0.,np.clip(v,0.,1.)))

def population_revenues(prices,dist,cv,age=T-1,quadrature=160):
    """EXACT two-point mixture or high accuracy independent continuous quadrature.

    D(p) = integral f_X(x) F_theta( log(x/p)/age ) dx,
    with f_X the *normalized truncated* Logistic density.
    """
    prices=np.asarray(prices,dtype=float)
    if np.any(prices<=0) or np.any(prices>=X_HIGH): raise ValueError('Prices must be between 0 and 10')
    if age <= 0: return prices * truncated_logistic_survival(prices)
    if dist=='two-point':
        pr=theta_parameters(dist,cv)
        demand=.5*truncated_logistic_survival(prices*np.exp(pr['theta_low']*age)) \
              +.5*truncated_logistic_survival(prices*np.exp(pr['theta_high']*age))
        return prices*demand
    nodes,weights=roots_legendre(quadrature)
    lo=np.maximum(prices,X_LOW)
    half=(X_HIGH-lo)/2
    x=(X_HIGH+lo)[:,None]/2 + half[:,None]*nodes[None,:]
    FX=expit((x-X_LOC)/X_SCALE)
    fX=FX*(1.-FX)/(X_SCALE*(F_HIGH-F_LOW))
    cut=np.maximum(np.log(x/prices[:,None])/age,0.)
    cdf=theta_cdf(cut,dist,cv)
    prob=half*np.sum(weights[None,:]*fX*cdf,axis=1)
    return prices*prob

def theoretical_moments(dist,cv):
    p=theta_parameters(dist,cv)
    if dist=='two-point':
        mean=.5*(p['theta_low']+p['theta_high'])
        variance=.5*((p['theta_low']-mean)**2+(p['theta_high']-mean)**2)
    elif dist=='gamma':
        mean=p['shape']*p['scale'];variance=p['shape']*p['scale']**2
    else:
        mean=float(np.exp(p['mu_log']+.5*p['sigma_log']**2))
        variance=float((np.exp(p['sigma_log']**2)-1)*np.exp(2*p['mu_log']+p['sigma_log']**2))
    return mean,np.sqrt(variance)/mean
