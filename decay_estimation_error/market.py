"""Data aging market. This module alone knows the simulation's true distributions."""
import numpy as np
from scipy.special import expit, logit, gammainc, roots_legendre

LOC=5.2
SCALE=0.79
LOW=1.
HIGH=10.
MEAN_THETA=.042
PRICE_GRID=np.geomspace(.01,5.,81)


def sample_histories(rng, B, n, cv, t=40):
    """Sample ONLY the t-1 observable historic bid cohorts; pad dummy target row.
    No simulated target bids are accessible in the learned policies.
    shape returned (B,t,n) for legacy verified ACRP core; last row is dummy.
    """
    flo=expit((LOW-LOC)/SCALE)
    fhi=expit((HIGH-LOC)/SCALE)
    u=flo+(fhi-flo)*rng.random((B,t-1,n))
    x=LOC+SCALE*logit(u)
    theta=rng.gamma(shape=1./cv**2, scale=MEAN_THETA*cv**2, size=(B,t-1,n))
    ages=np.arange(t-1).reshape((1,t-1,1))
    logv=np.log(x)-theta*ages
    padded=np.empty((B,t,n), dtype=np.float64)
    padded[:,:-1,:]=np.sort(logv,axis=2)
    padded[:,-1,:]=-1000.  # target placeholder, mathematically never observed by learner
    return padded


def population_revenues(prices, cv, age=39, quadrature=96):
    """Population expected revenue p*P(X exp(-age*Theta) >=p).
    Use exact truncated logistic density and gamma CDF, X-axis Gauss-Legendre.
    Used ONLY by evaluator and independent Oracle-SW parameter selection.
    """
    prices=np.asarray(prices, dtype=np.float64)
    if np.any(prices<=0) or np.any(prices>=10):
        raise ValueError('Price must lie in (0,10)')
    k=1./cv**2
    theta_scale=MEAN_THETA*cv**2
    nodes,weights=roots_legendre(quadrature)
    low=np.maximum(prices,LOW)
    half=(HIGH-low)/2.
    middle=(HIGH+low)/2.
    xx=middle[:,None]+half[:,None]*nodes[None,:]
    flo=expit((LOW-LOC)/SCALE)
    fhi=expit((HIGH-LOC)/SCALE)
    F=expit((xx-LOC)/SCALE)
    pdf=F*(1-F)/(SCALE*(fhi-flo))
    gcut=np.maximum(0.,np.log(xx/prices[:,None])/(age*theta_scale))
    demand=half*np.sum(weights[None,:]*pdf*gammainc(k,gcut),axis=1)
    return prices*demand
