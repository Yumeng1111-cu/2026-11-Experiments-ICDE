"""Independent-market mathematical checks; accepts custom CV from CLI."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from independent_market import (theoretical_moments,theta_from_uniform,theta_parameters,
    population_revenues,PRICES,generate_latents,sample_histories_from_uniforms,
    truncated_logistic_survival)
pa=argparse.ArgumentParser();pa.add_argument('--cv',type=float,default=.4);cv=pa.parse_args().cv
for dist in ['gamma','lognormal','two-point']:
    mu,c=theoretical_moments(dist,cv)
    assert abs(mu-.042)<1.e-12 and abs(c-cv)<1.e-11,(dist,mu,c)
    p=population_revenues(PRICES,dist,cv,quadrature=144)
    assert np.all(np.isfinite(p)) and np.all(p>=0.) and np.all(p<=PRICES),dist
    if dist!='two-point':
        p2=population_revenues(PRICES,dist,cv,quadrature=200)
        assert np.max(abs(p-p2))<2.e-6,(dist,np.max(abs(p-p2)))
    rng=np.random.default_rng(9128)
    u1,u2=generate_latents(rng,3)
    logs=sample_histories_from_uniforms(u1,u2,dist,cv)
    assert logs.shape==(3,40,80) and np.isfinite(logs).all()
    assert np.all(np.diff(logs[:,:39],axis=2)>=0)
# CDF quantile two point exactly matches analytic 50/50 form
p=theta_parameters('two-point',cv)
t=theta_from_uniform(np.array([.25,.75]),'two-point',cv)
assert np.allclose(t,[p['theta_low'],p['theta_high']])
# Analytical gamma independence expectation matches normalized-truncated-X quadrature.
from scipy.special import roots_legendre, expit, gammainc
from independent_market import X_LOC,X_SCALE,F_LOW,F_HIGH
nodes,w=roots_legendre(160)
lo=np.maximum(PRICES,1.);half=(10.-lo)/2
x=(10.+lo)[:,None]/2+half[:,None]*nodes
FX=expit((x-X_LOC)/X_SCALE)
pdf=FX*(1.-FX)/(X_SCALE*(F_HIGH-F_LOW))
cut=np.maximum(0.,np.log(x/PRICES[:,None])/(39.*.042*cv**2))
check=PRICES*half*np.sum(w[None,:]*pdf*gammainc(1./cv**2,cut),axis=1)
assert np.max(abs(check-population_revenues(PRICES,'gamma',cv)))<1.e-10
print('PASS: moments/independence sampler/population revenue/quadrature/two-point support; CV=',cv)
