"""Independent check of uncorrelated exact marginal demand and copula sampler."""
import sys
import argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import numpy as np
from copula_market import population_revenues_gaussian,PRICES,generate_latents,sample_histories_from_latents
from scipy.special import gammainc,expit
from scipy.stats import logistic, gamma
from scipy.special import roots_legendre

pa=argparse.ArgumentParser()
pa.add_argument('--cv', type=float, default=0.3)
args=pa.parse_args()
assert args.cv>0
p1=population_revenues_gaussian(PRICES,0.,cv=args.cv,quadrature=160)
nodes,w=roots_legendre(160)
# independent formula
mu=5.2;s=.79;f0=logistic.cdf(1,loc=mu,scale=s);f1=logistic.cdf(10,loc=mu,scale=s)
a=np.maximum(PRICES,1);dx=(10-a)/2
x=(10+a)[:,None]/2+dx[:,None]*nodes
F=logistic.cdf(x,loc=mu,scale=s)
pdf=F*(1-F)/(s*(f1-f0))
thcut=np.maximum(0,np.log(x/PRICES[:,None])/(39*.042*args.cv**2))
demand=dx*(w*pdf*gammainc(1/args.cv**2,thcut)).sum(axis=1)
assert np.max(np.abs(p1-PRICES*demand))<1e-11
for rho in [-.6,-.3,0.,.3,.6]:
  a=population_revenues_gaussian(PRICES,rho,cv=args.cv,quadrature=96)
  b=population_revenues_gaussian(PRICES,rho,cv=args.cv,quadrature=160)
  assert max(abs(a-b))<2e-6, (rho,max(abs(a-b)))
  assert np.all(b>=-1e-13) and np.all(b<=PRICES+1e-13)
print('PASS: independence limiting case, quadrature convergence, valid revenues')
