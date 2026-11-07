from settings import QUICK_MODE
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from core import T, SIGMAS, PRICE_MIN, PRICE_MAX, DATA, demand


def main():
    prices = np.unique(np.r_[np.geomspace(PRICE_MIN, PRICE_MAX, 1400),
                             np.linspace(PRICE_MIN, PRICE_MAX, 1400)])
    rows=[]
    for sigma in SIGMAS:
        current = demand(prices,T-1,sigma)
        for d in range(T):
            if d==0 or sigma==0:
                mismatch=0.0
            else:
                hist=demand(prices,T-1-d,sigma, d)
                delta=np.abs(current-hist)
                k=int(np.argmax(delta))
                l,r=prices[max(0,k-1)],prices[min(len(prices)-1,k+1)]
                opt=minimize_scalar(lambda p:-abs(float(demand([p],T-1,sigma)[0]-demand([p],T-1-d,sigma,d)[0])),
                                    bounds=(l,r),method='bounded')
                mismatch=max(float(delta[k]),float(-opt.fun))
            rows.append((sigma,d,mismatch))
    df=pd.DataFrame(rows,columns=['sigma_theta','historical_distance_d','population_mismatch'])
    mode = "quick" if QUICK_MODE else "formal"
    output_dir = DATA / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "experiment1_mismatch.csv"
    df.to_csv(path, index=False)
    print(f"Experiment 1 data generated: {path}", flush=True)


if __name__ == "__main__":
    main()
