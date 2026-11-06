from settings import QUICK_MODE
import numpy as np
import pandas as pd
from core import SIGMAS, T, N, LAMBDAS, DATA, reps, truth_grid, run_scenario


def main():
    test_count,_ = reps()
    rows=[]
    for sigma in SIGMAS:
        sw,ew=run_scenario(sigma,N,'test',test_count)
        _,_,r_star=truth_grid(sigma)
        swloss=100*(r_star-sw)/r_star
        ewloss=100*(r_star-ew)/r_star
        for h,value in enumerate(swloss,1):
            rows.append((sigma,'SW',h,value))
        for lam,value in zip(LAMBDAS,ewloss):
            rows.append((sigma,'EW',lam,value))
    df=pd.DataFrame(rows,columns=['sigma_theta','method','memory_parameter','relative_revenue_loss_percent'])
    mode = "quick" if QUICK_MODE else "formal"
    output_dir = DATA / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "experiment2_tradeoff.csv"
    df.to_csv(path, index=False)
    print(f"Experiment 2 data generated: {path}", flush=True)


if __name__ == "__main__":
    main()
