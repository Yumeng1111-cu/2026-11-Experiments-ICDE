from settings import QUICK_MODE
import pandas as pd
from core import SIGMAS, SAMPLE_SIZES, N, LAMBDAS, DATA, reps, run_scenario, calc_bound


def optima(sigma,n,pilot):
    sw,_=run_scenario(sigma,n,'pilot_sw',pilot)
    _,ew=run_scenario(sigma,n,'pilot_ew',pilot)
    sw_best=1+int(sw.argmax())
    ew_best=float(LAMBDAS[ew.argmax()])
    bound_h,bound_lam=calc_bound(sigma,n)
    return sw_best,bound_h,ew_best,bound_lam


def main():
    _,pilot=reps()
    rows_sig=[]
    for sigma in SIGMAS:
        hr,hb,lr,lb=optima(sigma,N,pilot)
        rows_sig.append((sigma,N,hr,hb,lr,lb))
    rows_n=[]
    for n in SAMPLE_SIZES:
        hr,hb,lr,lb=optima(.02,n,pilot)
        rows_n.append((.02,n,hr,hb,lr,lb))
    cols=['sigma_theta','n','h_revenue','h_bound','lambda_revenue','lambda_bound']
    a=pd.DataFrame(rows_sig,columns=cols)
    b=pd.DataFrame(rows_n,columns=cols)
    mode = "quick" if QUICK_MODE else "formal"
    output_dir = DATA / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    a.to_csv(output_dir / "experiment3_heterogeneity.csv", index=False)
    b.to_csv(output_dir / "experiment3_sample_size.csv", index=False)
    print(f"Experiment 3 data generated: {output_dir}", flush=True)
    print("Heterogeneity comparison:\n", a.to_string(index=False))
    print("Sample-size comparison:\n", b.to_string(index=False))


if __name__ == "__main__":
    main()
