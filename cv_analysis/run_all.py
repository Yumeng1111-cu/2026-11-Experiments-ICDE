"""PyCharm one-click entry point.
Run tests; if raw MC results are present, rebuild CSV/Excel tables and figures.
This source-only repository has no precomputed MC outputs. Pass --run-simulation
to generate results before creating tables and figures.
"""
import subprocess,sys,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def call(cmd):
    print('RUN:',sys.executable, *cmd,flush=True)
    subprocess.run([sys.executable,*cmd],cwd=str(ROOT),check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-simulation',action='store_true',help='Recompute all formal MC data first, default false.')
    ap.add_argument('--mc',type=int,default=5000)
    ap.add_argument('--oracle-mc',type=int,default=5000)
    ap.add_argument('--seed',type=int,default=20331129)
    args=ap.parse_args()
    call(['tests/test_algorithms.py'])
    if args.run_simulation:
        call(['run_main_experiment.py','--mc',str(args.mc),'--oracle-mc',str(args.oracle_mc),'--seed',str(args.seed)])
    if not all((ROOT/'results/raw'/f'cv_{cv:.1f}_raw_results.csv').exists() for cv in (.2,.3,.4,.5)):
        raise SystemExit('Raw MC data are absent. Run: python run_all.py --run-simulation --mc 100 --oracle-mc 100 (smoke test), or use 5000 for formal results.')
    call(['build_tables.py'])
    call(['make_figures.py'])
    print('DONE. Saved CSV / Excel / Figures 1–3. Word and Figure 4 are not generated. No simulation rerun unless --run-simulation is passed.')

if __name__=='__main__':main()
