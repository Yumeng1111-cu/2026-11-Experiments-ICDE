"""PyCharm friendly entry point. No Word document and no Figure 4.

python run_all.py --run-simulation --mc 100 --oracle-mc 100 --cv 0.4
python run_all.py --run-simulation --mc 5000 --oracle-mc 5000 --cv 0.5 --data-root runs/cv05_5000
python run_all.py --data-root runs/distribution_cv04_pilot
"""
import sys,argparse,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def call(name,*arguments):
    command=[sys.executable,str(ROOT/name),*map(str,arguments)]
    print('RUN:',' '.join(command),flush=True)
    subprocess.run(command,cwd=ROOT,check=True)

def main():
    pa=argparse.ArgumentParser(description='X independent of theta; theta marginal shape experiment.')
    pa.add_argument('--run-simulation',action='store_true')
    pa.add_argument('--mc',type=int,default=100);pa.add_argument('--oracle-mc',type=int,default=100)
    pa.add_argument('--cv',type=float,default=None);pa.add_argument('--seed',type=int,default=20601121)
    pa.add_argument('--data-root',default='runs/distribution_cv04_pilot')
    a=pa.parse_args()
    if a.run_simulation:
        cv=.4 if a.cv is None else a.cv
        call('tests/test_market.py','--cv',cv)
        call('simulate.py','--mc',a.mc,'--oracle-mc',a.oracle_mc,'--cv',cv,
             '--seed',a.seed,'--data-root',a.data_root)
    else:
        p=ROOT/a.data_root/'raw/experiment_config.json'
        if not p.exists(): pa.error('Saved simulation not found; add --run-simulation.')
        cv=json.loads(p.read_text(encoding='utf-8'))['CV']
        if a.cv is not None and abs(cv-a.cv)>1e-12:
            pa.error('The saved dataset has a different CV. Specify --run-simulation and separate --data-root.')
    call('build_tables.py','--data-root',a.data_root)
    call('make_figures.py','--data-root',a.data_root)
    print('DONE: no Word, no Figure 4. CV=',cv,flush=True)
if __name__=='__main__':main()
