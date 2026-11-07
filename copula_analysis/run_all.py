#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entry point: run simulation optionally, then tables and Figures 1–3 only.

Examples:
    python run_all.py --run-simulation --mc 5000 --oracle-mc 5000 --seed 20521120 --cv 0.4 --data-root runs/cv04_5000
    python run_all.py --data-root runs/cv04_5000
"""
import argparse, subprocess, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def call(*args):
    cmd = [sys.executable, str(ROOT / args[0]), *args[1:]]
    print('RUN:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)

def main():
    pa = argparse.ArgumentParser(description='Gaussian copula pricing, six methods; no Word or Fig4')
    pa.add_argument('--run-simulation', action='store_true')
    pa.add_argument('--mc', type=int, default=1000)
    pa.add_argument('--oracle-mc', type=int, default=1000)
    pa.add_argument('--seed', type=int, default=20511120)
    pa.add_argument('--cv', type=float, default=None, help='CV of Gamma time-sensitivity; simulation defaults to 0.3')
    pa.add_argument('--data-root', type=str, default='results', help='Results directory; custom dirs also store figures in figures/ subdirectory')
    pa.add_argument('--rhos', type=float, nargs='+', default=None,
                    help='Optional Gaussian-copula rho values; only for --run-simulation')
    a = pa.parse_args()
    if a.run_simulation:
        cv=0.3 if a.cv is None else a.cv
        if not 0 < cv < 3: pa.error('--cv must be in (0,3)')
        call('test_copula.py', '--cv', str(cv))
        simargs=['simulate.py', '--mc', str(a.mc), '--oracle-mc', str(a.oracle_mc),
                 '--seed', str(a.seed), '--cv', str(cv), '--data-root', a.data_root]
        if a.rhos is not None:
            simargs += ['--rhos', *[str(r) for r in a.rhos]]
        call(*simargs)
    else:
        if a.rhos is not None:
            pa.error('--rhos is only used when --run-simulation is specified; existing output follows saved rho values')
        metafile=ROOT/a.data_root/'raw'/'experiment_config.json'
        if not metafile.is_file(): pa.error(f'No saved simulation found in {a.data_root}; add --run-simulation')
        cv=float(json.loads(metafile.read_text(encoding='utf-8'))['CV'])
        if a.cv is not None and abs(a.cv-cv)>1e-12:
            pa.error(f'Saved data are CV={cv}; you requested --cv {a.cv}. Use --run-simulation in a new --data-root.')
    call('build_tables.py', '--data-root', a.data_root)
    call('make_figures.py', '--data-root', a.data_root)
    print(f'DONE: results for CV={cv} at {a.data_root}. No Word or Figure 4.', flush=True)

if __name__ == '__main__':
    main()
