from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile
from src.analysis import aggregate, fit_model, bootstrap, save_csv, dump_json
from src.figures import create_figures

PROJECT = Path(__file__).resolve().parent


def ensure_data(zip_path: Path, extraction: Path):
    needed = [Path('articles.parquet'), Path('train/behaviors.parquet'), Path('validation/behaviors.parquet')]
    if all((extraction / f).is_file() for f in needed):
        return extraction
    if not zip_path.is_file():
        raise FileNotFoundError(f'No dataset: {zip_path}. Place ebnerd_small.zip in data/ or pass --data-dir to extracted folder.')
    extraction.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as z:
        members = {s.filename for s in z.infolist()}
        for file in needed:
            name = file.as_posix()
            if name not in members:
                raise RuntimeError(f'ZIP does not contain {name}: {zip_path}')
            dest = extraction / file
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                continue
            print(f'  extract {name}', flush=True)
            with z.open(name) as r, dest.open('wb') as w:
                shutil.copyfileobj(r, w, length=1024 * 1024)
    return extraction


def main():
    parser = argparse.ArgumentParser(description='From EB-NeRD Small ZIP to hourly CTR, exponential fits, Bootstrap and figures.')
    parser.add_argument('--zip', dest='zip_file', type=Path, default=PROJECT/'data'/'ebnerd_small.zip', help='Path to a locally downloaded, officially licensed EB-NeRD Small ZIP')
    parser.add_argument('--data-dir', type=Path, default=None, help='Already-extracted data directory containing articles.parquet')
    parser.add_argument('--output', type=Path, default=PROJECT/'results', help='Output folder')
    parser.add_argument('--bootstrap', type=int, default=500, help='Article bootstrap replicates, default 500; 0 for quick test')
    parser.add_argument('--seed', type=int, default=20261020, help='Bootstrap RNG seed')
    parser.add_argument('--min-exposures', type=int, default=100)
    args=parser.parse_args()
    if args.bootstrap < 0:
        parser.error('--bootstrap must be >= 0')
    if args.min_exposures < 1:
        parser.error('--min-exposures must be >= 1')
    if args.data_dir is not None and not args.data_dir.is_dir():
        parser.error(f'--data-dir does not exist: {args.data_dir}')
    data_root=args.data_dir if args.data_dir else ensure_data(args.zip_file, PROJECT/'data'/'extracted')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    print('\n[1/5] Reading data and aggregating 48 publication-age hours...', flush=True)
    exp, cli, arrays, _article_ids, metadata=aggregate(data_root, 48)
    print(f"  {metadata['observed_articles']:,} observed articles, {metadata['total_exposures']:,} inviews, {metadata['total_clicks']:,} clicks",flush=True)
    print('[2/5] Binomial maximum likelihood fits...',flush=True)
    fits={}
    for start in [0,3,6,8,10,12]:
        fits[f'exp_{start}_48']=fit_model(exp,cli,start=start,end=48,min_exposures=args.min_exposures)
    mainfit=fits['exp_8_48']
    print(f"  8-48h: lambda={mainfit['lambda_per_hour']:.6f}/h, half-life={mainfit['half_life_hours']:.2f}h, weighted R²={mainfit['weighted_r2']:.4f}",flush=True)
    print('[3/5] Article-cluster bootstrap...',flush=True)
    if args.bootstrap:
        boot=bootstrap(arrays, n_rep=args.bootstrap, seed=args.seed, min_exposures=args.min_exposures)
    else:
        boot={'n_requested':0,'seed':args.seed,'resampling_unit':'article','fits':{}}
    print('[4/5] Writing LOCAL aggregate outputs (ignored by Git)...',flush=True)
    csvfile=save_csv(output, exp, cli, fits)
    # Per-article IDs and counts are intentionally NOT written to disk.
    dump_json(output/'fit_results.json',{'metadata':metadata,'fits':fits,'bootstrap':boot,
        'notes':['CTR is not monetary valuation or willingness to pay.',
        'Post-8h window selected after seeing the empirical curve: in-sample fit only.',
        'Full 48h does not exhibit simple monotone exponential decay.']})
    create_figures(output, exp, cli, fits)
    print('[5/5] Completed. Outputs are local-only and must not be committed without permission.',flush=True)
    print('\nSUCCESS: generated files:',flush=True)
    for name in ('ctr_decay_actual_vs_exponential.png','ctr_decay_actual_vs_exponential.svg',csvfile.name,'fit_results.json'):
        print('  '+str(output/name), flush=True)


if __name__=='__main__':
    main()
