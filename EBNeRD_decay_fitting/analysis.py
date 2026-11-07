from __future__ import annotations
import csv
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from .reader import read_articles, read_impression_range, iter_behaviors

US_PER_HOUR = 3_600_000_000


def timestamp_iso(t_us):
    return (datetime(1970, 1, 1) + timedelta(microseconds=int(t_us))).strftime('%Y-%m-%d %H:%M:%S')


def aggregate(data_root: Path, hours: int, verbose=True):
    articles = data_root / 'articles.parquet'
    behaviors = [data_root / 'train' / 'behaviors.parquet', data_root / 'validation' / 'behaviors.parquet']
    for p in [articles, *behaviors]:
        if not p.is_file():
            raise FileNotFoundError(f'Missing required parquet file: {p}')
    publication, n_articles = read_articles(articles)
    earliest, latest, splits = read_impression_range(behaviors)
    cohort = {a: t for a, t in publication.items() if earliest <= t <= latest - hours * US_PER_HOUR}
    if not cohort:
        raise RuntimeError(f'No articles with a full {hours}h coverage opportunity.')
    counts_by_article = {}
    unmatched_click_ids = 0
    counted_candidate_rows = 0
    for path in behaviors:
        n_rows = 0
        for time_us, in_view, clicked in iter_behaviors(path):
            n_rows += 1
            clickset = set(clicked)
            # Exposure is each candidate appearing in the in-view list, not a behavior row.
            # Count a click only if a candidate ID also exists in its own clicked list.
            for a in in_view:
                pub = cohort.get(a)
                if pub is None:
                    continue
                h = (time_us - pub) // US_PER_HOUR
                if not 0 <= h < hours:
                    continue
                counted_candidate_rows += 1
                m = counts_by_article.get(a)
                if m is None:
                    m = np.zeros((hours, 2), dtype=np.int64)
                    counts_by_article[a] = m
                m[int(h), 0] += 1
                m[int(h), 1] += int(a in clickset)
        if verbose:
            print(f'  {path.parent.name}: read {n_rows:,} behavior rows', flush=True)
    observed_ids = np.array(list(counts_by_article), dtype=np.int64)  # insertion order, matches initial research run
    if not len(observed_ids):
        raise RuntimeError('No observed candidate exposures after filtering.')
    article_counts = np.stack([counts_by_article[int(i)] for i in observed_ids])
    exposure = article_counts[:, :, 0].sum(axis=0)
    clicks = article_counts[:, :, 1].sum(axis=0)
    assert int(exposure.sum()) == counted_candidate_rows
    info = {
        'data_name': 'EB-NeRD Small', 'hours': hours, 'article_metadata_rows': n_articles,
        'cohort_articles': len(cohort), 'observed_articles': len(observed_ids),
        'total_exposures': int(exposure.sum()), 'total_clicks': int(clicks.sum()),
        'pooled_ctr': float(clicks.sum() / exposure.sum()),
        'log_start': timestamp_iso(earliest), 'log_end': timestamp_iso(latest),
        'splits': {k: {'rows': v['rows'], 'start': timestamp_iso(v['start_us']), 'end': timestamp_iso(v['end_us'])} for k, v in splits.items()},
        'peak_hour': int(np.argmax(np.divide(clicks, exposure, out=np.zeros(hours, dtype=float), where=exposure > 0))),
        'candidate_definition': 'One (behavior, inview_article_id) per candidate; clicked when ID in same behavior clicked list.',
    }
    return exposure, clicks, article_counts, observed_ids, info


def fit_model(exposure, clicks, start=8, end=None, min_exposures=100):
    if end is None:
        end = len(exposure)
    idx = np.arange(start, end, dtype=np.int32)
    exposure = np.asarray(exposure)
    clicks = np.asarray(clicks)
    valid = exposure[start:end] >= min_exposures
    hours = idx[valid]
    n = exposure[hours].astype(float)
    y = clicks[hours].astype(float)
    if len(hours) < 6:
        raise ValueError(f'{start}–{end}h: fewer than six time bins with >= {min_exposures} exposures')
    x = hours.astype(float) - start + 0.5
    observed = y / n

    def objective(params):
        c, lam = params
        p = np.clip(c * np.exp(-lam * x), 1e-12, 1 - 1e-12)
        return float(-np.sum(y * np.log(p) + (n - y) * np.log1p(-p)))

    res = minimize(objective, [0.13, 0.04], method='L-BFGS-B',
                   bounds=[(1e-9, 0.999), (0.0, 1.0)],
                   options={'maxiter': 600, 'ftol': 1e-12})
    if not np.all(np.isfinite(res.x)):
        raise RuntimeError(f'Nonfinite fitted parameters for exponential model {start}–{end}')
    c, lam = map(float, res.x)
    pred = c * np.exp(-lam * x)
    wmean = float(np.average(observed, weights=n))
    sst = float(np.sum(n * (observed - wmean) ** 2))
    sse = float(np.sum(n * (observed - pred) ** 2))
    r2 = float(1 - sse / sst) if sst else float('nan')
    rmse = float(math.sqrt(sse / n.sum()))
    return {
        'start_h': int(start), 'end_h': int(end),
        'c': c, 'lambda_per_hour': lam,
        'half_life_hours': float(math.log(2) / lam) if lam > 0 else None,
        'weighted_r2': r2, 'weighted_rmse': rmse,
        'negative_log_likelihood': float(res.fun),
        'aic': float(2 * 2 + 2 * res.fun), 'used_hour_bins': int(len(hours)),
        'exposures_in_fit': int(n.sum()), 'success': bool(res.success),
        'optimizer_message': str(res.message),
    }


def predict(fit, h):
    h = np.asarray(h, dtype=float)
    return fit['c'] * np.exp(-fit['lambda_per_hour'] * (h + 0.5 - fit['start_h']))


def bootstrap(exposure_by_article: np.ndarray, n_rep: int = 500, seed: int = 20261020, min_exposures: int = 100):
    """Article-cluster bootstrap of the exponential rate for each start hour."""
    counts = exposure_by_article[:, :, 0]
    clicks = exposure_by_article[:, :, 1]
    n_articles, horizon = counts.shape
    rng = np.random.default_rng(seed)
    report = {'n_requested': n_rep, 'seed': seed, 'resampling_unit': 'article', 'fits': {}}
    starts = [0, 3, 6, 8, 10, 12]
    saved = {f'exp_{st}_{horizon}': [] for st in starts if st < horizon}
    for j in range(n_rep):
        sampled = rng.multinomial(n_articles, np.full(n_articles, 1 / n_articles))
        e = sampled @ counts
        c = sampled @ clicks
        for st in starts:
            if st >= horizon:
                continue
            key = f'exp_{st}_{horizon}'
            try:
                fit = fit_model(e, c, start=st, end=horizon, min_exposures=min_exposures)
                if fit['success'] and np.isfinite(fit['lambda_per_hour']):
                    saved[key].append(fit['lambda_per_hour'])
            except (RuntimeError, ValueError, FloatingPointError):
                pass
        if (j + 1) % 100 == 0:
            print(f'  Bootstrap {j + 1}/{n_rep}', flush=True)
    for key, xs in saved.items():
        vals = np.asarray(xs, dtype=float)
        report['fits'][key] = {
            'successful': int(len(vals)),
            'lambda_percentiles_2_5_50_97_5': [float(z) for z in np.percentile(vals, [2.5, 50, 97.5])] if len(vals) else None,
        }
    return report


def save_csv(output: Path, exposure, clicks, fits):
    n = len(exposure)
    fitted = {name: predict(f, np.arange(n)) for name, f in fits.items()}
    path = output / f'hourly_ctr_{n}h.csv'
    keys = [f'exp_0_{n}', f'exp_8_{n}']
    with path.open('w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['age_bin_start_h', 'exposures', 'clicks', 'ctr'] + [name + '_prediction' for name in keys])
        for h in range(n):
            ctr = clicks[h] / exposure[h] if exposure[h] else None
            prediction = [f'{fitted[key][h]:.9f}' if key in fitted and h >= fits[key]['start_h'] else '' for key in keys]
            w.writerow([h, int(exposure[h]), int(clicks[h]), f'{ctr:.9f}' if ctr is not None else ''] + prediction)
    return path


def dump_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
