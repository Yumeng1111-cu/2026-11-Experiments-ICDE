"""Check consistency of locally regenerated *aggregate* outputs.

No private dataset or expected research numbers are bundled in this repository.
Optional --reference accepts a user-local, previously generated fit_results.json.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def check(output: Path, reference: Path | None = None):
    with (output / 'fit_results.json').open(encoding='utf-8') as f:
        summary = json.load(f)
    with (output / 'hourly_ctr_48h.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 48, 'Expected 48 hours of aggregated results'
    assert [int(r['age_bin_start_h']) for r in rows] == list(range(48))
    e = sum(int(r['exposures']) for r in rows)
    c = sum(int(r['clicks']) for r in rows)
    assert e > 0 and 0 <= c <= e, 'Invalid aggregate exposure/click counts'
    meta = summary['metadata']
    if 'total_exposures' in meta:
        assert e == meta['total_exposures'], 'Exposure total mismatch'
        assert c == meta['total_clicks'], 'Click total mismatch'
    fit = summary['fits']['exp_8_48']
    assert fit['start_h'] == 8 and fit['end_h'] == 48
    assert 0 < fit['c'] < 1 and 0 <= fit['lambda_per_hour'] <= 1
    assert np.isfinite(fit['weighted_r2'])
    if reference:
        with reference.open(encoding='utf-8') as f:
            ref = json.load(f)
        for key in ('total_exposures', 'total_clicks', 'cohort_articles', 'observed_articles'):
            assert meta.get(key) == ref['metadata'].get(key), f'Reference mismatch: {key}'
        for name in ('exp_0_48', 'exp_3_48', 'exp_6_48', 'exp_8_48', 'exp_10_48', 'exp_12_48'):
            if name not in ref['fits'] or name not in summary['fits']:
                continue
            for metric in ('lambda_per_hour', 'c', 'weighted_r2'):
                assert np.isclose(summary['fits'][name][metric], ref['fits'][name][metric],
                                  rtol=1e-3, atol=1e-5), f'Reference mismatch: {name}.{metric}'
    print(f'PASS: 48 bins, {e:,} aggregate exposures, {c:,} aggregate clicks')
    print(f'8–48 h: lambda={fit["lambda_per_hour"]:.6f}/h, weighted R²={fit["weighted_r2"]:.4f}')


def main():
    p = argparse.ArgumentParser(description='Validate local aggregate reproducibility')
    p.add_argument('--output', type=Path, default=ROOT / 'results')
    p.add_argument('--reference', type=Path, default=None,
                   help='Optional local original fit_results.json, not included in Git repo')
    a = p.parse_args()
    check(a.output, a.reference)


if __name__ == '__main__':
    main()
