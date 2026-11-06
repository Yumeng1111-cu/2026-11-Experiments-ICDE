"""Check bundled 7.2 datasets and figures without installing dependencies."""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'outputs' / 'data' / 'formal'
FIG = ROOT / 'outputs' / 'figures'
EXPECTED = {'A': 12, 'B': 18, 'C': 21}
REQUIRED = {'mu','cv','n','t','Full-history ERM','Calibrated Full-history',
            'Uncalibrated SW-FV','ACRP-FV','Oracle-SW','Oracle-EW'}


def main() -> None:
    checked = set()
    for exp, n_expected in EXPECTED.items():
        summary = DATA / f'summary_{exp}.csv'
        config = DATA / f'config_{exp}.json'
        assert summary.is_file(), f'Missing: {summary}'
        assert config.is_file(), f'Missing: {config}'
        cfg = json.loads(config.read_text(encoding='utf-8'))
        with summary.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            assert REQUIRED <= set(reader.fieldnames or []), f'Missing columns: {summary}'
            rows = list(reader)
        assert len(rows) == n_expected == cfg['num_scenarios'], f'Wrong row count: {summary}'
        assert cfg['quick'] is False, f'Not formal data: {summary}'
        for row in rows:
            name = (f"mu{float(row['mu']):.3f}_cv{float(row['cv']):.2f}_"
                    f"n{int(row['n'])}_t{int(row['t'])}")
            fn = f"{name}_test{cfg['test']}_pilot{cfg['pilot_per_oracle']}_seed{cfg['seed']}_M{cfg['validation_periods']}.csv"
            scenario_file = DATA / fn
            assert scenario_file.is_file(), f'Missing scenario CSV: {scenario_file}'
            checked.add(fn)
        for ext in ('png','pdf','svg'):
            fig = FIG / f'Figure_7_2_{exp}.{ext}'
            assert fig.is_file() and fig.stat().st_size > 0, f'Missing figure: {fig}'
        print(f'PASS experiment {exp}: {len(rows)} settings, figure PNG/PDF/SVG')
    print(f'PASS formal dataset: {sum(EXPECTED.values())} configurations, {len(checked)} distinct scenario CSVs')
    print('Existing quick-mode files are retained but not included in formal validation.')


if __name__ == '__main__':
    main()
