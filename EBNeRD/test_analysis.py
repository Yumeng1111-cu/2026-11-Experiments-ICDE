import csv
import json
import numpy as np
import pytest
from src.analysis import predict, fit_model, save_csv, dump_json
from check_reproduction import check


def test_fit_recovers_synthetic_decay_rate():
    h = np.arange(48)
    true_lambda = 0.042
    exposure = np.full(48, 400_000, dtype=np.int64)
    p = 0.12 * np.exp(-true_lambda * (h + 0.5 - 8))
    clicks = np.rint(exposure * p).astype(np.int64)
    fitted = fit_model(exposure, clicks, start=8, end=48)
    assert fitted['success']
    assert fitted['lambda_per_hour'] == pytest.approx(true_lambda, abs=0.0005)
    assert fitted['weighted_r2'] > 0.999
    assert np.allclose(predict(fitted, h[8:]), p[8:], rtol=0.01)


def test_csv_and_result_validation_without_private_dataset(tmp_path):
    exposure = np.full(48, 200_000, dtype=np.int64)
    h = np.arange(48)
    clicks = np.rint(exposure * (0.12 * np.exp(-0.04 * (h + 0.5 - 8)))).astype(int)
    fits = {
        'exp_0_48': fit_model(exposure, clicks, start=0),
        'exp_8_48': fit_model(exposure, clicks, start=8),
    }
    save_csv(tmp_path, exposure, clicks, fits)
    dump_json(tmp_path / 'fit_results.json', {
        'metadata': {'data_name': 'SYNTHETIC DEMO', 'total_exposures': int(exposure.sum()),
                     'total_clicks': int(clicks.sum())},
        'fits': fits,
    })
    check(tmp_path)
    with (tmp_path / 'hourly_ctr_48h.csv').open(encoding='utf-8-sig') as f:
        header = next(csv.reader(f))
    assert header == ['age_bin_start_h', 'exposures', 'clicks', 'ctr',
                      'exp_0_48_prediction', 'exp_8_48_prediction']
    assert 'article_id' not in header
