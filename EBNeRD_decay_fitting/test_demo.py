import json
from demo import main as demo_main
from plot_results import load_results, make_plot


def test_demo_is_explicitly_synthetic(tmp_path, monkeypatch):
    import demo
    monkeypatch.setattr(demo, 'ROOT', tmp_path)
    demo_main()
    folder = tmp_path / 'results' / 'demo'
    with (folder / 'fit_results.json').open(encoding='utf-8') as f:
        report = json.load(f)
    assert report['metadata']['data_name'] == 'SYNTHETIC DEMO'
    assert 'NOT EB-NeRD' in report['metadata']['warning']
    hours, observed, fit, summary = load_results(folder)
    assert len(hours) == len(observed) == 48
    make_plot(folder, folder)
    for suffix in ('.pdf','.png','.svg'):
        assert (folder / f'ctr_decay_publication{suffix}').is_file()
    assert not list(folder.glob('*.npz'))
