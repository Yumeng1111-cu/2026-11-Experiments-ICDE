# EB-NeRD Small: Publication-Age Click-Through-Rate Decay

A reproducible **research-code-only** pipeline for studying how candidate news
click-through rate changes with article age in **EB-NeRD Small**. The pipeline
extracts hourly aggregate statistics from licensed local Parquet files, fits
exponential decay models, and computes article-cluster bootstrap uncertainty.

> **Data and licensing:** This repository contains **no EB-NeRD dataset**, no
> article-level derived tables or identifiers, and no real-data-derived results.
> The EB-NeRD license limits permitted uses; permission for use outside news
> recommendation research should not be assumed. See
> [Data access and licensing](docs/DATA_ACCESS_AND_LICENSE.md) before running
> or publicly releasing derived results.

## Quick start (Windows / PyCharm / Linux / macOS)

Use Python **3.10 or later**, create a local virtual environment if desired:

```bash
python -m pip install -r requirements.txt
```

### Offline demo (no EB-NeRD files needed)

```bash
python demo.py
python plot_results.py --input results/demo
python check_reproduction.py --output results/demo
```

All demo values are **synthetic and not EB-NeRD measurements**. The scripts
create figures and outputs under `results/demo/`, which is ignored by Git.

### Reproduce the research analysis from licensed files

Download **EB-NeRD Small** through its official website, accepting its license:
https://recsys.eb.dk/

Place the official ZIP in `data/ebnerd_small.zip`, then run:

```bash
python run.py --zip data/ebnerd_small.zip --bootstrap 500
python plot_results.py --input results
python check_reproduction.py --output results
```

Or point to locally extracted files:

```bash
python run.py --data-dir /path/to/extracted/ebnerd_small --bootstrap 500
```

For a quick pipeline check, use `--bootstrap 0` (does not provide bootstrap
intervals). The ZIP must contain `articles.parquet`, `train/behaviors.parquet`,
and `validation/behaviors.parquet` at its root.

### What's generated locally?

- `results/hourly_ctr_48h.csv`: 48 hourly aggregate exposure, click, and CTR bins
- `results/fit_results.json`: model statistics and article-bootstrap percentiles
- `results/ctr_decay_actual_vs_exponential.png` / `.svg`: full-window and later-window diagnostics
- `results/ctr_decay_publication.pdf` / `.png` / `.svg`: observed series and 8–48 h fit


## Method at a glance

1. Combine the EB-NeRD Small `train` and `validation` behavior periods.
2. Keep articles whose publication times allow a complete 48 h observation
   opportunity within the combined behavior-log time span.
3. Group candidate exposures and clicks by integer hour since publication:
   one `article_ids_inview` entry is one candidate exposure; it is counted
   as clicked when the same ID appears in `article_ids_clicked` for that
   impression.
4. Pool clicks / exposures in each of the 48 age bins to obtain hourly CTR.
5. Estimate `p_h = c * exp[-lambda * (h + 0.5 - start_h)]` using binomial
   maximum likelihood, separately for start hours 0, 3, 6, 8, 10 and 12.
6. Report exposure-weighted in-sample R² and article-resampled bootstrap
   percentiles (500 resamples by default).

The principal later-period model uses the **8–48 h** window; the early period
can rise before decaying, so a single exponential over 0–48 h need not be a
good approximation. The 8 h starting point was chosen after inspecting the
pattern and the reported fit is **in-sample**. Crucially, **CTR is not a direct
measurement of willingness to pay or monetary valuation**.

## Repository structure

```text
.
├── README.md
├── README_zh-CN.md
├── requirements.txt
├── requirements-dev.txt
├── run.py                      # official data -> aggregate stats/fits
├── demo.py                     # offline synthetic demonstration
├── plot_results.py             # plot from locally generated aggregates
├── check_reproduction.py      # aggregate consistency & optional reference
├── Run_in_Windows.bat
├── src/
│   ├── reader.py               # streaming Parquet reader
│   ├── analysis.py             # aggregation, fitting, bootstrap
│   └── figures.py              # diagnostic figures
├── tests/                       # synthetic-only tests
├── docs/
│   ├── METHODOLOGY.md
│   └── DATA_ACCESS_AND_LICENSE.md
├── data/README.md              # instructions only
└── results/README.md           # instructions only
```

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

GitHub Actions runs these tests and the synthetic demo without downloading or
accessing any EB-NeRD data.

## Citation and permissions

Please cite the original EB-NeRD dataset paper and follow the license provided
by Ekstra Bladet, rather than treating the term "open source dataset" as a
blanket right to redistribute or repurpose the files.

- Official dataset: https://recsys.eb.dk/
- License terms: https://recsys.eb.dk/assets/pdf/general_license_terms.pdf
- See [Data access and licensing](docs/DATA_ACCESS_AND_LICENSE.md).

