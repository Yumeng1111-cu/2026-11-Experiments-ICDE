# Dynamic Pricing for Aging Data: Time-Sensitivity Heterogeneity (CV) Experiments

This repository contains the Python implementation of the **coefficient-of-variation (CV) comparison experiment** for dynamic pricing of aging data with heterogeneous buyer time sensitivities. It evaluates **ACRP-FV** against five benchmarks as the heterogeneity of buyer time sensitivity increases.

> **Data availability:** The source archive supplied for this repository contains code only, not the previously computed 5,000-run Monte Carlo (MC) CSV/Excel tables or figures. Generate the outputs using the commands below. There is no external raw dataset required: valuation and time-sensitivity samples are simulated locally.

## Experimental setting

For buyer \(i\) in period \(t\), valuation follows

\[
V_{i,t}=X_{i,t}\exp[-\Theta_{i,t}(t-1)].
\]

- **Base valuation:** truncated Logistic distribution on \([1,10]\), with location \(5.2\) and scale \(0.79\).
- **Time sensitivity:** Gamma distribution with mean \(\bar\theta=0.042\) and coefficient of variation \(CV\in\{0.2,0.3,0.4,0.5\}\). Gamma shape is \(1/CV^2\) and scale is \(\bar\theta CV^2\).
- **Independence:** \(X\) and \(\Theta\) are independently sampled.
- **Evaluation period:** \(t=40\); **buyers per period:** \(n=80\); **forward-validation periods:** \(M=10\).
- **Candidate prices:** 81 logarithmically spaced points between 0.01 and 5.
- **Reported metric:** mean **relative revenue loss (RRL, %)** at the evaluation period, compared with the continuous-price population optimum.

All implementable policies choose their price (or pricing distribution) **before** observing period-40 bids. Population revenue is used for evaluation and for the independent Oracle-SW training procedure, not to tune implementable methods.

## Compared methods

| Method | Description | Selection rule |
| --- | --- | --- |
| **ACRP-FV** | Mean-decay calibration followed by adaptive sliding-window pricing | Forward validation over \(h=1,\ldots,29\) |
| **SW-FV** | Uncalibrated adaptive sliding-window pricing | Forward validation over \(h=1,\ldots,29\) |
| **SW-FI-UCB-FV** | Full-information adaptation of sliding-window UCB | Joint forward validation over \(h\in\{1,\ldots,29\}\) and 11 bonus coefficients \(\beta\); \(\delta=0.05\) |
| **Fixed-Share** | Exponential expert weighting with off-diagonal weight sharing | Forward validation over 18 learning rates \(\eta\) and 21 sharing rates \(\alpha\) |
| **Calibrated Full-history** | Mean-decay calibration using all 39 previous periods | No window tuning |
| **Oracle-SW** | Independently trained benchmark for selecting a fixed window | Independent simulated histories, \(h=1,\ldots,39\) |

**Important implementation notes:** `SW-FI-UCB-FV` is a custom **full-feedback** adaptation, not the original bandit-feedback SW-UCB implementation; its literature guarantees are not automatically applicable. Fixed-Share produces a randomized final pricing policy, evaluated using its population **expected revenue**. Oracle-SW is a reference within the class of fixed-window strategies; it is not an upper bound on all adaptive strategies.

The full frozen candidate grids are recorded in [`config/frozen_config.json`](config/frozen_config.json). Plot colors and markers can be edited in [`config/plot_style.json`](config/plot_style.json) without changing simulation results.

## Requirements

- Python **3.11 or 3.12** recommended.
- Install the dependencies from `requirements.txt`.
- `artifact_tool` is **optional** for formatted `.xlsx` export. If it is unavailable, `build_tables.py` still generates the summary CSV files.

```bash
python -m pip install -r requirements.txt
```

## How to run

Run these commands from the repository root (e.g. the PyCharm terminal).

**Quick smoke test** (generates a small dataset for all four CV levels, then builds summary tables and plots):

```bash
python run_all.py --run-simulation --mc 5 --oracle-mc 5
```

**Reproduce the formal experiment** (5,000 test MC histories and 5,000 independent Oracle training histories **per CV**):

```bash
python run_all.py --run-simulation --mc 5000 --oracle-mc 5000 --seed 20331129
```

**Rebuild tables and plots without rerunning Monte Carlo**, after all four CV result files already exist:

```bash
python run_all.py
```

You may also run each stage separately:

```bash
python tests/test_algorithms.py
python run_main_experiment.py --mc 5000 --oracle-mc 5000 --seed 20331129
python build_tables.py
python make_figures.py
```

To simulate a single CV scenario (e.g., in a separate process):

```bash
python run_main_experiment.py --cv 0.4 --mc 5000 --oracle-mc 5000 --seed 20331129
```

`build_tables.py` and `make_figures.py` expect outputs for **all four CV values**. When running CVs separately, complete all four before running these two scripts. Because `build_tables.py` uses fixed output names containing `5000MC`, avoid treating a small smoke-test output as a formal 5,000-run result; the `MC` column in the summary records the actual number of repetitions.

## Project structure

```text
CV_github/
├── README.md                 # Project documentation (the only README)
├── requirements.txt          # Python dependencies
├── .gitignore                # Excludes generated data, plots, and local caches
├── config/
│   ├── frozen_config.json    # Frozen market settings and hyperparameter grids
│   └── plot_style.json       # Shared plot appearance across experiments
├── run_all.py                # End-to-end entry point
├── run_main_experiment.py    # Independent Monte Carlo simulation
├── build_tables.py           # Summary CSVs and optional Excel workbook
├── make_figures.py           # Figures in PNG, PDF, and SVG
├── src/
│   ├── market.py             # Sample generation and population revenue integration
│   ├── acrp_core.py          # Decay calibration and forward validation
│   ├── competitors.py        # UCB and accelerated Fixed-Share
│   └── fixedshare_reference.py  # Reference Fixed-Share for testing
└── tests/
    └── test_algorithms.py    # Consistency and sanity checks
```

Directories under `results/` and `figures/` are created when the pipeline runs and are excluded from Git via `.gitignore`.

## Output files

After successful simulation and post-processing:

- `results/raw/cv_*.csv` (including per-repetition results, selected parameters, and independent Oracle training scores), with companion `cv_*.json` metadata and progress logs.
- `results/raw/all_5000MC_six_method_results.csv` and `all_5000MC_selected_parameters.csv` (combined records; filenames retain the original script convention).
- `results/summary/summary_rrl.csv`, `summary_rrl_wide.csv`, `paired_differences.csv`, `window_summary.csv`, `parameter_summary.csv`, `parameter_frequencies.csv`, and `oracle_summary.csv`.
- Optional `results/summary/formal_results.xlsx` when `artifact_tool` is available.
- `figures/Fig1_six_methods_RRL.*`, `Fig1b_detail_excluding_UCB.*`, `Fig2_paired_RRL_difference.*`, and `Fig3_window_selection.*`, each in PNG, PDF, and SVG format.

The project does **not** generate a Word report or Figure 4.

## Reproducibility and interpretation

- Each MC repetition uses a **shared simulated historical-bid panel** across all six methods, allowing paired comparison within each CV scenario.
- Each CV scenario receives a different deterministic seed offset. Oracle-SW uses separate simulated training histories to select its fixed window.
- Frozen grids, seeds, number of buyers, price candidates, and evaluation settings are recorded in the code, configuration, and output metadata.
- The implementation evaluates **single-period per-buyer expected revenue loss**, not cumulative bandit regret.
- `tests/test_algorithms.py` checks that the optimized Fixed-Share routine agrees with its reference implementation and performs basic algorithm sanity checks.

No raw FCC or EB-NeRD records are distributed with this repository. Their distributions motivate the simulation parameters, but this project only generates synthetic samples.
