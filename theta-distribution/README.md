# Time-Sensitivity Distribution Robustness Experiment

Reproducible Python implementation for evaluating dynamic pricing of aging data under **alternative distributions of buyer time sensitivity**. The experiment varies the shape of the time-sensitivity distribution while holding its mean and coefficient of variation (CV) fixed. Base valuation and time sensitivity remain independent.

This directory is one experiment from the ACRP-FV research codebase. **The supplied archive contains source code only**: no Monte Carlo runs, tables, figures, or external/raw datasets are included. Generate results locally using the commands below.

## Experimental design

Buyer valuation for data of age $a=t-1$ is

$$V_{i,t}=X_{i,t}\exp(-\Theta_{i,t}a).$$

Base valuation follows a truncated Logistic distribution, $X\sim\mathrm{Logistic}(5.2,0.79)$ conditioned on $X\in[1,10]$. The buyer-specific time sensitivity $\Theta$ is independent of $X$, with default mean $\bar\theta=0.042$ and $\mathrm{CV}(\Theta)=0.4$.

The three candidate time-sensitivity distributions have the **same theoretical mean and CV**:

| Distribution | Parameterization |
| --- | --- |
| Gamma | Shape $k=1/\mathrm{CV}^2$; scale $s=\bar\theta\,\mathrm{CV}^2$ |
| Lognormal | $\sigma_{\log}^2=\ln(1+\mathrm{CV}^2)$; $\mu_{\log}=\ln(\bar\theta)-\sigma_{\log}^2/2$ |
| Two-point | $\Theta=\bar\theta(1-\mathrm{CV})$ or $\bar\theta(1+\mathrm{CV})$, each with probability $1/2$ |

The supported CV range is $0<\mathrm{CV}\le 1$ to keep the symmetric two-point distribution nonnegative.

Default settings:

| Setting | Value |
| --- | --- |
| Target period | $t=40$ (39 historical periods) |
| Buyers per period | $n=80$ |
| Forward-validation periods | $M=10$ (periods 30–39) |
| Candidate prices | 81 logarithmically spaced prices in $[0.01,5]$ |
| FV window choices | $h\in\{1,\dots,29\}$ |
| Oracle-SW fixed-window choices | $h\in\{1,\dots,39\}$ |
| Default Monte Carlo runs | 100 test histories and 100 independent Oracle training histories **per distribution** |

Methods compared: **ACRP-FV**, **SW-FV**, **SW-FI-UCB-FV** (a full-bid-feedback adaptation of sliding-window UCB), **Fixed-Share**, **Calibrated Full-history**, and **Oracle-SW**. Frozen tuning grids are specified in [`config/frozen_config.json`](config/frozen_config.json). For SW-FI-UCB-FV, $\beta\in\{0,0.025,0.05,0.1,0.2,0.35,0.5,0.75,1,1.5,2\}$, with $\delta=0.05$; Fixed-Share uses 18 learning-rate and 21 sharing-rate candidates.

**Evaluation.** At period 40, each method is evaluated using population revenue $r_{40}(p)=p\,\Pr(V_{i,40}\ge p)$ for the corresponding distribution. Relative revenue loss (RRL) is measured against the best population revenue on a **10,001-point fine price grid** in $[0.01,5]$. Two-point demand is evaluated as an exact mixture; continuous distributions use numerical quadrature. Oracle-SW selects its fixed window using **separate simulated training histories**, not the evaluation histories.

To reduce cross-distribution simulation noise, the three scenarios share underlying uniform random draws; the draws for $X$ and $\Theta$ are independent. All six methods within a scenario evaluate the same test histories.

## Requirements

- Python 3.11 or 3.12 recommended (other versions may work with compatible dependencies).
- Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Optional: `artifact_tool` enables `.xlsx` export by `build_tables.py`. **CSV summaries do not require it.**

## Quick start

From the project root (the directory containing `run_all.py`), run a small experiment and produce summaries and figures:

```bash
python run_all.py --run-simulation --mc 100 --oracle-mc 100 --cv 0.4 --seed 20601121 --data-root runs/distribution_cv04_pilot
```

Run a larger experiment (5,000 test and 5,000 independent Oracle training histories *per distribution*):

```bash
python run_all.py --run-simulation --mc 5000 --oracle-mc 5000 --cv 0.4 --seed 20621121 --data-root runs/distribution_cv04_5000
```

The pipeline checks the market model, runs simulation, summarizes results, and produces figures. For large MC counts, runtime and memory use increase substantially. Preserve the frozen tuning grids for direct comparison with the original experiments.

**Rebuild summaries and figures without rerunning simulation** (after generating the `raw/` files):

```bash
python build_tables.py --data-root runs/distribution_cv04_pilot
python make_figures.py --data-root runs/distribution_cv04_pilot
```

Alternatively, `python run_all.py --data-root runs/distribution_cv04_pilot` rebuilds both after saved raw data exist.

**Generate a selected-method plot**, without overwriting the main figures:

```bash
python make_figures.py --data-root runs/distribution_cv04_pilot --methods ACRP-FV SW-FV --only-selected
```

Plot style is configured in [`config/plot_style.json`](config/plot_style.json). Use `--no-ci` to suppress the 95% confidence intervals in Figure 1.

## Output structure

Generated files are written to the chosen `--data-root` and are excluded from Git by `.gitignore`:

```text
runs/<experiment>/
├── raw/
│   ├── experiment_config.json
│   ├── all_methods_by_distribution.csv
│   ├── selected_parameters_by_distribution.csv
│   ├── oracle_training_windows.csv
│   └── market_details.csv
├── summary/
│   ├── RRL_Wide.csv
│   ├── RRL_Statistics.csv
│   ├── Paired_Difference.csv
│   ├── Window_Statistics.csv
│   ├── Hyperparameter.csv
│   └── Market_Diagnostics.csv
└── figures/
    ├── Fig1_RRL_by_theta_distribution.{png,pdf,svg}
    ├── Fig1_zoom_without_full_history.{png,pdf,svg}
    ├── Fig2_paired_RRL_difference.{png,pdf,svg}
    └── Fig3_window_length.{png,pdf,svg}
```

An Excel workbook may also appear in `summary/` if the optional spreadsheet export dependency is installed.

## Repository layout

```text
theta_distribution/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   ├── frozen_config.json
│   └── plot_style.json
├── run_all.py
├── simulate.py
├── build_tables.py
├── make_figures.py
├── src/
│   ├── independent_market.py
│   ├── acrp_core.py
│   ├── competitors.py
│   └── fixedshare_reference.py
└── tests/
    └── test_market.py
```

- `simulate.py`: samples independent market histories, runs all six methods, and saves per-run results and tuned parameters.
- `src/independent_market.py`: implements the three $\Theta$ distributions, independent sampling, and true population revenue.
- `src/acrp_core.py`: price selection for calibrated/uncalibrated window methods and calibrated full history.
- `src/competitors.py`: full-feedback UCB adaptation and Fixed-Share implementation.
- `build_tables.py`: summary metrics and paired confidence intervals.
- `make_figures.py`: plots from saved summaries without rerunning simulation.
- `tests/test_market.py`: parameter-matching, numerical integration, and basic sampler checks.

## Verification and reproducibility notes

Run the market-model checks independently:

```bash
python tests/test_market.py --cv 0.4
```

The simulation saves its seed, parameter settings, distribution labels, and experiment configuration with the raw outputs. For statistical conclusions intended for publication, prefer independent large-sample runs; the default 100-run setting is a pilot configuration. Distinct experiments should use separate `--data-root` directories to avoid overwriting results.

**Implementation note:** `SW-FI-UCB-FV` uses full historical bid feedback. It is an adaptation of SW-UCB, not the original bandit-feedback algorithm, and the original SW-UCB theoretical guarantees should not be transferred automatically.
