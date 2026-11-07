# Gaussian-Copula Robustness Experiments for Aging-Data Pricing

Reproducible Python experiments for evaluating dynamic pricing of aging data when **base valuation** and **time sensitivity** are dependent. A Gaussian copula introduces dependence while preserving the original marginal distributions. The project compares six pricing methods using independent Monte Carlo test histories, forward validation, and population-revenue evaluation.

> **Scope:** This repository contains the experimental source code and configuration. The uploaded archive did **not** contain previously generated simulation results; run the commands below to produce CSV tables and figures. No external dataset download is required for this simulation.

## Experiment setup

For a buyer in period \(t\), valuation is

\[
V_{i,t}=X_{i,t}\exp[-\Theta_{i,t}(t-1)].
\]

| Setting | Value |
| --- | --- |
| Base valuation \(X\) | Truncated logistic; location 5.2, scale 0.79, truncated to \([1,10]\) |
| Time sensitivity \(\Theta\) | Gamma with mean \(\bar\theta=0.042\), shape \(1/\mathrm{CV}^2\), scale \(0.042\,\mathrm{CV}^2\) |
| Dependence | Gaussian copula, default latent correlations \(\rho\in\{-0.6,-0.3,0,0.3,0.6\}\) |
| Default coefficient of variation | \(\mathrm{CV}=0.3\) (configurable) |
| Buyers per period | \(n=80\) |
| Target period | \(t=40\) |
| Forward-validation periods | \(M=10\) (periods 30–39) |
| Candidate prices | 81 logarithmically spaced values in \([0.01,5]\) |

**Important:** \(\rho\) is the correlation of the *latent Gaussian variables*, **not** the Pearson correlation of the observed \(X\) and \(\Theta\). When \(\rho=0\), their joint distribution is independent; their marginals remain unchanged for all tested \(\rho\). The code also reports empirical Pearson and Spearman correlations for reference.

### Methods and tuning

| Method | Implementation / tuning |
| --- | --- |
| **ACRP-FV** | Mean-decay calibration; sliding-window length \(h\in\{1,\ldots,29\}\) chosen by forward validation |
| **SW-FV** | Uncalibrated sliding window; \(h\in\{1,\ldots,29\}\) chosen by forward validation |
| **SW-FI-UCB-FV** | Full-information adaptation of sliding-window UCB; forward validation jointly selects \(h\in\{1,\ldots,29\}\) and 11 confidence-bonus coefficients \(\beta\); \(\delta=0.05\) is fixed |
| **Fixed-Share** | Candidate prices are experts; forward validation jointly selects 18 learning rates \(\eta\) and 21 sharing rates \(\alpha\) |
| **Calibrated Full-history** | Mean-decay calibration using all 39 historical periods |
| **Oracle-SW** | A fixed calibrated window \(h\in\{1,\ldots,39\}\) selected from an **independent training simulation**; not a deployable policy or theoretical upper bound |

The exact \(\beta\), \(\eta\), and \(\alpha\) grids are defined in [`src/competitors.py`](src/competitors.py) and recorded in [`config/frozen_config.json`](config/frozen_config.json). They are not retuned between correlation scenarios. SW-FI-UCB-FV uses **complete historical bid feedback**, and should not be confused with the original bandit-feedback SW-UCB algorithm.

Fixed-Share is evaluated using the expected revenue of its randomized pricing distribution. All methods are evaluated using true population revenue under the corresponding Gaussian-copula market, *not* a demand model that assumes independence. Relative revenue loss is

\[
\mathrm{RRL}(\%)=100\,\frac{r^*-r(\widehat p)}{r^*},
\]

where \(r^*\) is approximated on a separate dense 10,001-price grid. For Fixed-Share, \(r(\widehat p)\) in this expression denotes the randomized policy's **expected** revenue rather than the revenue at its mean price.

## Requirements

- Python **3.11 or 3.12** recommended (other versions depend on NumPy / SciPy / Numba compatibility).
- Install Python dependencies from `requirements.txt`.
- Optional: `artifact_tool` for formatted `.xlsx` exports. The experiment always generates CSV summaries even without it.

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick start

Run the numerical consistency check before a simulation:

```bash
python test_copula.py --cv 0.4
```

**Small smoke test** (separate output directory; two Monte Carlo repetitions per \(\rho\)):

```bash
python run_all.py --run-simulation --mc 2 --oracle-mc 2 --cv 0.4 --seed 11 --data-root runs/smoke_cv04
```

**Main experiment**, with 1,000 test replications and 1,000 independent Oracle-SW training replications *per correlation level*:

```bash
python run_all.py --run-simulation --mc 1000 --oracle-mc 1000 --cv 0.3 --seed 20511120 --data-root runs/cv03_1000
```

**Higher-replication experiment**, using \(\mathrm{CV}=0.4\):

```bash
python run_all.py --run-simulation --mc 5000 --oracle-mc 5000 --cv 0.4 --seed 20521120 --data-root runs/cv04_5000
```

**Custom correlation values** (each value must satisfy \(-1<\rho<1\)):

```bash
python run_all.py --run-simulation --mc 1000 --oracle-mc 1000 --cv 0.4 --rhos -0.6 -0.3 0 0.3 0.6 --data-root runs/custom_rhos
```

To **rebuild tables and figures without rerunning simulations**, point to an existing output directory:

```bash
python run_all.py --data-root runs/cv03_1000
```

Or invoke either stage individually:

```bash
python build_tables.py --data-root runs/cv03_1000
python make_figures.py --data-root runs/cv03_1000
python verify_results.py --data-root runs/cv03_1000
```

> Use a **different** `--data-root` for each new run. Reusing a location while invoking `--run-simulation` overwrites saved outputs there. `--rhos` affects simulations only; simply rebuilding figures will use the correlations recorded in the output metadata.

## Output files

For `--data-root runs/cv03_1000`, the program writes:

```text
runs/cv03_1000/
├── raw/
│   ├── experiment_config.json
│   ├── all_methods_by_rho.csv
│   ├── selected_parameters_by_rho.csv
│   ├── oracle_training_windows.csv
│   └── rho_market_details.csv
├── summary/
│   ├── summary_rrl.csv
│   ├── summary_rrl_wide.csv
│   ├── paired_differences.csv
│   ├── window_summary.csv
│   ├── parameter_summary.csv
│   ├── market_details.csv
│   └── correlation_<MC>mc_results.xlsx    # only if artifact_tool is available
└── figures/
    ├── Fig1_six_methods_RRL_by_rho.{png,pdf,svg}
    ├── Fig2_paired_RRL_difference_by_rho.{png,pdf,svg}
    └── Fig3_window_length_by_rho.{png,pdf,svg}
```

The figures show mean RRL and 95% normal-approximation intervals, paired RRL differences with intervals, and selected history-window lengths. CSV tables include method-level means, Monte Carlo standard errors, and paired comparisons. No Word document or fourth figure is generated.

## Reproducibility notes

- **Causality:** All prices and hyperparameters are selected using historical bids only; the target period's buyer bids are not observed during selection.
- **Independent Oracle training:** Oracle-SW's fixed window is selected using a separate Monte Carlo sample from that used to evaluate methods.
- **Paired evaluation:** All six methods use the same test history for a given replication and \(\rho\).
- **Common random numbers across \(\rho\):** The Gaussian latent draws are reused across the correlation conditions. Accordingly, estimates for different \(\rho\) values are *not* independent.
- **Deterministic seeds:** `--seed` controls the random-number streams. Save the generated `raw/experiment_config.json` with reported results.
- **Numerical checks:** `test_copula.py` checks the \(\rho=0\) independence formula and quadrature stability; `verify_results.py` validates saved output counts, finite values, and parameter ranges.
- **Frozen tuning:** Hyperparameter candidate sets are unchanged across scenarios; see `config/frozen_config.json`.

## Repository layout

```text
.
├── README.md                   # This documentation
├── requirements.txt            # Python dependencies
├── .gitignore                  # Excludes generated results and caches
├── config/
│   ├── frozen_config.json      # Recorded market settings and parameter grids
│   └── plot_style.json         # Figure styles
├── src/
│   ├── acrp_core.py            # Calibration and forward-validation pricing
│   ├── competitors.py          # Full-feedback UCB and Fixed-Share
│   ├── copula_market.py        # Copula sampling and population revenues
│   └── fixedshare_reference.py # Fixed-Share reference implementation
├── run_all.py                  # Pipeline entry point
├── simulate.py                 # Simulation and raw results
├── build_tables.py             # Statistical summaries and optional Excel export
├── make_figures.py             # Figures 1–3
├── test_copula.py              # Mathematical / numerical check
└── verify_results.py           # Saved-output integrity checks
```

## Background

The comparison implementations adapt sliding-window UCB (Garivier and Moulines, 2011) and Fixed-Share (Herbster and Warmuth, 1998) to full-information bid-feedback pricing. The experiment is a simulation informed by fitted marginal parameters; it does **not** include original FCC or EB-NeRD datasets.
