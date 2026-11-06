# ACRP-FV: Robustness to Mean Time-Sensitivity Estimation Errors

Reproducible Python implementation of a Monte Carlo study investigating how errors in the **estimated mean time sensitivity** affect pricing for aging data. The experiments compare **ACRP-FV** with five reference methods under a fixed true market distribution.

## Research question

Buyers have heterogeneous time sensitivities. The seller observes historical bids but calibrates them using an estimated average decay rate, which may differ from the true mean. Does forward-validation-based history selection reduce the revenue impact of this misspecification?

The true mean decay rate is fixed at `0.042`; the seller's assumed rate is

```text
theta_assumed = 0.042 * (1 + relative_error)
```

Only the **assumed** rate changes across scenarios. The true distribution and the evaluation target do not.

## Experimental setup

| Setting | Value |
| --- | --- |
| Target period | `t = 40` (age 39) |
| Buyers per period | `n = 80` |
| Validation periods | `M = 10` |
| True mean time sensitivity | `0.042` |
| Time-sensitivity distribution | Gamma, CV = `0.4` |
| Baseline valuation | Truncated logistic, location `5.2`, scale `0.79`, support `[1, 10]` |
| Dependence | Baseline valuation and time sensitivity are independent |
| Candidate prices | 81 geometrically spaced prices in `[0.01, 5]` |
| Relative errors | `-50%, -30%, -20%, -10%, 0%, +10%, +20%, +30%, +50%` |
| Default test histories | 500 Monte Carlo replications |
| Independent Oracle-SW training histories | 500 Monte Carlo replications |
| Default random seed | `20821104` |

For a buyer with intrinsic valuation `X` and time sensitivity `Theta`, valuation at age `a` is `V(a) = X * exp(-Theta * a)`. Revenue at posted price `p` is `p * Pr(V(a) >= p)`.

All relative-error scenarios **reuse exactly the same test histories**, enabling paired comparisons. Oracle-SW uses an additional, independent sample of training histories to select a fixed window separately for each assumed decay rate. True revenues are computed from the unchanged market distribution, not the seller's estimated decay rate.

## Compared methods

- **ACRP-FV:** mean-decay calibration and sliding-window selection by forward validation.
- **SW-FV:** uncalibrated sliding window selected by forward validation.
- **SW-UCB-FV:** full-feedback adaptation of sliding-window UCB; jointly selects window length and confidence-bonus coefficient by forward validation. It is **not** the original bandit-feedback implementation.
- **Fixed-Share:** candidate prices are experts whose weights are updated with observed historical-bid revenues; learning rate and sharing rate are selected through forward validation.
- **Calibrated Full-history:** calibrates and pools the entire available history.
- **Oracle-SW:** selects the best fixed calibrated window using independent simulation and true expected revenue; a reference benchmark, not a deployable learner.

For sliding-window forward validation, `h = 1, ..., 29`; Oracle-SW searches `h = 1, ..., 39`. The frozen SW-UCB confidence grid contains 11 values (`0`, `0.025`, `0.05`, `0.1`, `0.2`, `0.35`, `0.5`, `0.75`, `1`, `1.5`, `2`) with `delta = 0.05`. Fixed-Share uses 18 learning rates and 21 sharing rates. The complete fixed grids are defined in `src/competitors.py` and verified against `config/frozen_config.json` when an experiment starts.

## Quick start

Use **Python 3.11 or 3.12** (recommended) in a virtual environment:

```bash
python -m pip install -r requirements.txt
```

Run the original 500-replication setup:

```bash
python run_experiment.py --mc 500 --oracle-mc 500 --cv 0.4 --seed 20821104 --data-root runs/validation_cv04_500mc
python check_results.py --data-root runs/validation_cv04_500mc
python make_figures.py --data-root runs/validation_cv04_500mc
```

A faster smoke test, writing to a separate folder:

```bash
python run_experiment.py --mc 2 --oracle-mc 2 --data-root runs/smoke_test
python check_results.py --data-root runs/smoke_test
python make_figures.py --data-root runs/smoke_test
```

To test custom errors, pass the full set including zero, for example:

```bash
python run_experiment.py --errors -0.3 -0.1 0 0.1 0.3 --mc 100 --oracle-mc 100 --data-root runs/custom_errors
```

**Caution:** rerunning a job with the same `--data-root` overwrites its output CSVs. Use a new output folder for additional runs.

## Output files

After running `run_experiment.py`, files are written under the supplied `--data-root`:

```text
runs/<run_name>/
├── experiment_config.json
├── raw/
│   ├── all_method_runs.csv              # MC-level method revenues and relative revenue losses
│   └── selected_parameters.csv          # MC-level selected windows and hyperparameters
└── summary/
    ├── summary_rrl.csv                  # Mean RRL, standard error, normal-approximation 95% CI
    ├── ACRP_paired_vs_correct_mean.csv  # Paired RRL differences relative to zero error
    ├── oracle_window.csv                # Independently trained Oracle-SW windows
    └── window_summary.csv               # Summary of selected window lengths
```

`make_figures.py` reads the CSV files without rerunning the simulations and creates three figures (`Fig1_all_methods`, `Fig1b_without_full_history`, and `Fig2_ACRP_window`) in PNG, PDF, and SVG formats under `figures/`.

Relative revenue loss is reported **in percent**, with smaller values indicating better performance. Confidence intervals use a normal approximation; the paired analysis reuses the same Monte Carlo histories across assumed-decay scenarios.

### Optional Excel export

`export_excel.py` creates an `.xlsx` summary using the optional [`artifact_tool`](https://pypi.org/project/artifact-tool/) dependency (not required for the experiments, tests, figures, or CSV outputs). If installed:

```bash
python export_excel.py --data-root runs/validation_cv04_500mc
```

## Project structure

```text
decay-error/
├── README.md
├── .gitignore
├── requirements.txt
├── config/
│   └── frozen_config.json
├── run_experiment.py
├── check_results.py
├── make_figures.py
├── export_excel.py
├── src/
│   ├── market.py
│   ├── acrp_core.py
│   ├── competitors.py
│   └── fixedshare_reference.py
└── docs/
    └── RESULTS_500MC.md
```

The GitHub upload contains **source code, fixed configuration, and documentation**, but **not** generated Monte Carlo data or original market datasets. Generated outputs are excluded by `.gitignore`.

## Reproducibility and interpretation

This package originally lacked `config/frozen_config.json`, required by `run_experiment.py`. That file has been restored from the literal candidate grids in `src/competitors.py`, without changing the pricing algorithms. The original script also used `±25%` as two default relative-error levels, while its experimental report and intended design specified `±30%`; the defaults were corrected to `±30%` for consistency. All other model and algorithm code is retained.

For the previously reported 500-MC summary (the raw runs were **not** included in the uploaded archive), see [`docs/RESULTS_500MC.md`](docs/RESULTS_500MC.md). Rerun the scripts to independently regenerate and verify the values.

Mean-decay calibration optimality under a squared log-valuation error criterion does not imply that the exact mean decay rate minimizes revenue loss; pricing performance depends on the shape of the demand curve and on window selection.
