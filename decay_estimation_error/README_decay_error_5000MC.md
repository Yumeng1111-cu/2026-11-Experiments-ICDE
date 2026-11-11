# ACRP-FV: Robustness to Mean Time-Sensitivity Estimation Errors

Reproducible Python implementation of a Monte Carlo study investigating how errors in the **estimated mean time sensitivity** affect pricing for aging data. The main experiment uses **5,000 Monte Carlo replications** and compares **ACRP-FV** with five reference methods under a fixed true market distribution.

## Research question

Buyers have heterogeneous time sensitivities. The seller observes historical bids but calibrates them using an estimated average decay rate, which may differ from the true mean. Does forward-validation-based history selection reduce the revenue impact of this misspecification?

The true mean decay rate is fixed at `0.042`; the seller's assumed rate is

```python
theta_assumed = 0.042 * (1 + relative_error)
```

The main experiment evaluates **five relative estimation errors**:

| Relative error | Assumed mean decay rate |
| ---: | ---: |
| `-50%` | `0.0210` |
| `-25%` | `0.0315` |
| `0%` | `0.0420` |
| `+25%` | `0.0525` |
| `+50%` | `0.0630` |

Negative errors mean **underestimation**, positive errors mean **overestimation**, and zero denotes the correctly specified mean decay rate. Only the **assumed** rate changes across scenarios; the true market distribution and evaluation target remain unchanged.

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
| Relative estimation errors | `-50%, -25%, 0%, +25%, +50%` |
| Main experiment test histories | **5,000 Monte Carlo replications** |
| Independent Oracle-SW training histories | **5,000 Monte Carlo replications** |
| Random seed for the main run | `20821104` |

For a buyer with intrinsic valuation `X` and time sensitivity `Theta`, valuation at age `a` is `V(a) = X * exp(-Theta * a)`. Revenue at posted price `p` is `p * Pr(V(a) >= p)`.

All relative-error scenarios **reuse exactly the same test histories**, enabling paired comparisons. Oracle-SW uses an additional, independent sample of training histories to select a fixed window separately for each assumed decay rate. True revenues are computed from the unchanged market distribution, not the seller's estimated decay rate.

## Compared methods

- **ACRP-FV:** Mean-decay calibration and sliding-window selection by forward validation.
- **SW-FV:** Uncalibrated sliding window selected by forward validation.
- **SW-UCB-FV:** Full-feedback adaptation of sliding-window UCB; jointly selects window length and confidence-bonus coefficient by forward validation. It is **not** the original bandit-feedback implementation.
- **Fixed-Share:** Candidate prices are experts whose weights are updated with observed historical-bid revenues; learning rate and sharing rate are selected through forward validation.
- **Calibrated Full-history:** Calibrates and pools the entire available history.
- **Oracle-SW:** Selects the best fixed calibrated window using independent simulation and true expected revenue; a reference benchmark, not a deployable learner.

For sliding-window forward validation, `h = 1, ..., 29`; Oracle-SW searches `h = 1, ..., 39`. The frozen SW-UCB confidence grid contains 11 values (`0`, `0.025`, `0.05`, `0.1`, `0.2`, `0.35`, `0.5`, `0.75`, `1`, `1.5`, `2`) with `delta = 0.05`. Fixed-Share uses 18 learning rates and 21 sharing rates. The complete fixed grids are defined in `src/competitors.py` and verified against `config/frozen_config.json` when an experiment starts.

## Quick start

Use **Python 3.11 or 3.12** (recommended) in a virtual environment:

```bash
python -m pip install -r requirements.txt
```

Run the **main 5,000-replication experiment** with the five specified error levels:

```bash
python run_experiment.py --mc 5000 --oracle-mc 5000 --cv 0.4 --seed 20821104 --errors -0.5 -0.25 0 0.25 0.5 --data-root runs/formal_cv04_5000mc
python check_results.py --data-root runs/formal_cv04_5000mc
python make_figures.py --data-root runs/formal_cv04_5000mc
```

A faster smoke test, writing to a separate folder:

```bash
python run_experiment.py --mc 2 --oracle-mc 2 --cv 0.4 --seed 20821104 --errors -0.5 -0.25 0 0.25 0.5 --data-root runs/smoke_test
python check_results.py --data-root runs/smoke_test
python make_figures.py --data-root runs/smoke_test
```

To test another set of errors, pass the full set including zero, for example:

```bash
python run_experiment.py --errors -0.3 -0.1 0 0.1 0.3 --mc 100 --oracle-mc 100 --data-root runs/custom_errors
```



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

Relative revenue loss (RRL) is reported **in percent**, with smaller values indicating better performance. Confidence intervals use a normal approximation; paired analysis reuses the same Monte Carlo histories across assumed-decay scenarios.

### Optional Excel export

`export_excel.py` creates an `.xlsx` summary using the optional [`artifact_tool`](https://pypi.org/project/artifact-tool/) dependency (not required for the experiments, tests, figures, or CSV outputs). If installed:

```bash
python export_excel.py --data-root runs/formal_cv04_5000mc
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
