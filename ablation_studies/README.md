# Ablation Studies (Section 7.2)

## Experiments

| Study | Independent variables | Constant settings |
| --- | --- | --- |
| A: mean decay and heterogeneity | Mean decay `mu` = 0.01, 0.02, 0.03; CV = 0.2, 0.3, 0.4, 0.5 | `n=80`, `t=80` |
| B: sample size and heterogeneity | `n` = 20, 40, 60, 80, 100, 120; CV = 0.3, 0.4, 0.5 | `mu=0.02`, `t=80` |
| C: target period and heterogeneity | `t` = 40, 50, ..., 100; CV = 0.3, 0.4, 0.5 | `mu=0.02`, `n=80` |

The code implements six methods, presented in the figures as **U-FH**, **C-FH**, **U-SW-FV**, **ACRP-FV**, **Oracle-SW**, and **Oracle-EW**. The mathematical simulation model in `settings.py` uses lognormal baseline valuations and Gamma-distributed time sensitivity (independent): `log(X) ~ N(0, 0.5^2)` and `Theta ~ Gamma(mean=mu, CV=cv)`. The forward-validation length is `M=20`.

### Stored figures

| Experiment A | Experiment B | Experiment C |
| --- | --- | --- |
| ![Ablation A](outputs/figures/Figure_7_2_A.png) | ![Ablation B](outputs/figures/Figure_7_2_B.png) | ![Ablation C](outputs/figures/Figure_7_2_C.png) |

PDF and SVG counterparts are available in [`outputs/figures/`](outputs/figures/).

## Quick start

From this **directory** (`ablation_studies/`), using Python 3.10+ in a fresh environment:

```bash
python -m pip install -r requirements.txt
python verify_saved_data.py
python plot_all.py
```

`plot_all.py` **only reads** `outputs/data/formal/summary_{A,B,C}.csv`; it does not rerun the Monte Carlo simulation. For figure style (fonts, dimensions, colors), edit `plot_settings.py` and rerun `plot_all.py`. You can also use `plot_A.py`, `plot_B.py`, or `plot_C.py` individually.

For computational correctness checks:

```bash
python test_pricing.py
```

To reproduce the calculations rather than reuse saved data:

1. Open `settings.py` and set `FORCE_RECOMPUTE = True`.
2. Use `QUICK_MODE = True` and `QUICK_REPS = 5` for a preliminary test. For formal simulations, set `QUICK_MODE = False`, `TEST_REPS = 5000`, and `PILOT_REPS = 5000`.
3. Execute `python run_all.py` or individual `python experiment_A.py` / `experiment_B.py` / `experiment_C.py`.
4. Switch `plot_settings.py`'s `PLOT_DATA_MODE` to `formal` or `quick` to redraw saved summaries. The supplied quick cache contains experiments A and B, **not C**; run C in quick mode before attempting to plot all quick panels.

**Important:** full formal simulation is computationally intensive. By default `FORCE_RECOMPUTE = False` so matching stored per-scenario CSVs are reused. The simulation command also has an older, separate figure routine (`core.plot_three_panels`). Use `plot_all.py` for the consistent, compact publication-style plots in this repository.

## Data and output organization

```text
ablation_studies/
├── README.md / README.zh-CN.md
├── requirements.txt / .gitignore
├── core.py                 # Monte Carlo engine and numerical model
├── settings.py             # Simulation setup
├── experiment_A.py, experiment_B.py, experiment_C.py
├── run_all.py              # Generate or reuse Monte Carlo results
├── plot_settings.py        # Publication figure styling
├── plot_saved.py           # Plot from saved summary CSV (no simulation)
├── plot_A.py, plot_B.py, plot_C.py, plot_all.py
├── test_pricing.py         # Computational checks
├── verify_saved_data.py   # Validate bundled results without third-party packages
└── outputs/
    ├── data/formal/
    │   ├── config_{A,B,C}.json
    │   ├── summary_{A,B,C}.csv
    │   └── mu*_test5000_pilot5000_*.csv
    ├── data/quick/          # Supplied quick-mode records (A and B)
    └── figures/             # A/B/C in PNG, PDF, and SVG
```

`formal/` contains **51 experiment configurations**, with **45 distinct per-scenario CSV files**, because some scenarios are shared across experiments. The quick folder contains **27 per-scenario CSVs**, summaries A and B, and config A; the quick results are not publication-quality estimates. All of these files are retained as provided.
