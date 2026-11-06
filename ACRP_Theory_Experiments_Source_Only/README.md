# Numerical Verification of Memory Selection for Aging-Data Pricing

Source-only Python repository for **three numerical verification experiments** associated with average-decay calibration and recency-weighted pricing. The project is based on the author's `ACRP_7_1_PyCharm_Python.zip` code. **No precomputed experiment data, generated plots, cached simulation histories, or legacy results are committed.** Run the code to produce your own outputs.

[中文使用说明](README.zh-CN.md)

## Experiments

1. **Historical demand mismatch:** variation with historical age gap and heterogeneity.
2. **Memory / revenue trade-off:** sliding-window length and exponential forgetting versus relative revenue loss.
3. **Revenue-optimal memory:** comparative statics in heterogeneity and number of buyers.

## Requirements

Python 3.11–3.13 with compatible packages (tested on Python 3.13). Install:

```bash
python -m pip install -r requirements.txt
```

To use a virtual environment, create and activate one before installing dependencies. In PyCharm, select the environment as the project interpreter. Times New Roman is used when available, with a serif fallback elsewhere; no font files are distributed.

## Quick reproduction from scratch

```bash
python reproduce.py --quick --reps 2
```

This **generates new CSVs**, then **plots all three figures**. Quick mode is a smoke test, not paper-quality statistical evidence; use a larger repetition count or the formal run for scientific results. The first run may need to compile Numba kernels.

Formal experiments use `TEST_REPS = 5000` and `PILOT_REPS = 5000` by default:

```bash
python reproduce.py
```

**Formal runs can be computationally expensive.** The default settings are in `settings.py`; plotting controls (including experiment 1's independent label fonts, colors, line widths and figure dimensions) are in `plot_settings.py`.

## Separate computation from plotting

To generate only CSVs, edit `QUICK_MODE` in `settings.py` and run:

```bash
python generate_all.py  # or generate_1.py / generate_2.py / generate_3.py
```

To redraw figures without Monte Carlo, set `PLOT_DATA_MODE` in `plot_settings.py` to the corresponding `"formal"` or `"quick"` folder and run:

```bash
python plot_all.py  # or plot_1.py / plot_2.py / plot_3.py
```

These plot scripts read CSVs only; they never recompute simulation data. **They require the CSVs to have been generated first.** Output filenames are controlled by `plot_settings.py`.

## Generated outputs (not included in Git)

```text
outputs/
├── data/
│   ├── quick/                       # quick-run CSVs (if selected)
│   ├── formal/                      # formal CSVs (if selected)
│   └── sigma*_*.csv                 # intermediate simulation caches
└── figures/
    ├── Experiment1_Mismatch.png / .pdf / .svg
    ├── Experiment2_MemoryTradeoff.png / .pdf / .svg
    └── Experiment3_ComparativeStatics_4Panels.png / .pdf / .svg
```

`outputs/` is created automatically when the programs run. The entire directory is ignored by Git, preventing accidental submission of stale or mixed results. Generated data are **not necessarily byte-identical** to any earlier local results if settings or numerical-library versions differ.

## Numerical model and settings

The supplied numerical verification code uses independent Gamma time sensitivities with fixed mean (`MU = 0.02`) and log-normal baseline valuations:

$$\log X \sim \mathcal{N}(0,0.5^2),\qquad V_{i,t}=X_{i,t}e^{-\Theta_{i,t}(t-1)}.$$

Default target period `T = 80`, buyers per period `N = 80`, heterogeneity levels `SIGMAS = (0, 0.005, 0.010, 0.015, 0.020)`. It does **not** load FCC or EB-NeRD data and is separate from experiments using truncated-logistic baseline valuations.

## Validation

```bash
python test_pricing.py
python scripts/verify_data.py --mode quick  # after a quick run
python scripts/verify_data.py --mode formal # after a formal run
```

GitHub Actions runs the pricing test, regenerates a tiny quick dataset, validates the generated CSV schema, and redraws figures. All outputs remain ignored by Git.

## Repository layout

```text
core.py                    # pricing, sampling, theoretical quantities and Monte Carlo
settings.py                # numerical parameters
experiment1_mismatch.py   # experiment 1 CSV generator
experiment2_tradeoff.py   # experiment 2 CSV generator
experiment3_comparative.py# experiment 3 CSV generators
reproduce.py               # generates all data and figures (one command)
generate_*.py              # data-only entry points
plot_saved.py              # CSV-only Matplotlib plotter
plot_settings.py           # figure appearance / palettes
plot_*.py                  # plot-only entry points
test_pricing.py             # numerical optimizer test
scripts/verify_data.py     # check CSV outputs after generation
.github/workflows/check.yml# lightweight reproducibility checks
```

**License and citation:** No license or author identity was inferred from the source. Add an appropriate `LICENSE` and final paper citation before public release when decided by the authors.
