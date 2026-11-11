# FCC Auction 97 Block-G Base-Valuation Model Selection

This repository extracts and models base valuations using public bidding
records from FCC Auction 97 (AWS-3). It compares nine bounded parametric
models and selects the final distribution by five-fold held-out negative log
likelihood.

This workflow estimates only the base-valuation distribution. It does not
model data age or buyers' time sensitivities.

## Main result

The processed sample contains 4,286 block-G exit bids. The selected model is:

```text
X ~ TruncatedLogistic(mu=5.195693, scale=0.792559; lower=1, upper=10)
```

| Model | Parameters | Five-fold NLL | BIC | KS |
|---|---:|---:|---:|---:|
| Truncated logistic | 2 | **1.7292** | **14835.61** | 0.0147 |
| Logit-normal mixture K=6 | 17 | 1.7303 | 14901.10 | 0.0088 |
| Logit-normal mixture K=5 | 14 | 1.7334 | 14920.84 | 0.0125 |
| Logit-normal mixture K=4 | 11 | 1.7343 | 14889.00 | 0.0099 |
| Truncated normal | 2 | 1.7374 | 14905.89 | 0.0354 |
| Logit-normal mixture K=2 | 5 | 1.7405 | 14954.72 | 0.0150 |
| Logit-normal mixture K=3 | 8 | 1.7423 | 14953.68 | 0.0117 |
| Scaled beta | 2 | 1.7749 | 15227.18 | 0.0558 |
| Logit-normal mixture K=1 | 2 | 1.8167 | 15584.69 | 0.0666 |

The 500-repetition percentile-bootstrap intervals are:

```text
mu:    [5.151997, 5.235126]
scale: [0.770015, 0.813495]
```

## Figures

![Empirical distribution and selected model](outputs/g_valuation_fit.png)

![Model diagnostics](outputs/g_model_diagnostics.png)

## Repository structure

```text
fcc-auction97-g-valuation-fit/
├── README.md
├── requirements.txt
├── run_all.py
├── download_data.py
├── src/
│   └── fcc_g_model_selection.py
├── data/
│   ├── README.md
│   └── processed/
│       └── fcc_auction97_block_g_exit_bids.csv
└── outputs/
    ├── README.md
    ├── model_comparison.csv
    ├── selected_model.json
    ├── g_valuation_fit.png
    └── g_model_diagnostics.png
```

The original FCC ZIP and XML are intentionally excluded from the repository.
The included download script retrieves them only when a full reproduction is
requested.

## Installation

Python 3.10-3.12 is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproduce the results

Fast reproduction using the included processed sample:

```bash
python run_all.py
```

Full reproduction from the FCC cumulative XML:

```bash
python run_all.py --from-raw --download
```

The full route downloads the raw archive into `data/raw/`, extracts the last
observed bid for each block-G `(license, bidder)` pair, and reruns the same
model comparison. Raw files remain ignored by Git.

## Data processing

For each bidder-license pair, let `B` be the final observed bid. The normalized
base valuation is

```text
X = 1 + 9 * (log(B) - min(log(B))) / (max(log(B)) - min(log(B))).
```

The minimum and maximum are calculated within block G. Candidate models are:

- scaled beta;
- truncated normal;
- truncated logistic; and
- logit-normal mixtures with 1-6 components.

The primary criterion is the mean held-out negative log likelihood under
five-fold cross-validation. AIC, BIC, and the Kolmogorov-Smirnov distance are
reported as additional diagnostics. Final parameters use the full sample, and
uncertainty is evaluated with 500 nonparametric bootstrap repetitions. The
default random seed is `2026`.

## Data source and attribution

The source is the public FCC Auction 97 bidding record:

- Federal Communications Commission, **Auction 97: Advanced Wireless Services
  (AWS-3)**: <https://www.fcc.gov/auction/97>
- Original archive: `97_341_all_files.zip`
- Cumulative XML: `97_341_database_cumulative.xml`

An exit bid is used only as a proxy for valuation; it is not a direct
observation of a bidder's private valuation. 

The data treatment follows the real-data preprocessing used in:

- Gao et al., *Learn Then Decide: A Learning Approach for Designing Data
  Marketplaces*, arXiv:2503.10773.
