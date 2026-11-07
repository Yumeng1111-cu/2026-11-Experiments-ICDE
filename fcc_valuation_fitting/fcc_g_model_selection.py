#!/usr/bin/env python3
"""Fit a reusable base-valuation model to FCC Auction 97 block-G exit bids.

The script reproduces the preprocessing used by the MAPP paper:
1. Read the FCC cumulative XML file from the official ZIP archive.
2. Keep the last observed bid for each (license, bidder) pair.
3. Select block G, take log(bid), and min-max scale it to [1, 10].
4. Compare bounded parametric models by five-fold held-out log likelihood.
5. Save the selected model, diagnostics, figures, and the processed sample.

Usage
-----
Quick reproduction from the included processed data::

    python src/fcc_g_model_selection.py \
        --input-csv data/processed/fcc_auction97_block_g_exit_bids.csv \
        --output-dir outputs

Full reproduction from the included official FCC ZIP::

    python src/fcc_g_model_selection.py \
        --zip data/raw/97_341_all_files.zip \
        --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lxml import etree
from scipy import optimize, stats
from scipy.special import expit, logit, logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold


LOWER = 1.0
UPPER = 10.0
SPAN = UPPER - LOWER
EXPECTED_G_EXIT_BIDS = 4_286


def extract_block_g_exit_bids(zip_path: Path) -> pd.DataFrame:
    """Stream the 1.1 GB cumulative XML and retain block-G exit bids only."""
    last: dict[tuple[str, str], dict[str, object]] = {}

    with zipfile.ZipFile(zip_path) as archive:
        xml_names = [
            name
            for name in archive.namelist()
            if name.endswith("97_341_database_cumulative.xml")
        ]
        if len(xml_names) != 1:
            raise RuntimeError(
                "Expected exactly one 97_341_database_cumulative.xml in the ZIP"
            )

        with archive.open(xml_names[0]) as stream:
            context = etree.iterparse(stream, events=("end",), tag="ALL_BIDS")
            for _, elem in context:
                row = {child.tag: child.text for child in elem}
                if row.get("block") == "G":
                    item = row.get("item_name")
                    bidder = row.get("bidder_name")
                    amount = row.get("bid_amount")
                    round_text = row.get("round")
                    if item and bidder and amount and round_text:
                        auction_round = int(round_text)
                        key = (item, bidder)
                        if key not in last or auction_round > int(last[key]["round"]):
                            last[key] = {
                                "item_name": item,
                                "item_description": row.get("item_description", ""),
                                "market": row.get("market", ""),
                                "block": "G",
                                "bidder_name": bidder,
                                "bidder_frn": row.get("bidder_frn", ""),
                                "round": auction_round,
                                "bid_amount": float(amount),
                            }

                elem.clear()
                parent = elem.getparent()
                if parent is not None:
                    while elem.getprevious() is not None:
                        del parent[0]

    frame = pd.DataFrame(last.values()).sort_values(
        ["item_name", "bidder_name"], ignore_index=True
    )
    if (frame["bid_amount"] <= 0).any():
        raise ValueError("All retained bid amounts must be positive")

    frame["log_bid"] = np.log(frame["bid_amount"].to_numpy())
    lo = float(frame["log_bid"].min())
    hi = float(frame["log_bid"].max())
    frame["base_valuation"] = LOWER + SPAN * (frame["log_bid"] - lo) / (hi - lo)
    return frame


def load_processed_exit_bids(csv_path: Path) -> pd.DataFrame:
    """Load the packaged G-block exit-bid sample for a fast reproduction."""
    frame = pd.read_csv(
        csv_path,
        dtype={
            "item_name": "string",
            "bidder_name": "string",
            "bidder_frn": "string",
        },
    )
    required = {"item_name", "bidder_name", "round", "bid_amount"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Processed CSV is missing columns: {sorted(missing)}")
    if (frame["bid_amount"] <= 0).any():
        raise ValueError("All retained bid amounts must be positive")

    # Recompute transformation columns rather than trusting cached values.
    frame["log_bid"] = np.log(frame["bid_amount"].to_numpy(dtype=float))
    lo = float(frame["log_bid"].min())
    hi = float(frame["log_bid"].max())
    frame["base_valuation"] = LOWER + SPAN * (frame["log_bid"] - lo) / (hi - lo)
    return frame.sort_values(["item_name", "bidder_name"], ignore_index=True)


@dataclass
class FittedModel:
    name: str
    n_params: int
    logpdf: Callable[[np.ndarray], np.ndarray]
    cdf: Callable[[np.ndarray], np.ndarray]
    ppf: Callable[[np.ndarray], np.ndarray]
    sample: Callable[[int, np.random.Generator], np.ndarray]
    parameters: dict[str, object]


def _clip_unit(x: np.ndarray, n_ref: int) -> np.ndarray:
    # Rank-size boundary correction keeps transformed endpoints finite.
    eps = 0.5 / (n_ref + 1.0)
    return np.clip((np.asarray(x) - LOWER) / SPAN, eps, 1.0 - eps)


def fit_scaled_beta(x: np.ndarray) -> FittedModel:
    n = len(x)
    y = _clip_unit(x, n)
    a, b, _, _ = stats.beta.fit(y, floc=0.0, fscale=1.0)

    def logpdf(v: np.ndarray) -> np.ndarray:
        yy = _clip_unit(v, n)
        return stats.beta.logpdf(yy, a, b) - math.log(SPAN)

    def cdf(v: np.ndarray) -> np.ndarray:
        raw = (np.asarray(v) - LOWER) / SPAN
        return np.where(
            raw <= 0, 0.0, np.where(raw >= 1, 1.0, stats.beta.cdf(raw, a, b))
        )

    def ppf(q: np.ndarray) -> np.ndarray:
        return LOWER + SPAN * stats.beta.ppf(q, a, b)

    def sample(size: int, rng: np.random.Generator) -> np.ndarray:
        return LOWER + SPAN * rng.beta(a, b, size=size)

    return FittedModel(
        "Scaled beta",
        2,
        logpdf,
        cdf,
        ppf,
        sample,
        {"alpha": float(a), "beta": float(b), "lower": LOWER, "upper": UPPER},
    )


def fit_truncated_normal(x: np.ndarray) -> FittedModel:
    x = np.asarray(x)

    def nll(theta: np.ndarray) -> float:
        mu, log_sigma = theta
        sigma = math.exp(log_sigma)
        a, b = (LOWER - mu) / sigma, (UPPER - mu) / sigma
        return float(-np.sum(stats.truncnorm.logpdf(x, a, b, loc=mu, scale=sigma)))

    result = optimize.minimize(
        nll,
        x0=np.array([float(np.mean(x)), math.log(float(np.std(x, ddof=1)))]),
        method="Nelder-Mead",
        options={"maxiter": 20_000, "xatol": 1e-10, "fatol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Truncated-normal fit failed: {result.message}")
    mu, log_sigma = result.x
    sigma = math.exp(log_sigma)
    a, b = (LOWER - mu) / sigma, (UPPER - mu) / sigma

    def logpdf(v: np.ndarray) -> np.ndarray:
        return stats.truncnorm.logpdf(v, a, b, loc=mu, scale=sigma)

    def cdf(v: np.ndarray) -> np.ndarray:
        return stats.truncnorm.cdf(v, a, b, loc=mu, scale=sigma)

    def ppf(q: np.ndarray) -> np.ndarray:
        return stats.truncnorm.ppf(q, a, b, loc=mu, scale=sigma)

    def sample(size: int, rng: np.random.Generator) -> np.ndarray:
        return stats.truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size, random_state=rng)

    return FittedModel(
        "Truncated normal",
        2,
        logpdf,
        cdf,
        ppf,
        sample,
        {"mu": float(mu), "sigma": float(sigma), "lower": LOWER, "upper": UPPER},
    )


def fit_truncated_logistic(x: np.ndarray) -> FittedModel:
    x = np.asarray(x)

    def nll(theta: np.ndarray) -> float:
        loc, log_scale = theta
        scale = math.exp(log_scale)
        z = stats.logistic.cdf(UPPER, loc=loc, scale=scale) - stats.logistic.cdf(
            LOWER, loc=loc, scale=scale
        )
        return float(-np.sum(stats.logistic.logpdf(x, loc=loc, scale=scale) - math.log(z)))

    start_scale = max(float(np.std(x, ddof=1)) * math.sqrt(3) / math.pi, 1e-3)
    result = optimize.minimize(
        nll,
        x0=np.array([float(np.median(x)), math.log(start_scale)]),
        method="Nelder-Mead",
        options={"maxiter": 20_000, "xatol": 1e-10, "fatol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Truncated-logistic fit failed: {result.message}")
    loc, log_scale = result.x
    scale = math.exp(log_scale)
    f_lo = stats.logistic.cdf(LOWER, loc=loc, scale=scale)
    f_hi = stats.logistic.cdf(UPPER, loc=loc, scale=scale)
    z_norm = f_hi - f_lo

    def logpdf(v: np.ndarray) -> np.ndarray:
        vv = np.asarray(v)
        out = stats.logistic.logpdf(vv, loc=loc, scale=scale) - math.log(z_norm)
        return np.where((vv < LOWER) | (vv > UPPER), -np.inf, out)

    def cdf(v: np.ndarray) -> np.ndarray:
        vv = np.asarray(v)
        out = (stats.logistic.cdf(vv, loc=loc, scale=scale) - f_lo) / z_norm
        return np.clip(out, 0.0, 1.0)

    def ppf(q: np.ndarray) -> np.ndarray:
        return stats.logistic.ppf(f_lo + np.asarray(q) * z_norm, loc=loc, scale=scale)

    def sample(size: int, rng: np.random.Generator) -> np.ndarray:
        return ppf(rng.uniform(size=size))

    return FittedModel(
        "Truncated logistic",
        2,
        logpdf,
        cdf,
        ppf,
        sample,
        {"loc": float(loc), "scale": float(scale), "lower": LOWER, "upper": UPPER},
    )


def fit_logit_normal_mixture(
    x: np.ndarray, components: int, seed: int = 2026
) -> FittedModel:
    x = np.asarray(x)
    n = len(x)
    u = _clip_unit(x, n)
    z = logit(u).reshape(-1, 1)
    gm = GaussianMixture(
        n_components=components,
        covariance_type="full",
        n_init=20,
        max_iter=2_000,
        reg_covar=1e-6,
        random_state=seed,
    ).fit(z)
    weights = gm.weights_.copy()
    means = gm.means_.ravel().copy()
    sigmas = np.sqrt(gm.covariances_.reshape(components)).copy()
    order = np.argsort(means)
    weights, means, sigmas = weights[order], means[order], sigmas[order]

    def _log_mixture_density(z_values: np.ndarray) -> np.ndarray:
        terms = []
        for weight, mean, sigma in zip(weights, means, sigmas):
            terms.append(
                math.log(weight) + stats.norm.logpdf(z_values, loc=mean, scale=sigma)
            )
        return logsumexp(np.vstack(terms), axis=0)

    def logpdf(v: np.ndarray) -> np.ndarray:
        vv = np.asarray(v)
        uu = _clip_unit(vv, n)
        zz = logit(uu)
        log_jacobian = math.log(SPAN) + np.log(uu) + np.log1p(-uu)
        out = _log_mixture_density(zz) - log_jacobian
        return np.where((vv < LOWER) | (vv > UPPER), -np.inf, out)

    def cdf(v: np.ndarray) -> np.ndarray:
        vv = np.asarray(v)
        raw = (vv - LOWER) / SPAN
        uu = np.clip(raw, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
        zz = logit(uu)
        out = np.zeros_like(zz, dtype=float)
        for weight, mean, sigma in zip(weights, means, sigmas):
            out += weight * stats.norm.cdf(zz, loc=mean, scale=sigma)
        return np.where(raw <= 0, 0.0, np.where(raw >= 1, 1.0, out))

    def ppf(q: np.ndarray) -> np.ndarray:
        qq = np.asarray(q, dtype=float)
        flat = qq.ravel()
        vals = np.array(
            [
                optimize.brentq(lambda value: float(cdf(np.array([value]))[0] - qi), LOWER, UPPER)
                for qi in flat
            ]
        )
        return vals.reshape(qq.shape)

    def sample(size: int, rng: np.random.Generator) -> np.ndarray:
        labels = rng.choice(components, size=size, p=weights)
        draws = rng.normal(means[labels], sigmas[labels])
        return LOWER + SPAN * expit(draws)

    return FittedModel(
        f"Logit-normal mixture K={components}",
        3 * components - 1,
        logpdf,
        cdf,
        ppf,
        sample,
        {
            "components": components,
            "weights": weights.tolist(),
            "logit_means": means.tolist(),
            "logit_sigmas": sigmas.tolist(),
            "lower": LOWER,
            "upper": UPPER,
            "generation": "Choose k~Categorical(weights), Z~Normal(mu_k,sigma_k), X=1+9*sigmoid(Z)",
        },
    )


def fit_candidate(name: str, x: np.ndarray, seed: int) -> FittedModel:
    if name == "Scaled beta":
        return fit_scaled_beta(x)
    if name == "Truncated normal":
        return fit_truncated_normal(x)
    if name == "Truncated logistic":
        return fit_truncated_logistic(x)
    if name.startswith("Logit-normal mixture K="):
        return fit_logit_normal_mixture(x, int(name.rsplit("=", 1)[1]), seed=seed)
    raise KeyError(name)


def compare_models(x: np.ndarray, seed: int) -> tuple[pd.DataFrame, dict[str, FittedModel]]:
    names = ["Scaled beta", "Truncated normal", "Truncated logistic"] + [
        f"Logit-normal mixture K={k}" for k in range(1, 7)
    ]
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    fold_scores: dict[str, list[float]] = {name: [] for name in names}

    for fold_id, (train_idx, test_idx) in enumerate(folds.split(x)):
        train, test = x[train_idx], x[test_idx]
        for name in names:
            model = fit_candidate(name, train, seed + fold_id)
            fold_scores[name].append(float(-np.mean(model.logpdf(test))))

    final_models = {name: fit_candidate(name, x, seed) for name in names}
    rows = []
    for name in names:
        model = final_models[name]
        log_likelihood = float(np.sum(model.logpdf(x)))
        k = model.n_params
        ks = stats.kstest(x, lambda values: model.cdf(np.asarray(values))).statistic
        scores = np.asarray(fold_scores[name])
        rows.append(
            {
                "model": name,
                "parameters": k,
                "cv_nll_mean": float(scores.mean()),
                "cv_nll_se": float(scores.std(ddof=1) / math.sqrt(len(scores))),
                "log_likelihood": log_likelihood,
                "aic": float(2 * k - 2 * log_likelihood),
                "bic": float(k * math.log(len(x)) - 2 * log_likelihood),
                "ks_statistic": float(ks),
            }
        )
    table = pd.DataFrame(rows).sort_values(["cv_nll_mean", "bic"], ignore_index=True)
    return table, final_models


def make_fit_figure(x: np.ndarray, selected: FittedModel, output_path: Path) -> None:
    grid = np.linspace(LOWER + 1e-4, UPPER - 1e-4, 1_200)
    kde = stats.gaussian_kde(x, bw_method="scott")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        }
    )
    fig, ax = plt.subplots(figsize=(9.2, 5.7), dpi=220)
    ax.hist(
        x,
        bins="fd",
        density=True,
        color="#DCE6F1",
        edgecolor="white",
        linewidth=0.6,
        label="Block G empirical distribution",
    )
    ax.plot(grid, kde(grid), color="#7A869A", linewidth=2.0, label="KDE reference")
    ax.plot(
        grid,
        np.exp(selected.logpdf(grid)),
        color="#C63C3C",
        linewidth=2.8,
        label=selected.name,
    )
    ax.set(
        xlabel="Normalized log exit bid X",
        ylabel="Density",
        title="FCC Auction 97 block G base valuation fit",
        xlim=(LOWER, UPPER),
    )
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def make_diagnostic_figure(
    x: np.ndarray, comparison: pd.DataFrame, selected: FittedModel, output_path: Path
) -> None:
    ordered = np.sort(x)
    empirical = np.arange(1, len(x) + 1) / len(x)
    probs = (np.arange(1, len(x) + 1) - 0.5) / len(x)
    fitted_quantiles = selected.ppf(probs)
    top = comparison.sort_values("cv_nll_mean").head(9).iloc[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), dpi=220)

    axes[0].step(ordered, empirical, where="post", color="#4B657E", label="Empirical")
    grid = np.linspace(LOWER, UPPER, 1_000)
    axes[0].plot(grid, selected.cdf(grid), color="#C63C3C", linewidth=2.2, label="Fitted")
    axes[0].set(title="CDF comparison", xlabel="X", ylabel="Cumulative probability")
    axes[0].legend(frameon=False)

    axes[1].scatter(fitted_quantiles, ordered, s=7, alpha=0.35, color="#4B657E")
    axes[1].plot([LOWER, UPPER], [LOWER, UPPER], color="#C63C3C", linewidth=1.8)
    axes[1].set(title="Quantile diagnostic", xlabel="Fitted quantiles", ylabel="Observed quantiles")

    y_pos = np.arange(len(top))
    axes[2].errorbar(
        top["cv_nll_mean"],
        y_pos,
        xerr=top["cv_nll_se"],
        fmt="o",
        markersize=5,
        color="#6F8FAF",
        ecolor="#263746",
        capsize=2,
        linewidth=1,
    )
    axes[2].set_yticks(y_pos, labels=top["model"])
    x_min = float((top["cv_nll_mean"] - top["cv_nll_se"]).min())
    x_max = float((top["cv_nll_mean"] + top["cv_nll_se"]).max())
    pad = 0.08 * (x_max - x_min)
    axes[2].set_xlim(x_min - pad, x_max + pad)
    axes[2].set(title="Five-fold predictive fit", xlabel="Mean held-out NLL (lower is better)")

    for ax in axes:
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Selected model: {selected.name}", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def bootstrap_parameter_ci(
    x: np.ndarray, selected: FittedModel, seed: int, repetitions: int = 500
) -> dict[str, list[float]]:
    """Nonparametric percentile intervals for the selected two-parameter model."""
    if selected.name != "Truncated logistic":
        return {}
    rng = np.random.default_rng(seed)
    estimates = np.empty((repetitions, 2), dtype=float)
    for idx in range(repetitions):
        boot = rng.choice(x, size=len(x), replace=True)
        fit = fit_truncated_logistic(boot)
        estimates[idx] = [fit.parameters["loc"], fit.parameters["scale"]]
    bounds = np.quantile(estimates, [0.025, 0.975], axis=0)
    return {
        "loc_95_percentile_ci": bounds[:, 0].tolist(),
        "scale_95_percentile_ci": bounds[:, 1].tolist(),
        "bootstrap_repetitions": repetitions,
    }


def write_selected_model(
    path: Path,
    selected: FittedModel,
    comparison: pd.DataFrame,
    frame: pd.DataFrame,
    parameter_uncertainty: dict[str, object] | None = None,
) -> None:
    summary = {
        "data": {
            "source": "FCC Auction 97 cumulative bid file",
            "block": "G",
            "definition": "last observed bid for each (item_name, bidder_name) pair",
            "n": int(len(frame)),
            "raw_bid_min": float(frame["bid_amount"].min()),
            "raw_bid_max": float(frame["bid_amount"].max()),
            "log_bid_min": float(frame["log_bid"].min()),
            "log_bid_max": float(frame["log_bid"].max()),
            "normalized_mean": float(frame["base_valuation"].mean()),
            "normalized_std": float(frame["base_valuation"].std(ddof=1)),
        },
        "selection_rule": "lowest five-fold held-out negative log likelihood; BIC used as secondary diagnostic",
        "selected_model": selected.name,
        "parameters": selected.parameters,
        "parameter_uncertainty": parameter_uncertainty or {},
        "diagnostics": comparison.iloc[0].to_dict(),
        "normalization": "X = 1 + 9*(log(bid)-min_log_bid)/(max_log_bid-min_log_bid), computed within block G",
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--zip", type=Path, help="FCC 97_341_all_files.zip")
    source.add_argument(
        "--input-csv",
        type=Path,
        help="Previously extracted G-block exit-bid CSV for a fast reproduction",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = (
        extract_block_g_exit_bids(args.zip)
        if args.zip is not None
        else load_processed_exit_bids(args.input_csv)
    )
    if len(frame) != EXPECTED_G_EXIT_BIDS:
        print(
            f"WARNING: extracted {len(frame):,} G exit bids; "
            f"MAPP reports {EXPECTED_G_EXIT_BIDS:,}."
        )
    else:
        print(f"Validated {len(frame):,} block-G exit bids.")

    frame.to_csv(args.output_dir / "fcc_auction97_block_g_exit_bids.csv", index=False)
    x = frame["base_valuation"].to_numpy(dtype=float)
    comparison, models = compare_models(x, args.seed)
    selected_name = str(comparison.iloc[0]["model"])
    selected = models[selected_name]
    parameter_uncertainty = bootstrap_parameter_ci(x, selected, args.seed + 10_000)

    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    write_selected_model(
        args.output_dir / "selected_model.json",
        selected,
        comparison,
        frame,
        parameter_uncertainty,
    )
    make_fit_figure(x, selected, args.output_dir / "g_valuation_fit.png")
    make_diagnostic_figure(
        x, comparison, selected, args.output_dir / "g_model_diagnostics.png"
    )

    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nSelected:", selected.name)
    print(json.dumps(selected.parameters, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
