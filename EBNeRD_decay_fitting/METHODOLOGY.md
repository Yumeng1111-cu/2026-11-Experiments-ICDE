# Methodology and reproducibility notes

This file documents the **method implemented in the original uploaded code**;
it does not claim an independent audit of the raw dataset.

## Input and observation definition

The project reads the **EB-NeRD Small** archive containing:

- `articles.parquet`: `article_id`, `published_time`
- `train/behaviors.parquet`: `impression_time`, `article_ids_inview`, `article_ids_clicked`
- `validation/behaviors.parquet`: same fields

`read_impression_range` finds the minimum and maximum impression timestamps
across the training and validation splits. The eligible cohort contains
articles published within the log range with at least a 48-hour opportunity
for observation: `earliest <= published_time <= latest - 48h`.

For each behavior log, every candidate appearing in its `article_ids_inview`
list contributes one exposure, provided the article is eligible and its age
falls in `[0, 48)`. The same candidate contributes a click if that article ID
also occurs in the same log's `article_ids_clicked` list.

Article age is computed as impression timestamp minus publication timestamp,
then floored to an integer hour. The pooled rate is

`CTR_h = sum_articles(clicks[article, h]) / sum_articles(exposures[article, h])`.

**Interpretation:** the denominator counts candidate news exposures, not all
pageviews, not all behavior rows, and not distinct users. This computation is
not a causal estimate of aging effects; position, recommendation policy,
article mix and audience effects can also contribute.

## Statistical models

For bins `h` in a candidate fitting interval `[start, 48)`, the model is

`p_h = c * exp(-lambda * (h + 0.5 - start))`,

where `p_h` is the modeled pooled click probability. `c` and `lambda` are
estimated by maximizing the binomial likelihood, equivalently minimizing

`-sum_h [ clicks_h log(p_h) + (exposures_h - clicks_h) log(1-p_h) ]`.

Optimization: SciPy `minimize(method='L-BFGS-B')`, with
`c in (0, 0.999]` and `lambda in [0,1]`.

The original project estimates intervals starting at `0, 3, 6, 8, 10, 12` h.
Only hours with at least 100 candidate exposures are included in a fit, by
default. Weighted in-sample goodness of fit is calculated as

`R²_w = 1 - sum_h n_h (CTR_h - p_h)^2 / sum_h n_h (CTR_h - CTR_wmean)^2`.

The 8–48 h fit is the focus of the later-decay visualization. The starting
hour was selected after inspecting the observed curve; it should **not** be
presented as a prespecified or out-of-sample optimal selection.

## Article bootstrap

The bootstrap resamples *article-level 48-hour count trajectories* via a
multinomial draw of article weights and recomputes each fit. Results include
2.5th, 50th and 97.5th percentiles of `lambda` over successful optimizations.

The script defaults to 500 replications (`--bootstrap 500`) with a fixed
pseudorandom seed. Due to optimizer, floating-point and dependency versions,
small differences may occur across environments.

Article trajectories are retained **in memory only** during execution.
The GitHub-ready pipeline does not serialize per-article identifiers or counts.

## Scientific limitations

- The 0–48 h trajectory is not necessarily monotonically decaying; the early
  publication period can increase in observed click-through rate.
- The measured target is **click-through rate**, not economic valuation,
  willingness to pay, or a direct estimate of buyer-level time sensitivity.
- The data reflect a recommendation environment and its candidate exposure
  mechanism, not randomized sampling across all readers or articles.
- In-sample weighted R² should not be interpreted as predictive validation.
- Published research outputs and any cross-domain uses remain subject to
  the dataset's license and any additional written authorization.
