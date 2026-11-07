# Data access, redistribution, and licensing notice

**The repository includes research code only. It does not distribute EB-NeRD
Small or any real-data-derived results.**

## Official sources

- Dataset: https://recsys.eb.dk/
- Terms: https://recsys.eb.dk/assets/pdf/general_license_terms.pdf

Users must review and accept the original terms, then obtain the dataset from
servers controlled by the provider. Downloading this GitHub code does **not**
grant access to the dataset or a data license.

## Important distinctions in the license

- Section 1.1 reserves ownership to the data provider and limits usage to
  research, not commercial exploitation.
- Sections 4.1–4.2 discuss the licensed download channel, noncommercial
  research, and publication of results that do not contain exact portions of
  the dataset.
- **Section 4.6 further narrows the stated allowed purposes to recommender
  systems development / results.** The fact that a project is academic or
  noncommercial alone does not settle authorization for an unrelated purpose.
- Section 4.10 prohibits attempts to de-anonymize data subjects.
- Sections 3.1 and 6 specify the license duration and deletion obligations.

For an application to data pricing, buyer valuation modeling, or other work
outside the stated news-recommendation purpose, seek **written confirmation or
separate authorization** from the provider before relying on the licensed
records. This documentation is an engineering precaution, not legal advice.

## What's intentionally excluded

- The official ZIP and all Parquet files
- `results/per_article_48h.npz` from the original private analysis, which
  contained **article IDs and article-level hourly exposure/click counts**
- Aggregate CSV / JSON and figures derived from the licensed real dataset,
  unless the user separately authorizes release under the applicable terms
- Credentials, account information, or other locally sensitive files

`data/` and `results/` are excluded by `.gitignore` except for explanatory
README files. The software also does not save per-article trajectories by
default. The in-memory aggregation is still required to run the bootstrap.

## Git history warning

A `.gitignore` prevents new Git tracking of matching files, but **does not
remove files already committed**. If sensitive files were previously committed,
remove them from the repository and follow GitHub guidance for cleaning Git
history; deleting a tracked file in a later commit is not sufficient to make
old revisions inaccessible.

## Code license

This repository intentionally does not choose a code license on behalf of the
researcher. A later code license would cover the repository code **only**,
not EB-NeRD data, news article text, or the data provider's intellectual
property.
