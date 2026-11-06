# Local dataset directory (not committed)

Download **EB-NeRD Small** only from the official distribution site after
reviewing and accepting the applicable license:

- https://recsys.eb.dk/
- https://recsys.eb.dk/assets/pdf/general_license_terms.pdf

Supported layouts:

1. Store the officially downloaded archive as `data/ebnerd_small.zip` and run
   `python run.py`; or
2. Extract locally, then run `python run.py --data-dir /path/to/extracted`.

The extracted directory must contain:

```text
articles.parquet
train/behaviors.parquet
validation/behaviors.parquet
```

Do not commit or redistribute the files here. This folder is ignored by Git
except for this README.
