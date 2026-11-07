from __future__ import annotations
import datetime as dt
from pathlib import Path
import numpy as np


def _micro(value) -> int:
    """Interpret Parquet timestamp as UTC without changing its displayed wall time."""
    if isinstance(value, np.datetime64):
        return int(value.astype('datetime64[us]').astype('int64'))
    if hasattr(value, 'value') and not isinstance(value, (int, np.integer)):
        return int(value.value // 1000)
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(dt.timezone.utc).replace(tzinfo=None)
        epoch = dt.datetime(1970, 1, 1)
        d = value - epoch
        return d.days * 86400000000 + d.seconds * 1000000 + d.microseconds
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise TypeError(f'Unexpected datetime: {value!r} ({type(value)})')


def read_articles(path: Path):
    import pyarrow.parquet as pq
    tab = pq.read_table(path, columns=['article_id', 'published_time']).to_pydict()
    pub = {int(i): _micro(t) for i, t in zip(tab['article_id'], tab['published_time']) if t is not None}
    return pub, len(tab['article_id'])


def read_impression_range(paths: list[Path]):
    import pyarrow.parquet as pq
    lo, hi = None, None
    per_split = {}
    for path in paths:
        ar = pq.read_table(path, columns=['impression_time']).column(0)
        ar_us = ar.cast('timestamp[us]').cast('int64').to_numpy()
        this_lo, this_hi, n = int(np.min(ar_us)), int(np.max(ar_us)), len(ar_us)
        lo = this_lo if lo is None else min(lo, this_lo)
        hi = this_hi if hi is None else max(hi, this_hi)
        per_split[path.parent.name] = {'rows': int(n), 'start_us': this_lo, 'end_us': this_hi}
    return lo, hi, per_split


def iter_behaviors(path: Path, batch_size: int = 4096):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    cols = ['impression_time', 'article_ids_inview', 'article_ids_clicked']
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        b = batch.to_pydict()
        for t, iv, ck in zip(b['impression_time'], b['article_ids_inview'], b['article_ids_clicked']):
            yield _micro(t), (iv or []), (ck or [])
