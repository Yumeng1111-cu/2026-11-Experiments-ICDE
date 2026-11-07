# Data

## Processed sample

`processed/fcc_auction97_block_g_exit_bids.csv` contains 4,286 observations,
one for each block-G `(license, bidder)` pair after retaining its last observed
bid.

| Column | Description |
|---|---|
| `item_name` | FCC license identifier |
| `item_description` | FCC market description |
| `market` | FCC market code |
| `block` | Spectrum block, fixed to `G` |
| `bidder_name` | Public bidder name in the FCC record |
| `bidder_frn` | FCC registration number, stored as text |
| `round` | Last round with an observed bid for the pair |
| `bid_amount` | Exit bid in U.S. dollars |
| `log_bid` | Natural logarithm of the exit bid |
| `base_valuation` | Log bid rescaled to `[1, 10]` within block G |

## Original data

The original FCC ZIP and XML are not included in this repository. For full
reproduction, run:

```bash
python run_all.py --from-raw --download
```

The raw archive is downloaded locally to `data/raw/`, which is excluded by
`.gitignore`. The SHA-256 checksum of the archive used for this analysis is:

```text
12fb07207a4ba9d15180af6e906263af5e4484a5a4618033fd469f769f66c8e3
```
