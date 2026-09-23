# Bitcoin Macro Lab

Small, reproducible Bitcoin and macro research tools. The first analysis measures the spread between Bitcoin's rolling returns and changes in broad U.S. dollar liquidity using timestamped input data.

## Run

```bash
python -m pip install -e '.[dev]'
python -m bitcoin_macro_lab.liquidity --btc data/btc_daily.csv --liquidity data/us_liquidity_weekly.csv --out output --lag-days 2
pytest
```

## Data contract

`btc_daily.csv` must contain `date,close`. `us_liquidity_weekly.csv` must contain `date,liquidity_index`. Raw inputs are kept outside the package so each analysis can pin its sources and as-of date. Never mix publication dates with observation dates without recording the lag.

`us_liquidity_weekly.csv` may also contain `available_at` (UTC timestamp when the value was first published). When present it is used as-is; otherwise availability is `date + --lag-days`.

## Point-in-time alignment

Each Friday BTC close is treated as known at 00:00 UTC the following day. It is paired with the latest liquidity observation whose `available_at` is at or before that moment (a backward as-of join), so a release published after the close can never be used for it. `aligned.csv` keeps `liquidity_observed_at` and `liquidity_available_at` on every row. There is exactly one row per Friday, so an N-week change always spans N weeks, even when the macro series is dated on another weekday.

`lag_sensitivity.csv` and `lag_sensitivity.png` report the correlation and sample size for each lag in `--lag-grid` (default 0, 1, 2, 7, 14 days). If the relationship only looks strong at lag 0, it probably depends on data that was not yet public.

### Example: Fed balance sheet (H.4.1)

| Item | Value |
|---|---|
| Series | FRED `WALCL`, Total Assets (Less Eliminations from Consolidation), Wednesday level, millions of USD: https://fred.stlouisfed.org/series/WALCL |
| Release | H.4.1, published Thursdays around 4:30 p.m. ET, moved to the next business day after a federal holiday: https://www.federalreserve.gov/releases/h41/about.htm |
| Timezone | Dates are parsed as UTC midnight. Thursday 4:30 p.m. ET is 20:30 or 21:30 UTC, so `--lag-days 2` (Friday 00:00 UTC) is a safe fixed lag; holiday weeks need `available_at`. |
| Revisions | Past values can be revised. For a strict backtest use ALFRED vintages (https://alfred.stlouisfed.org/) and set `available_at` from the vintage date. |
| Retrieval | Record the download URL and retrieval time (UTC) with each published run. |

Suggested public sources: FRED for macro series (https://fred.stlouisfed.org/) and a clearly named market-data vendor for BTC spot history. Each published run must record exact series IDs, URLs, retrieval time, and transformations.

## Interpretation

This is descriptive research, not a causal model or trading signal. Correlation can change sign across regimes and macro series are revised. Rolling N-week changes computed every week overlap, so `observations` overstates the independent sample; `non_overlapping_windows` (observations // N) is a rough floor on the effective sample size and should be reported alongside any correlation.
