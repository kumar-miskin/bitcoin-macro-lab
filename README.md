# Bitcoin Macro Lab

Small, reproducible Bitcoin and macro research tools. The first analysis measures the spread between Bitcoin's rolling returns and changes in broad U.S. dollar liquidity using timestamped input data.

## Run

```bash
python -m pip install -e '.[dev]'
python -m bitcoin_macro_lab.liquidity --btc data/btc_daily.csv --liquidity data/us_liquidity_weekly.csv --out output
pytest
```

## Data contract

`btc_daily.csv` must contain `date,close`. `us_liquidity_weekly.csv` must contain `date,liquidity_index`. Raw inputs are kept outside the package so each analysis can pin its sources and as-of date. Never mix publication dates with observation dates without recording the lag.

Suggested public sources: FRED for macro series (https://fred.stlouisfed.org/) and a clearly named market-data vendor for BTC spot history. Each published run must record exact series IDs, URLs, retrieval time, and transformations.

## Interpretation

This is descriptive research, not a causal model or trading signal. Correlation can change sign across regimes, macro series are revised, and overlapping rolling returns inflate apparent sample size.
