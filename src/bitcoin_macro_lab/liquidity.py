"""Reproducible Bitcoin macro research utilities."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_LAGS = (0, 1, 2, 7, 14)


def _weekly_btc(btc: pd.DataFrame) -> pd.DataFrame:
    b = btc.assign(date=pd.to_datetime(btc["date"], utc=True)).set_index("date")["close"].sort_index()
    weekly = b.resample("W-FRI").last().dropna().rename("btc_close").to_frame()
    # A daily close labelled D is only known at the end of D (00:00 UTC on D+1).
    weekly["btc_asof"] = weekly.index + pd.Timedelta(days=1)
    return weekly


def _liquidity_with_availability(liquidity: pd.DataFrame, lag_days: float) -> pd.DataFrame:
    l = liquidity.copy()
    l["liquidity_observed_at"] = pd.to_datetime(l["date"], utc=True)
    if "available_at" in l.columns:
        # Explicit release timestamps (e.g. from ALFRED vintages) win over a fixed lag.
        l["liquidity_available_at"] = pd.to_datetime(l["available_at"], utc=True)
    else:
        l["liquidity_available_at"] = l["liquidity_observed_at"] + pd.Timedelta(days=lag_days)
    if (l["liquidity_available_at"] < l["liquidity_observed_at"]).any():
        raise ValueError("available_at earlier than observation date")
    return l[["liquidity_observed_at", "liquidity_available_at", "liquidity_index"]].sort_values("liquidity_available_at")


def align_and_measure(btc: pd.DataFrame, liquidity: pd.DataFrame, weeks: int = 13, lag_days: float = 0) -> pd.DataFrame:
    """One row per Friday BTC close, joined to the latest liquidity value already published.

    A liquidity observation is used only if its ``liquidity_available_at`` is at or
    before the end of the Friday whose close it is paired with. ``lag_days`` sets
    availability as observation date + lag unless the input has an ``available_at``
    column. Rolling changes are over ``weeks`` Friday rows.
    """
    weekly = _weekly_btc(btc).reset_index(names="week")
    lq = _liquidity_with_availability(liquidity, lag_days)
    aligned = pd.merge_asof(
        weekly.sort_values("btc_asof"), lq,
        left_on="btc_asof", right_on="liquidity_available_at", direction="backward",
    ).set_index("week")
    aligned.index.name = "date"
    aligned = aligned.dropna(subset=["liquidity_index"])
    aligned[f"btc_{weeks}w_return"] = aligned["btc_close"].pct_change(weeks)
    aligned[f"liquidity_{weeks}w_change"] = aligned["liquidity_index"].pct_change(weeks)
    return aligned.dropna(subset=[f"btc_{weeks}w_return", f"liquidity_{weeks}w_change"])


def summarize(frame: pd.DataFrame, weeks: int = 13) -> pd.Series:
    x, y = f"btc_{weeks}w_return", f"liquidity_{weeks}w_change"
    n = len(frame)
    return pd.Series({
        "observations": n,
        "non_overlapping_windows": n // weeks,
        "correlation": frame[x].corr(frame[y]),
        "window_weeks": weeks,
    })


def lag_sensitivity(btc: pd.DataFrame, liquidity: pd.DataFrame, weeks: int = 13, lags: Iterable[float] = DEFAULT_LAGS) -> pd.DataFrame:
    """Correlation and sample size for each assumed publication lag (days)."""
    rows = []
    for lag in lags:
        s = summarize(align_and_measure(btc, liquidity, weeks, lag), weeks)
        rows.append({"lag_days": lag, **s.to_dict()})
    grid = pd.DataFrame(rows)
    int_cols = ["observations", "non_overlapping_windows", "window_weeks"]
    grid[int_cols] = grid[int_cols].astype(int)
    return grid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--btc", type=Path, required=True)
    p.add_argument("--liquidity", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--weeks", type=int, default=13)
    p.add_argument("--lag-days", type=float, default=0, help="publication lag used when the input has no available_at column")
    p.add_argument("--lag-grid", type=float, nargs="+", default=list(DEFAULT_LAGS))
    a = p.parse_args()
    out = a.out; out.mkdir(parents=True, exist_ok=True)
    btc, liq = pd.read_csv(a.btc), pd.read_csv(a.liquidity)
    frame = align_and_measure(btc, liq, a.weeks, a.lag_days)
    frame.to_csv(out / "aligned.csv")
    summarize(frame, a.weeks).to_csv(out / "summary.csv", header=["value"])
    grid = lag_sensitivity(btc, liq, a.weeks, a.lag_grid)
    grid.to_csv(out / "lag_sensitivity.csv", index=False)

    ax = frame.plot.scatter(x=f"liquidity_{a.weeks}w_change", y=f"btc_{a.weeks}w_return", alpha=.65)
    ax.axhline(0, color="grey", lw=.7); ax.axvline(0, color="grey", lw=.7)
    ax.set_title(f"BTC return vs liquidity change ({a.weeks}-week windows, lag {a.lag_days:g}d)")
    plt.tight_layout(); plt.savefig(out / "liquidity_scatter.png", dpi=180); plt.close()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    positions = range(len(grid))  # evenly spaced so short lags stay readable
    ax.plot(positions, grid.correlation, marker="o")
    ax.set_xticks(positions, [f"{l:g}d\nn={n}" for l, n in zip(grid.lag_days, grid.observations)], fontsize=8)
    ax.axhline(0, color="grey", lw=.7)
    ax.set_xlabel("assumed publication lag (days) and sample size"); ax.set_ylabel("correlation")
    ax.set_title(f"Sensitivity to publication lag ({a.weeks}-week windows)")
    plt.tight_layout(); plt.savefig(out / "lag_sensitivity.png", dpi=180); plt.close()


if __name__ == "__main__":
    main()
