from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def align_and_measure(btc: pd.DataFrame, liquidity: pd.DataFrame, weeks: int = 13) -> pd.DataFrame:
    """Align Friday observations and calculate rolling percentage changes."""
    b = btc.assign(date=pd.to_datetime(btc["date"], utc=True)).set_index("date")["close"].sort_index()
    l = liquidity.assign(date=pd.to_datetime(liquidity["date"], utc=True)).set_index("date")["liquidity_index"].sort_index()
    weekly_btc = b.resample("W-FRI").last().rename("btc_close")
    aligned = pd.concat([weekly_btc, l.rename("liquidity_index")], axis=1).ffill().dropna()
    aligned[f"btc_{weeks}w_return"] = aligned["btc_close"].pct_change(weeks)
    aligned[f"liquidity_{weeks}w_change"] = aligned["liquidity_index"].pct_change(weeks)
    return aligned.dropna()


def summarize(frame: pd.DataFrame, weeks: int = 13) -> pd.Series:
    x, y = f"btc_{weeks}w_return", f"liquidity_{weeks}w_change"
    return pd.Series({"observations": len(frame), "correlation": frame[x].corr(frame[y]), "window_weeks": weeks})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--btc", type=Path, required=True)
    p.add_argument("--liquidity", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--weeks", type=int, default=13)
    a = p.parse_args()
    out = a.out; out.mkdir(parents=True, exist_ok=True)
    frame = align_and_measure(pd.read_csv(a.btc), pd.read_csv(a.liquidity), a.weeks)
    frame.to_csv(out / "aligned.csv")
    summarize(frame, a.weeks).to_csv(out / "summary.csv", header=["value"])
    ax = frame.plot.scatter(x=f"liquidity_{a.weeks}w_change", y=f"btc_{a.weeks}w_return", alpha=.65)
    ax.axhline(0, color="grey", lw=.7); ax.axvline(0, color="grey", lw=.7)
    ax.set_title(f"BTC return vs liquidity change ({a.weeks}-week windows)")
    plt.tight_layout(); plt.savefig(out / "liquidity_scatter.png", dpi=180)

if __name__ == "__main__": main()
