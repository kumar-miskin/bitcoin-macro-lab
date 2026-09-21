import pandas as pd
from bitcoin_macro_lab.liquidity import align_and_measure, summarize


def test_alignment_and_changes():
    dates = pd.date_range("2025-01-03", periods=20, freq="W-FRI")
    btc = pd.DataFrame({"date": dates, "close": range(100, 120)})
    liq = pd.DataFrame({"date": dates, "liquidity_index": range(200, 220)})
    got = align_and_measure(btc, liq, weeks=4)
    assert len(got) == 16
    assert summarize(got, 4)["correlation"] > 0.99
