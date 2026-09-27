import pandas as pd
import pytest

from bitcoin_macro_lab.liquidity import align_and_measure, lag_sensitivity, summarize


def test_alignment_and_changes():
    dates = pd.date_range("2025-01-03", periods=20, freq="W-FRI")
    btc = pd.DataFrame({"date": dates, "close": range(100, 120)})
    liq = pd.DataFrame({"date": dates, "liquidity_index": range(200, 220)})
    got = align_and_measure(btc, liq, weeks=4)
    assert len(got) == 16
    assert summarize(got, 4)["correlation"] > 0.99


def test_wednesday_series_gives_one_row_per_friday_and_true_week_windows():
    # H.4.1-style series: Wednesday levels. The old concat+ffill interleaved
    # Wednesday rows, doubling the rows and halving the real window length.
    fridays = pd.date_range("2025-01-03", periods=20, freq="W-FRI")
    wednesdays = pd.date_range("2025-01-01", periods=20, freq="W-WED")
    btc = pd.DataFrame({"date": fridays, "close": range(100, 120)})
    liq = pd.DataFrame({"date": wednesdays, "liquidity_index": range(200, 220)})
    got = align_and_measure(btc, liq, weeks=4, lag_days=1)
    assert (got.index.dayofweek == 4).all()
    assert got.index.is_unique and len(got) == 16
    first = got.iloc[0]  # Friday 2025-01-31 vs Friday 2025-01-03, four weeks apart
    assert got.index[0] == pd.Timestamp("2025-01-31", tz="UTC")
    assert first["btc_4w_return"] == pytest.approx(104 / 100 - 1)
    assert first["liquidity_4w_change"] == pytest.approx(204 / 200 - 1)


def test_future_release_cannot_leak_into_earlier_btc_row():
    # Each Friday close is known at 00:00 UTC the next day (Jan 10 close -> Jan 11 00:00).
    btc = pd.DataFrame({"date": ["2025-01-03", "2025-01-10", "2025-01-17"], "close": [100.0, 110.0, 120.0]})
    liq = pd.DataFrame({
        "date": ["2024-12-25", "2025-01-08"],
        "liquidity_index": [1.0, 1000.0],  # huge jump in the later observation
    })
    jan10 = pd.Timestamp("2025-01-10", tz="UTC")
    # Lag 2d: the Jan 8 value is published Jan 10 00:00, before the Jan 10 close.
    two = align_and_measure(btc, liq, weeks=1, lag_days=2)
    assert two.loc[jan10, "liquidity_index"] == 1000.0
    # Lag 4d: published Jan 12, after the Jan 10 close, so Jan 10 must still see 1.0.
    four = align_and_measure(btc, liq, weeks=1, lag_days=4)
    assert four.loc[jan10, "liquidity_index"] == 1.0
    assert four.loc[jan10, "liquidity_1w_change"] == 0.0
    assert four.loc[pd.Timestamp("2025-01-17", tz="UTC"), "liquidity_index"] == 1000.0
    for frame in (two, four):
        assert (frame.liquidity_available_at <= frame.btc_asof).all()


def test_explicit_available_at_overrides_lag():
    btc = pd.DataFrame({"date": ["2025-01-03", "2025-01-10"], "close": [100.0, 110.0]})
    liq = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-08"],
        "available_at": ["2025-01-02T21:30:00Z", "2025-01-09T21:30:00Z"],
        "liquidity_index": [100.0, 120.0],
    })
    got = align_and_measure(btc, liq, weeks=1, lag_days=30)
    assert got["liquidity_1w_change"].iloc[0] == pytest.approx(0.2)
    assert got["liquidity_available_at"].iloc[0] == pd.Timestamp("2025-01-09T21:30:00Z")


def test_lag_grid_reports_sample_size_per_lag():
    fridays = pd.date_range("2025-01-03", periods=30, freq="W-FRI")
    wednesdays = pd.date_range("2025-01-01", periods=30, freq="W-WED")
    btc = pd.DataFrame({"date": fridays, "close": [100 + i * i for i in range(30)]})
    liq = pd.DataFrame({"date": wednesdays, "liquidity_index": [200 + i for i in range(30)]})
    grid = lag_sensitivity(btc, liq, weeks=4, lags=[0, 2, 14])
    assert grid.lag_days.tolist() == [0, 2, 14]
    assert grid.observations.tolist() == [26, 26, 24]
    assert grid.non_overlapping_windows.tolist() == [6, 6, 6]


def test_first_liquidity_release_does_not_shorten_calendar_window():
    # First three Fridays have no published macro value. A 2-week return must
    # compare Feb 7 with Jan 24, not Jan 31 with Jan 24 (only one week apart).
    fridays = pd.date_range("2025-01-03", periods=7, freq="W-FRI")
    btc = pd.DataFrame({"date": fridays, "close": [100, 105, 110, 115, 120, 125, 130]})
    liq = pd.DataFrame({
        "date": ["2025-01-22", "2025-01-29", "2025-02-05", "2025-02-12"],
        "available_at": ["2025-01-23T21:30:00Z", "2025-01-30T21:30:00Z",
                         "2025-02-06T21:30:00Z", "2025-02-13T21:30:00Z"],
        "liquidity_index": [200, 210, 220, 230],
    })
    got = align_and_measure(btc, liq, weeks=2)
    assert got.index[0] == pd.Timestamp("2025-02-07", tz="UTC")
    assert len(got) == 2
    assert got.iloc[0]["liquidity_2w_change"] == pytest.approx(220/200-1)


def test_a_missing_availability_gap_never_forward_fills_or_stretches_window():
    fridays = pd.date_range("2025-01-03", periods=6, freq="W-FRI")
    btc = pd.DataFrame({"date": fridays, "close": [100, 110, 120, 130, 140, 150]})
    # First liquidity release arrives only after two Friday closes. For a 1-week
    # window Jan 17 is invalid; Jan 24 compares to Jan 17, not Jan 10.
    liq = pd.DataFrame({"date": ["2025-01-08", "2025-01-22"],
                        "available_at": ["2025-01-12T00:00:00Z", "2025-01-23T00:00:00Z"],
                        "liquidity_index": [100, 120]})
    got = align_and_measure(btc, liq, weeks=1)
    assert got.index[0] == pd.Timestamp("2025-01-24", tz="UTC")
    assert got.loc[pd.Timestamp("2025-01-24", tz="UTC"), "liquidity_1w_change"] == pytest.approx(.2)


def test_nonpositive_window_is_rejected():
    btc = pd.DataFrame({"date": ["2025-01-03"], "close": [100]})
    liq = pd.DataFrame({"date": ["2025-01-01"], "liquidity_index": [200]})
    with pytest.raises(ValueError, match="positive integer"):
        align_and_measure(btc, liq, weeks=0)


def test_null_macro_release_cannot_bridge_a_gap_in_weekly_return():
    fridays = pd.date_range("2025-01-03", periods=6, freq="W-FRI")
    btc = pd.DataFrame({"date": fridays, "close": [100, 110, 120, 130, 140, 150]})
    liq = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-08", "2025-01-15", "2025-01-22", "2025-01-29", "2025-02-05"],
        "liquidity_index": [100., 105., float("nan"), 115., 120., 125.],
    })
    got = align_and_measure(btc, liq, weeks=2, lag_days=1)
    # Jan 17 value is null: Jan 31 must not compare to Jan 10 as a "2-week" change.
    assert pd.Timestamp("2025-01-31", tz="UTC") not in got.index
    assert pd.Timestamp("2025-02-07", tz="UTC") in got.index
    assert got.loc[pd.Timestamp("2025-02-07", tz="UTC"), "liquidity_2w_change"] == pytest.approx(125/115-1)
