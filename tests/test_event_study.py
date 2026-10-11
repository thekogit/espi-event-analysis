import numpy as np
import pandas as pd

from espi.event_study import drop_overlapping, market_model_car, run_event_study, summarize


def synthetic_returns(jump: float = 0.05, n: int = 400, t0_pos: int = 300, seed: int = 1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    market = pd.Series(rng.normal(0, 0.01, n), index=idx)
    stock = 0.0002 + 1.2 * market + pd.Series(rng.normal(0, 0.002, n), index=idx)
    stock.iloc[t0_pos] += jump  # the event: +5% abnormal return on day 0
    return stock, market, idx[t0_pos]


def test_known_jump_is_recovered():
    stock, market, t0 = synthetic_returns()
    res = market_model_car(stock, market, t0)
    assert abs(res["beta"] - 1.2) < 0.05
    assert abs(res["ar_0"] - 0.05) < 0.006
    assert abs(res["car_0_5"] - 0.05) < 0.015


def test_event_too_early_for_estimation_window_is_skipped():
    stock, market, _ = synthetic_returns()
    assert market_model_car(stock, market, market.index[100]) is None


def test_summary_and_overlap_filter():
    stock, market, t0 = synthetic_returns()
    returns = pd.DataFrame({"abc": stock, "wig": market})
    days = returns.index
    events = pd.DataFrame({
        "geru_id": range(12),
        "stooq_symbol": ["abc"] * 12,
        "event_type": ["DIVIDEND"] * 12,
        "day0": [days[300]] + [days[300 + k] for k in [2, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90]],
    })
    per_event = run_event_study(events, returns, "wig")
    assert len(per_event) == 12
    kept = drop_overlapping(per_event, days, gap=5)
    assert len(kept) == 10  # days 302 and 304 are within 5 days of the previous event
    summary = summarize(per_event, min_n=10)
    assert summary.loc[0, "event_type"] == "DIVIDEND"
    assert summary.loc[0, "n"] == 12
