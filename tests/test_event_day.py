from datetime import datetime

import pandas as pd

from espi.prices import event_day

# Mon 2026-10-05 .. Fri 2026-10-09, then Mon 2026-10-12 (weekend skipped)
DAYS = pd.DatetimeIndex(["2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09", "2026-10-12"])


def test_before_session_is_same_day():
    assert event_day(datetime(2026, 10, 6, 8, 15), DAYS) == pd.Timestamp("2026-10-06")


def test_during_session_is_same_day():
    assert event_day(datetime(2026, 10, 6, 13, 0), DAYS) == pd.Timestamp("2026-10-06")


def test_after_close_is_next_trading_day():
    assert event_day(datetime(2026, 10, 6, 23, 30), DAYS) == pd.Timestamp("2026-10-07")


def test_friday_evening_rolls_to_monday():
    assert event_day(datetime(2026, 10, 9, 18, 0), DAYS) == pd.Timestamp("2026-10-12")


def test_saturday_rolls_to_monday():
    assert event_day(datetime(2026, 10, 10, 11, 0), DAYS) == pd.Timestamp("2026-10-12")


def test_after_last_known_day_is_none():
    assert event_day(datetime(2026, 10, 12, 18, 0), DAYS) is None
