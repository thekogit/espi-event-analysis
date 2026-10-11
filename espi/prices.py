"""Daily price series loading and session cut-off calculation."""
from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd

COLUMN_MAP = {
    "Data": "date", "Date": "date",
    "Zamkniecie": "close", "Zamknięcie": "close", "Close": "close",
}
SESSION_CLOSE = time(17, 0)  # GPW Warsaw trading session closing auction boundary


def load_stooq_csv(path: Path) -> pd.Series:
    """Load daily close price series from a Stooq CSV export."""
    df = pd.read_csv(path)
    df = df.rename(columns={c: COLUMN_MAP[c] for c in df.columns if c in COLUMN_MAP})
    if not {"date", "close"} <= set(df.columns):
        raise ValueError(f"{path.name}: missing required Date/Close columns")
    s = pd.Series(
        df["close"].astype(float).values,
        index=pd.to_datetime(df["date"]),
        name=path.stem.lower()
    )
    return s.sort_index()


def load_prices(prices_dir: Path, symbols: list[str]) -> pd.DataFrame:
    """Load wide DataFrame of daily closing prices across given symbols."""
    series = []
    for sym in symbols:
        path = prices_dir / f"{sym.lower()}_d.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing price series file: {path}")
        series.append(load_stooq_csv(path).rename(sym.lower()))
    return pd.concat(series, axis=1).sort_index()


def simple_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns from closing prices."""
    return closes.pct_change(fill_method=None)


def get_event_day(published_at: datetime, trading_days: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    """Calculate effective trading Day 0.
    
    If published at or after 17:00 CET, or during non-trading days/weekends,
    Day 0 rolls to the next active trading session.
    """
    day = pd.Timestamp(published_at.date())
    if published_at.time() >= SESSION_CLOSE:
        day = day + timedelta(days=1)
    pos = trading_days.searchsorted(day, side="left")
    if pos >= len(trading_days):
        return None
    return trading_days[pos]


# Alias for backward compatibility
event_day = get_event_day
