"""Unified pipeline execution for espi-event-analysis."""
from __future__ import annotations

import pandas as pd
from . import config
from .event_study import drop_overlapping, plot_apple_style, run_event_study, summarize
from .prices import load_prices, simple_returns


def run_pipeline(no_overlap: bool = False) -> pd.DataFrame:
    """Execute the end-to-end event study and produce publication-ready results."""
    print("Loading events and market price data...")
    if not config.EVENTS_PATH.exists():
        raise FileNotFoundError(f"Missing events database: {config.EVENTS_PATH}")

    events = pd.read_parquet(config.EVENTS_PATH).dropna(subset=["day0", "stooq_symbol"])
    symbols = sorted(set(events["stooq_symbol"]) | {config.MARKET_SYMBOL})
    
    closes = load_prices(config.PRICES_DIR, symbols)
    returns = simple_returns(closes)

    print(f"Executing market model across {len(events)} corporate announcements...")
    per_event = run_event_study(events, returns, config.MARKET_SYMBOL)

    if no_overlap:
        print("Filtering out clustered announcements (gap > 5 trading days)...")
        per_event = drop_overlapping(per_event, returns.index, gap=5)

    summary = summarize(per_event)

    # Save CSV tables to results directory
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_event_path = config.RESULTS_DIR / "event_study_per_event.csv"
    summary_path = config.RESULTS_DIR / "event_study_summary.csv"
    chart_path = config.RESULTS_DIR / "event_study_car.png"

    per_event.to_csv(per_event_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Generating Apple-style high-resolution visualization at {chart_path}...")
    plot_apple_style(per_event, summary, chart_path)

    print("Pipeline run successfully completed.")
    return summary
