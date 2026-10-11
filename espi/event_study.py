"""Market-model event study and Apple-inspired visualization engine."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from .schema import EVENT_DISPLAY_NAMES

EST_WINDOW = (-250, -30)        # Estimation window in trading days relative to Day 0
CAR_WINDOWS = ((0, 1), (0, 5))  # Event windows (short-term impact, multi-day drift)
MIN_EST_OBS = 120               # Minimum valid observations required for OLS market model


def market_model_car(stock: pd.Series, market: pd.Series, t0: pd.Timestamp) -> Optional[dict]:
    """Calculate market-model Cumulative Abnormal Returns for a single event."""
    idx = market.index
    if t0 not in idx:
        return None
    p0 = idx.get_loc(t0)
    lo, hi = p0 + EST_WINDOW[0], p0 + EST_WINDOW[1]
    last_needed = p0 + max(w[1] for w in CAR_WINDOWS)
    if lo < 0 or last_needed >= len(idx):
        return None

    est = pd.DataFrame({"r": stock.iloc[lo: hi + 1], "m": market.iloc[lo: hi + 1]}).dropna()
    if len(est) < MIN_EST_OBS:
        return None

    # OLS estimation: R_i = alpha + beta * R_m
    beta, alpha = np.polyfit(est["m"].values, est["r"].values, 1)
    resid_sd = float(np.std(est["r"] - (alpha + beta * est["m"]), ddof=2))

    out = {"alpha": float(alpha), "beta": float(beta), "resid_sd": resid_sd}
    for a, b in CAR_WINDOWS:
        win = slice(p0 + a, p0 + b + 1)
        ar = stock.iloc[win] - (alpha + beta * market.iloc[win])
        if ar.isna().any():
            return None
        out[f"car_{a}_{b}"] = float(ar.sum())
    out["ar_0"] = float(stock.iloc[p0] - (alpha + beta * market.iloc[p0]))
    return out


def run_event_study(events: pd.DataFrame, returns: pd.DataFrame, market_col: str) -> pd.DataFrame:
    """Execute market model across all valid events."""
    rows = []
    market = returns[market_col]
    for ev in events.itertuples(index=False):
        if ev.stooq_symbol not in returns:
            continue
        res = market_model_car(returns[ev.stooq_symbol], market, pd.Timestamp(ev.day0))
        if res:
            rows.append({
                "geru_id": ev.geru_id,
                "ticker_symbol": ev.stooq_symbol,
                "event_type": ev.event_type,
                "day0": ev.day0,
                **res
            })
    return pd.DataFrame(rows)


def drop_overlapping(per_event: pd.DataFrame, trading_days: pd.DatetimeIndex, gap: int = 5) -> pd.DataFrame:
    """Filter out clustered events occurring within `gap` trading days of a prior announcement."""
    pos_series = pd.Series(np.arange(len(trading_days)), index=trading_days)
    df = per_event.copy()
    df["_pos"] = pos_series.reindex(pd.to_datetime(df["day0"])).to_numpy()
    df = df.sort_values(["ticker_symbol", "_pos"])
    previous = df.groupby("ticker_symbol")["_pos"].shift(1)
    keep = previous.isna() | ((df["_pos"] - previous) > gap)
    return df.loc[keep].drop(columns="_pos")


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    """Compute 95% non-parametric bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(per_event: pd.DataFrame, min_n: int = 10) -> pd.DataFrame:
    """Aggregate per-event results into a category-level statistical summary."""
    rows = []
    for etype, grp in per_event.groupby("event_type"):
        if len(grp) < min_n:
            continue
        row = {"event_type": etype, "n": len(grp)}
        for a, b in CAR_WINDOWS:
            vals = grp[f"car_{a}_{b}"].to_numpy()
            lo, hi = bootstrap_ci(vals)
            row[f"mean_car_{a}_{b}_pct"] = round(100 * vals.mean(), 3)
            row[f"ci95_car_{a}_{b}_pct"] = f"[{100 * lo:.3f}, {100 * hi:.3f}]"
            std_err = vals.std(ddof=1) / np.sqrt(len(vals))
            row[f"t_car_{a}_{b}"] = round(vals.mean() / std_err, 2) if std_err > 0 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def plot_apple_style(per_event: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> Path:
    """Render a clean, publication-grade visualization inspired by Apple design language."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Set clean system font fallbacks
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort categories by mean CAR [0, +5] ascending
    df_sorted = summary.sort_values("mean_car_0_5_pct", ascending=True).reset_index(drop=True)
    
    categories = []
    sample_sizes = []
    mean_vals = []
    ci_lows = []
    ci_highs = []

    for _, row in df_sorted.iterrows():
        raw_type = row["event_type"]
        categories.append(EVENT_DISPLAY_NAMES.get(raw_type, raw_type.replace("_", " ").title()))
        sample_sizes.append(int(row["n"]))
        
        vals = per_event.loc[per_event["event_type"] == raw_type, "car_0_5"].to_numpy() * 100
        mean = float(vals.mean())
        lo, hi = bootstrap_ci(vals / 100)
        mean_vals.append(mean)
        ci_lows.append(100 * lo)
        ci_highs.append(100 * hi)

    n_cats = len(categories)
    y_pos = np.arange(n_cats)

    # Apple Design Palette
    COLOR_BG = "#FFFFFF"
    COLOR_CARD = "#F9F9FB"
    COLOR_TEXT_PRIMARY = "#1D1D1F"     # Apple Dark Charcoal
    COLOR_TEXT_SECONDARY = "#86868B"   # Apple Muted Neutral
    COLOR_LINE_ZERO = "#AEAEB2"        # Apple Mid Gray
    COLOR_GRID = "#E5E5EA"             # Apple Light Divider
    COLOR_BLUE = "#0071E3"             # Apple System Blue
    COLOR_GREEN = "#34C759"            # Apple System Green
    COLOR_RED = "#FF3B30"              # Apple System Red

    fig = plt.figure(figsize=(11.5, 7.5), dpi=300, facecolor=COLOR_BG)
    ax = fig.add_axes([0.28, 0.12, 0.67, 0.74])
    ax.set_facecolor(COLOR_BG)

    # Subtle horizontal zebra shading
    for y in y_pos:
        if y % 2 == 0:
            ax.axhspan(y - 0.44, y + 0.44, color=COLOR_CARD, alpha=0.8, zorder=0)

    # Delicate vertical gridlines
    for x_grid in np.arange(-3.0, 3.5, 0.5):
        if abs(x_grid) > 0.001:
            ax.axvline(x_grid, color=COLOR_GRID, linestyle=":", linewidth=0.75, alpha=0.7, zorder=1)

    # Zero benchmark reference line
    ax.axvline(0.0, color=COLOR_LINE_ZERO, linestyle="--", linewidth=1.2, zorder=2)

    # Plot confidence interval intervals and point estimates
    for i in range(n_cats):
        mean = mean_vals[i]
        lo = ci_lows[i]
        hi = ci_highs[i]
        
        # Color coding: Green for positive, Red for negative, Blue for close to zero
        if mean > 0.3:
            point_color = COLOR_GREEN
        elif mean < -0.3:
            point_color = COLOR_RED
        else:
            point_color = COLOR_BLUE

        # Confidence interval whisker with capped ends
        ax.plot([lo, hi], [y_pos[i], y_pos[i]], color=point_color, linewidth=2.5, alpha=0.8, zorder=3, solid_capstyle="round")
        ax.plot([lo, lo], [y_pos[i] - 0.14, y_pos[i] + 0.14], color=point_color, linewidth=1.8, alpha=0.8, zorder=3)
        ax.plot([hi, hi], [y_pos[i] - 0.14, y_pos[i] + 0.14], color=point_color, linewidth=1.8, alpha=0.8, zorder=3)

        # Point estimate marker with white ring
        ax.scatter(mean, y_pos[i], s=72, color=point_color, edgecolor=COLOR_BG, linewidth=2.0, zorder=4)

        # Floating badge above the marker (avoids horizontal line collision)
        sign_prefix = "+" if mean > 0 else ""
        ax.text(
            mean, y_pos[i] + 0.24,
            f"{sign_prefix}{mean:.2f}%",
            va="bottom", ha="center",
            fontsize=8.5, fontweight="bold",
            color=point_color,
            bbox=dict(boxstyle="round,pad=0.18", fc=COLOR_BG, ec="none", alpha=0.85),
            zorder=5
        )

    # Format Y-axis labels with aligned sample counts
    y_labels = [f"{cat}  (n = {n})" for cat, n in zip(categories, sample_sizes)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=10.0, fontweight="500", color=COLOR_TEXT_PRIMARY)
    ax.tick_params(axis="y", length=0, pad=12)

    # X-axis label & ticks
    ax.set_xlabel("5-Day Cumulative Abnormal Return (CAR [0, +5 Days], %)", fontsize=10.5, fontweight="600", color=COLOR_TEXT_PRIMARY, labelpad=12)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:+.1f}%" if abs(x) > 0.001 else "0.0%"))
    ax.tick_params(axis="x", labelsize=9.5, colors=COLOR_TEXT_SECONDARY)

    # Remove enclosing frame borders
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_GRID)
    ax.spines["bottom"].set_linewidth(1.0)

    # X-axis symmetrical bounds
    max_bound = max(abs(min(ci_lows)), abs(max(ci_highs))) + 0.65
    ax.set_xlim(-max_bound, max_bound)
    ax.set_ylim(-0.6, n_cats - 0.3)

    # Header & Subtitle (clean left alignment at top of card)
    fig.text(0.05, 0.94, "Warsaw Stock Exchange Corporate Event Study", fontsize=16.0, fontweight="bold", color=COLOR_TEXT_PRIMARY)
    fig.text(0.05, 0.90, "Mean 5-Day Cumulative Abnormal Return (CAR [0, +5]) with 95% Bootstrap Confidence Intervals relative to WIG Index", fontsize=10.0, fontweight="400", color=COLOR_TEXT_SECONDARY)

    # Footnote
    fig.text(
        0.05, 0.03,
        "Methodology: OLS Market Model (MacKinlay 1997) • Estimation Window: [-250, -30] Trading Days • Sample: 720 ESPI Disclosures (2025–2026)",
        fontsize=8.0, color=COLOR_TEXT_SECONDARY
    )

    fig.savefig(out_path, dpi=300, facecolor=COLOR_BG)
    plt.close(fig)
    return out_path
