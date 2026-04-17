"""
Drawdown helpers.

Pure math on a series of daily values (portfolio GBP value or a benchmark
close price). Keep these functions data-source agnostic — the view layer
decides where the values come from (PortfolioDaily vs PricesDaily).
"""

from typing import Optional, Sequence


def compute_max_drawdown(values: Sequence[Optional[float]]) -> dict[str, float]:
    """Return the worst peak-to-trough drawdown over a value series.

    Returns a dict with:
      - max_drawdown_pct: negative percentage (e.g. -18.4 means the value
        dropped 18.4% from its prior peak); 0.0 if never drew down.
      - max_drawdown_abs: negative amount in the same units as `values`
        (only meaningful when values are absolute currency amounts).
      - drawdown_duration_days: number of daily samples from the peak that
        preceded the worst trough to the trough itself.
    """
    peak = 0.0
    max_dd_pct = 0.0
    max_dd_abs = 0.0
    peak_idx = 0
    best_peak_idx = 0
    best_trough_idx = 0

    for i, v in enumerate(values):
        if v is None:
            continue
        if v > peak:
            peak = v
            peak_idx = i
        elif peak > 0:
            dd_abs = v - peak
            dd_pct = (dd_abs / peak) * 100.0
            if dd_pct < max_dd_pct:
                max_dd_pct = dd_pct
                max_dd_abs = dd_abs
                best_peak_idx = peak_idx
                best_trough_idx = i

    return {
        "max_drawdown_pct": max_dd_pct,
        "max_drawdown_abs": max_dd_abs,
        "drawdown_duration_days": best_trough_idx - best_peak_idx,
    }


def underwater_series(values: Sequence[Optional[float]]) -> list[Optional[float]]:
    """Underwater series expressed in percent: value[t] / running_max(value[0..t]) - 1.

    Output list is the same length as `values` and is always <= 0.
    None inputs are passed through as None so callers can align with their date axis.
    """
    out: list[Optional[float]] = []
    peak = 0.0
    for v in values:
        if v is None:
            out.append(None)
            continue
        if v > peak:
            peak = v
        if peak <= 0:
            out.append(0.0)
        else:
            out.append((v / peak - 1.0) * 100.0)
    return out
