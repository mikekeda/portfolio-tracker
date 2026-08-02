"""Historical P/E helpers shared by the InstrumentYahoo model and the view layer.

`InstrumentYahoo.pes` is a {"YYYY-MM-DD": {"pe_ratio": float, …}} map merged from
Macrotrends and Wisesheets. Keys dated after today are Wisesheets forward
estimates and never belong in a historical statistic.
"""

from datetime import date, datetime
from typing import Any, Optional

# A P/E diverges as earnings approach zero, so one trough quarter dominates an
# arithmetic mean; average the earnings yield and invert instead.
MIN_PE_SAMPLE = 8
# Yahoo `trailingPE` is partly adjusted-EPS while the scraped series is GAAP;
# past this the comparison measures the basis gap rather than a re-rating.
MAX_BASIS_DIVERGENCE = 0.35
BASIS_LOOKBACK_MONTHS = 9


def _shift_years(d: date, years: int) -> date:
    """Same calendar day `years` earlier, clamping 29 Feb onto 28 Feb."""
    try:
        return d.replace(year=d.year - years)
    except ValueError:
        return d.replace(year=d.year - years, day=28)


def _shift_months(d: date, months: int) -> date:
    """Same calendar day `months` earlier, clamping onto the last valid day."""
    total = d.year * 12 + (d.month - 1) - months
    year, month = divmod(total, 12)
    day = min(d.day, [31, 29 if year % 4 == 0 and (year % 100 or year % 400 == 0) else 28,
                      31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month])
    return date(year, month + 1, day)


def pe_series(pes: dict[str, Any], today: date, years: int) -> list[tuple[date, float]]:
    """Positive historical P/Es within `years` of `today`, oldest first."""
    start = _shift_years(today, years)

    series: list[tuple[date, float]] = []
    for key, entry in pes.items():
        period_end = datetime.strptime(key, "%Y-%m-%d").date()
        if period_end < start or period_end > today:
            continue
        pe = float(entry["pe_ratio"])
        if pe > 0:
            series.append((period_end, pe))

    series.sort()
    return series


def harmonic_mean_pe(values: list[float]) -> Optional[float]:
    """Harmonic mean of P/Es — the arithmetic mean of the earnings yields, inverted."""
    if len(values) < MIN_PE_SAMPLE:
        return None
    return len(values) / sum(1.0 / pe for pe in values)


def avg_pe(pes: dict[str, Any], today: date, years: int = 5) -> Optional[float]:
    """Representative P/E over the trailing `years`, or None below MIN_PE_SAMPLE."""
    if not pes:
        return None
    return harmonic_mean_pe([pe for _, pe in pe_series(pes, today, years)])


def basis_matches(series: list[tuple[date, float]], today: date, current_pe: float) -> bool:
    """Whether `current_pe` (Yahoo) is on a comparable basis to the scraped series.

    False when the most recent scraped point is missing or disagrees beyond
    MAX_BASIS_DIVERGENCE, in which case any current-vs-history comparison is noise.
    Takes a parsed series so callers spanning several statistics parse `pes` once;
    any window reaching back at least BASIS_LOOKBACK_MONTHS gives the same answer.
    """
    cutoff = _shift_months(today, BASIS_LOOKBACK_MONTHS)
    latest = next((pe for period_end, pe in reversed(series) if period_end >= cutoff), None)
    if latest is None:
        return False
    return abs(current_pe / latest - 1.0) <= MAX_BASIS_DIVERGENCE
