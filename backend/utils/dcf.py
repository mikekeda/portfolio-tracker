import asyncio
from typing import Dict, List, Optional, TypedDict, Any

from models import Instrument

# --- Constants ---
# More conventional to set a standard terminal growth rate representing long-term economic growth.
TERMINAL_GROWTH_RATE = 0.025  # 2.5%

SECTOR_GROWTH_DEFAULTS: Dict[str, float] = {
    "Technology": 0.15,
    "Healthcare": 0.07,
    "Industrials": 0.05,
    "Financial Services": 0.03,
    "Consumer Cyclical": 0.06,
    "Consumer Defensive": 0.04,
    "Energy": 0.04,
    "Utilities": 0.03,
    "Communication Services": 0.06,
    "Real Estate": 0.03,
    "Basic Materials": 0.04,
}


class DcfInputs(TypedDict):
    current_fcf: Optional[float]
    shares_outstanding: Optional[float]
    total_cash: float
    total_debt: float
    wacc: float
    initial_growth_rate: float
    terminal_growth_rate: float


def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamps a number to a given range."""
    return max(lo, min(hi, x))


def estimate_wacc(market_cap: Optional[float], operating_margin: Optional[float]) -> float:
    """
    Estimates the Weighted Average Cost of Capital (WACC) based on company size and profitability.
    A higher WACC (discount rate) is used for smaller, less profitable companies.
    """
    if not market_cap:
        return 0.10  # Default for unknown market cap

    # Base rate by market cap tier
    if market_cap > 2e11:  # Mega-cap (> $200B)
        base = 0.08
    elif market_cap > 2e10:  # Large-cap (> $20B)
        base = 0.09
    elif market_cap > 2e9:  # Mid-cap (> $2B)
        base = 0.11
    else:  # Small-cap
        base = 0.13

    # Adjust for profitability
    if isinstance(operating_margin, (int, float)):
        if operating_margin < 0.05:  # Low margin
            base += 0.02
        elif operating_margin > 0.20:  # High margin
            base -= 0.01

    return _clamp(base, 0.07, 0.20)  # Clamp to a reasonable range


def _safe_number(x: Any) -> Optional[float]:
    """Safely converts a value to float, returning None if invalid."""
    return (
        float(x)
        if isinstance(x, (int, float))
        and not (isinstance(x, float) and (x != x or x == float("inf") or x == float("-inf")))
        else None
    )


def _median(xs: List[float]) -> float:
    """Calculates the median of a list of numbers."""
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    if n == 0:
        return 0.0
    m = n // 2
    return xs_sorted[m] if n % 2 == 1 else 0.5 * (xs_sorted[m - 1] + xs_sorted[m])


def _calculate_cagr(first: float, last: float, periods: int) -> Optional[float]:
    """Calculates Compound Annual Growth Rate."""
    if periods <= 0 or first <= 0 or last <= 0:
        return None
    try:
        return (last / first) ** (1.0 / periods) - 1.0
    except (ValueError, ZeroDivisionError):
        return None


def _extract_trailing_fcf(cashflow: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """
    Extracts a smoothed, forward-looking Free Cash Flow (FCF) from historical data.
    Also returns the historical FCF CAGR to inform the growth rate assumption.

    Returns:
        A tuple containing (smoothed_fcf, fcf_cagr).
    """
    if not isinstance(cashflow, dict) or not cashflow:
        return None, None

    # 1) Build a chronological series of historical FCF
    years = sorted(cashflow.keys())
    series: List[float] = []
    for y in years:
        row = cashflow.get(y) or {}
        fcf = _safe_number(row.get("Free Cash Flow"))
        if fcf is None:
            ocf = _safe_number(row.get("Operating Cash Flow"))
            capex = _safe_number(row.get("Capital Expenditure"))
            if ocf is not None and capex is not None:
                fcf = ocf + capex  # Note: Capex is usually negative
        if fcf is not None:
            series.append(fcf)

    # 2) Use only positive FCF values for a stable DCF base
    pos_fcf = [v for v in series if v > 0]
    if not pos_fcf:
        return None, None

    # 3) Calculate components for smoothing
    last_fcf = pos_fcf[-1]
    median_last_3 = _median(pos_fcf[-3:])
    fcf_cagr = _calculate_cagr(pos_fcf[0], pos_fcf[-1], len(pos_fcf) - 1) if len(pos_fcf) >= 2 else None

    forecasted_fcf = None
    if fcf_cagr is not None:
        # Clamp growth rate for forecasting to avoid extreme swings
        clamped_g = _clamp(fcf_cagr, -0.30, 0.40)
        forecasted_fcf = last_fcf * (1.0 + clamped_g)

    # 4) Blend components with dynamic weights
    parts = [
        (0.40, last_fcf),
        (0.40, median_last_3),
    ]
    if forecasted_fcf is not None:
        parts.append((0.20, forecasted_fcf))

    total_w = sum(w for w, _ in parts)
    if total_w <= 0:
        return last_fcf, fcf_cagr

    smoothed_fcf = sum((w / total_w) * v for w, v in parts)

    # 5) Sanity clamp around the last FCF value
    final_fcf = _clamp(smoothed_fcf, last_fcf * 0.5, last_fcf * 1.8)

    return final_fcf, fcf_cagr


def _derive_shares_if_needed(info: Dict[str, Any], candidate_shares: Optional[float]) -> Optional[float]:
    """Uses market cap and price to derive or validate shares outstanding."""
    if not isinstance(candidate_shares, (int, float)) or candidate_shares <= 0:
        candidate_shares = None

    mcap = _safe_number(info.get("marketCap"))
    price = _safe_number(info.get("currentPrice"))
    if mcap is None or price is None or price <= 0:
        return candidate_shares

    derived_shares = mcap / price
    if candidate_shares is None:
        return derived_shares

    # If the provided shares differ by >50% from derived, prefer derived.
    # This often corrects for ADRs or data inconsistencies.
    if abs(candidate_shares - derived_shares) / derived_shares > 0.5:
        return derived_shares

    return candidate_shares


def _estimate_dcf_inputs(instrument: Instrument) -> DcfInputs:
    """Estimates all necessary inputs for a DCF valuation from instrument data."""
    info = instrument.yahoo.info or {}
    cashflow = instrument.yahoo.cashflow or {}
    sector = instrument.sector

    current_fcf, fcf_cagr = _extract_trailing_fcf(cashflow)

    raw_shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
    shares = _derive_shares_if_needed(info, _safe_number(raw_shares))

    total_cash: float = _safe_number(info.get("totalCash")) or 0.0
    total_debt: float = _safe_number(info.get("totalDebt")) or 0.0

    # Estimate initial growth rate, preferring analyst estimates, then historical FCF growth
    rev_g = _safe_number(info.get("revenueGrowth"))
    if rev_g is not None:
        initial_growth = rev_g
    elif fcf_cagr is not None:
        initial_growth = fcf_cagr
    else:
        initial_growth = SECTOR_GROWTH_DEFAULTS.get(sector, 0.08)

    initial_growth = _clamp(initial_growth, -0.20, 0.30)  # Clamp to a safer range

    market_cap = _safe_number(info.get("marketCap"))
    margin = _safe_number(info.get("operatingMargins"))
    wacc = estimate_wacc(market_cap, margin)

    return {
        "current_fcf": current_fcf,
        "shares_outstanding": shares,
        "total_cash": total_cash,
        "total_debt": total_debt,
        "wacc": wacc,
        "initial_growth_rate": initial_growth,
        "terminal_growth_rate": TERMINAL_GROWTH_RATE,
    }


async def get_dcf_prices(
    instruments: List[Instrument],
    years: int = 10,
    wacc: Optional[float] = None,
    growth: Optional[float] = None,
    terminal: Optional[float] = None,
) -> List[Optional[float]]:
    """Calculates DCF prices for a list of instruments concurrently."""
    return await asyncio.gather(
        *[_get_dcf_price(instrument, years, wacc, growth, terminal) for instrument in instruments]
    )


async def _get_dcf_price(
    instrument: Instrument,
    years: int = 10,
    wacc_override: Optional[float] = None,
    growth_override: Optional[float] = None,
    terminal_override: Optional[float] = None,
    allow_negative: bool = False,
) -> Optional[float]:
    """
    Calculates the intrinsic value per share for a single instrument using a 2-stage DCF model.

    Args:
        instrument: The instrument object with financial data.
        years: The number of years for the high-growth stage.
        wacc_override, growth_override, terminal_override: Optional values to override estimates.
        allow_negative: If True, returns negative DCF values; otherwise returns None.

    Returns:
        The calculated intrinsic value per share, or None if inputs are invalid.
    """
    est = _estimate_dcf_inputs(instrument)

    # Apply any user-provided overrides
    wacc = wacc_override if wacc_override is not None else est["wacc"]
    g0 = growth_override if growth_override is not None else est["initial_growth_rate"]
    gT = terminal_override if terminal_override is not None else est["terminal_growth_rate"]

    # --- Input Validation ---
    if (
        est["current_fcf"] is None
        or est["current_fcf"] <= 0
        or est["shares_outstanding"] is None
        or est["shares_outstanding"] <= 0
    ):
        return None

    # --- DCF Projection ---
    horizon_years = max(1, min(30, years))
    r = _clamp(wacc, 0.05, 0.20)

    # Ensure the discount rate is higher than the terminal growth rate
    min_spread = 0.03
    if r <= gT + min_spread:
        r = gT + min_spread

    # Project future free cash flows in the high-growth stage
    g_step = (gT - g0) / max(1, horizon_years - 1)
    g_list = [g0 + g_step * i for i in range(horizon_years)]

    fcf = est["current_fcf"]
    projected_fcf: List[float] = []
    for g in g_list:
        fcf *= 1.0 + g
        projected_fcf.append(fcf)

    # Discount the projected cash flows to present value
    inv_1pr = 1.0 / (1.0 + r)
    pv_fcf = sum(f_t * (inv_1pr ** (t + 1)) for t, f_t in enumerate(projected_fcf))

    # Calculate terminal value and discount to present value
    terminal_fcf = projected_fcf[-1] * (1.0 + gT)
    terminal_value = terminal_fcf / (r - gT)
    pv_terminal = terminal_value * (inv_1pr**horizon_years)

    # --- Final Valuation ---
    enterprise_value = pv_fcf + pv_terminal
    equity_value = enterprise_value + est["total_cash"] - est["total_debt"]
    per_share_value = equity_value / est["shares_outstanding"]

    if not allow_negative and per_share_value <= 0:
        return None

    return per_share_value
