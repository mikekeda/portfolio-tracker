import asyncio
from typing import Any, Optional, TypedDict

import aiohttp

from config import EQUITY_RISK_PREMIUM
from models import Instrument
from backend.utils.market_data import get_risk_free_rates

# --- Constants ---
TERMINAL_GROWTH_RATE = 0.025  # 2.5% — long-run nominal GDP growth

# Sectors where FCF-based DCF is meaningless. Banks' "free cash flow" is not
# comparable to an industrial company's; use P/BV or DDM for these instead.
SECTORS_WITHOUT_DCF: frozenset[str] = frozenset({"Financial Services"})

# Corporate tax rates by company domicile country.
# Used to compute after-tax cost of debt in WACC.
# Defaults to 0.25 (a conservative international average) for unknown countries.
COUNTRY_TAX_RATES: dict[str, float] = {
    "United States": 0.21,
    "United Kingdom": 0.25,
    "France": 0.25,
    "Germany": 0.30,
    "Netherlands": 0.258,
    "Sweden": 0.206,
    "Denmark": 0.22,
    "Canada": 0.265,
    "Italy": 0.24,
    "Israel": 0.23,
    "Brazil": 0.34,
    "Norway": 0.22,
    "Japan": 0.305,
    "South Korea": 0.24,
    "Australia": 0.30,
    "Switzerland": 0.185,
}

SECTOR_GROWTH_DEFAULTS: dict[str, float] = {
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


def calculate_cost_of_equity(risk_free_rate: float, beta: Optional[float]) -> float:
    """Calculates Cost of Equity using CAPM."""
    # If beta is missing (None), assume market average of 1.0
    safe_beta = beta if isinstance(beta, (int, float)) else 1.0

    # CAPM Formula
    ke = risk_free_rate + (safe_beta * EQUITY_RISK_PREMIUM)

    # Sanity check: Cost of Equity rarely drops below 6% or exceeds 20% for healthy firms
    return _clamp(ke, 0.06, 0.20)


def estimate_wacc_dynamic(
    market_cap: Optional[float],
    total_debt: float,
    beta: Optional[float],
    risk_free_rate: float,
    country: str = "United States",
) -> float:
    """
    Calculates WACC using Cost of Equity (CAPM) and Cost of Debt.
    WACC = (E/V * Ke) + (D/V * Kd * (1 - Tax))
    """

    # 1. Calculate Cost of Equity (Ke)
    ke = calculate_cost_of_equity(risk_free_rate, beta)

    # If we can't determine capital structure, return Ke (assumes 100% equity financing)
    if not market_cap:
        return ke

    # 2. Determine Capital Structure weights
    enterprise_value = market_cap + total_debt
    weight_equity = market_cap / enterprise_value
    weight_debt = total_debt / enterprise_value

    # 3. Estimate Cost of Debt (Kd)
    # Heuristic: risk-free rate + credit spread (1.5% large cap, 3% small cap)
    credit_spread = 0.015 if market_cap > 10e9 else 0.03
    kd = risk_free_rate + credit_spread

    # 4. Country-specific corporate tax rate for after-tax cost of debt.
    # Defaults to 25% (conservative international average) for unknown countries.
    tax_rate = COUNTRY_TAX_RATES.get(country, 0.25)

    # 5. Final WACC Formula
    wacc = (weight_equity * ke) + (weight_debt * kd * (1 - tax_rate))

    # Clamp results to realistic bounds (5% to 15%)
    return _clamp(wacc, 0.05, 0.15)


def _safe_number(x: Any) -> Optional[float]:
    """Safely converts a value to float, returning None if invalid."""
    return (
        float(x)
        if isinstance(x, (int, float))
        and not (isinstance(x, float) and (x != x or x == float("inf") or x == float("-inf")))
        else None
    )


def _median(xs: list[float]) -> float:
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


def _extract_trailing_fcf(cashflow: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
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
    series: list[float] = []
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

    # 2) Require the most recent FCF to be positive — we can't project forward
    # from a negative base.  Earlier negative years are kept in the series so
    # that the median and CAGR reflect the full history (including bad years)
    # rather than silently inflating the baseline by dropping them.
    if not series or series[-1] <= 0:
        return None, None

    # 3) Calculate components for smoothing using the full series
    last_fcf = series[-1]
    median_last_3 = _median(series[-3:])  # may include negative years — honest
    # _calculate_cagr returns None if first value is non-positive (sign changes)
    fcf_cagr = _calculate_cagr(series[0], series[-1], len(series) - 1) if len(series) >= 2 else None

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


def _derive_shares_if_needed(info: dict[str, Any], candidate_shares: Optional[float]) -> Optional[float]:
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


def _estimate_dcf_inputs(instrument: Instrument, risk_free_rate: float) -> DcfInputs:
    """Estimates all necessary inputs for a DCF valuation from instrument data."""
    if instrument.yahoo is None:
        return {
            "current_fcf": None,
            "shares_outstanding": None,
            "total_cash": 0.0,
            "total_debt": 0.0,
            "wacc": 0.10,
            "initial_growth_rate": TERMINAL_GROWTH_RATE,
            "terminal_growth_rate": TERMINAL_GROWTH_RATE,
        }
    info = instrument.yahoo.info or {}
    cashflow = instrument.yahoo.cashflow or {}
    sector = info.get("sector") or ""
    country = info.get("country") or ""

    current_fcf, fcf_cagr = _extract_trailing_fcf(cashflow)

    raw_shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
    shares = _derive_shares_if_needed(info, _safe_number(raw_shares))

    total_cash: float = _safe_number(info.get("totalCash")) or 0.0
    total_debt: float = _safe_number(info.get("totalDebt")) or 0.0

    # Blend available growth signals for a more stable estimate.
    # revenueGrowth alone is trailing TTM and can mislead for cyclical sectors
    # (e.g. semiconductor downcycles, defense budget ramp-ups).
    #
    # earningsGrowth from Yahoo is a single-quarter YoY figure — it can be
    # extremely volatile (a one-off bad quarter looks like -80%).  We pre-clamp
    # each signal individually before blending so no single noisy reading
    # dominates the average.  The outer clamp below is the final safety net.
    _SIG_LO, _SIG_HI = -0.25, 0.35
    rev_g = _safe_number(info.get("revenueGrowth"))
    earn_g = _safe_number(info.get("earningsGrowth"))
    if rev_g is not None:
        rev_g = _clamp(rev_g, _SIG_LO, _SIG_HI)
    if earn_g is not None:
        earn_g = _clamp(earn_g, _SIG_LO, _SIG_HI)

    if fcf_cagr is not None:
        fcf_cagr = _clamp(fcf_cagr, _SIG_LO, _SIG_HI)
    candidates = [g for g in [rev_g, earn_g, fcf_cagr] if g is not None]
    if candidates:
        initial_growth = sum(candidates) / len(candidates)
    else:
        initial_growth = SECTOR_GROWTH_DEFAULTS.get(sector, 0.08)

    initial_growth = _clamp(initial_growth, -0.20, 0.30)

    market_cap = _safe_number(info.get("marketCap"))
    wacc = estimate_wacc_dynamic(market_cap, total_debt, info.get("beta"), risk_free_rate, country)

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
    instruments: list[Instrument],
    years: int = 10,
    wacc: Optional[float] = None,
    growth: Optional[float] = None,
    terminal: Optional[float] = None,
) -> list[Optional[float]]:
    """Calculates DCF prices for a list of instruments concurrently."""
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=False), raise_for_status=True
    ) as aiohttp_session:
        risk_free_rates = await get_risk_free_rates(aiohttp_session)
    return await asyncio.gather(
        *[_get_dcf_price(instrument, risk_free_rates, years, wacc, growth, terminal) for instrument in instruments]
    )


async def _get_dcf_price(
    instrument: Instrument,
    risk_free_rates: dict[str, float],
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
        risk_free_rates: Dict of {currency: rate} from get_risk_free_rates().
        years: The number of years for the high-growth stage.
        wacc_override, growth_override, terminal_override: Optional values to override estimates.
        allow_negative: If True, returns negative DCF values; otherwise returns None.

    Returns:
        The calculated intrinsic value per share, or None if inputs are invalid.
    """
    info = (instrument.yahoo.info or {}) if instrument.yahoo else {}

    # Financial stocks (banks, insurance): FCF has a completely different meaning
    # for companies whose business IS managing cash. Skip DCF; use P/BV or DDM instead.
    if info.get("sector") in SECTORS_WITHOUT_DCF:
        return None

    # Use the risk-free rate matching the instrument's reporting currency so that
    # EUR cash flows are discounted at EUR rates, GBP at GBP rates, etc.
    # Yahoo returns "GBp" or "GBX" for LSE pence-denominated stocks — both map to GBP.
    currency = info.get("currency") or "USD"
    if currency.upper() in ("GBX", "GBP"):
        currency = "GBP"
    risk_free_rate = risk_free_rates.get(currency, risk_free_rates.get("USD", 0.04))

    est = _estimate_dcf_inputs(instrument, risk_free_rate)

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
    projected_fcf: list[float] = []
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
