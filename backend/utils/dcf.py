from datetime import datetime, timedelta
from typing import Any, Optional, TypedDict

import aiohttp
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import EQUITY_RISK_PREMIUM, RISK_FREE_RATE, SPY, TERMINAL_GROWTH_RATE, TIMEZONE
from models import Instrument, PricesDaily
from utils.market_data import get_risk_free_rates
from backend.utils.technical import PRICE_COLUMN, calculate_annualized_volatility

# --- Constants ---
# Plateau + fade projection: the first N years run at the initial growth rate,
# then fade linearly to the terminal rate over the remaining horizon. This is
# more honest than a straight linear fade from year 1 for quality compounders
# (MSFT, ASML etc. that historically held 15%+ top-line growth for 10–15 years).
PLATEAU_YEARS_DEFAULT = 3

# Sensitivity band: ±5pp initial growth, ±1pp WACC (inversely correlated for
# the "optimistic" and "pessimistic" bounds respectively).
SENSITIVITY_GROWTH_PP = 0.05
SENSITIVITY_WACC_PP = 0.01

# Reverse DCF bisection bounds. We search for the initial growth rate that
# makes intrinsic value equal current market price.
IMPLIED_GROWTH_LO = -0.10
IMPLIED_GROWTH_HI = 0.50
IMPLIED_GROWTH_MAX_ITERATIONS = 60
IMPLIED_GROWTH_TOLERANCE = 1e-4

# Sectors where FCF-based DCF is meaningless. Banks' "free cash flow" is not
# comparable to an industrial company's; use P/BV or DDM for these instead.
SECTORS_WITHOUT_DCF: frozenset[str] = frozenset({"Financial Services"})

# Effective beta for CAPM.
# Reported Yahoo beta is the regression slope against the S&P 500. For non-US
# stocks that move out of sync with the S&P, beta can be artificially low and
# produce absurdly low WACC (e.g. Saab at beta≈0.3 with true annualised vol of
# ~45%). We override such cases with a volatility-ratio proxy so the discount
# rate reflects actual systematic risk rather than correlation to the wrong
# benchmark.
EFFECTIVE_BETA_LO = 0.3  # floor — prevents non-US stocks ducking under realistic risk
EFFECTIVE_BETA_HI = 2.5  # ceiling — prevents meme spikes dominating WACC
REPORTED_BETA_TRUST_THRESHOLD = 0.3  # reported betas below this are treated as unreliable
# 420 calendar days ≈ 1 trading year plus buffer for holidays/missing days.
EFFECTIVE_BETA_LOOKBACK_DAYS = 420

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


class DcfAnalysis(TypedDict):
    """DCF output: point estimate, sensitivity band, and reverse-DCF implied
    growth. All fields are None when DCF can't be computed (unsupported
    sector, no positive trailing FCF, missing shares). The frontend treats
    a DCF as "low confidence" when implied_growth is None (market pricing
    outside [-10%, +50%] g0) or the high/low band spans more than ~4x."""

    price: Optional[float]  # point estimate intrinsic value per share
    low: Optional[float]  # pessimistic (higher WACC, lower g0)
    high: Optional[float]  # optimistic (lower WACC, higher g0)
    implied_growth: Optional[float]  # g0 that solves price == current market price


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


def _estimate_dcf_inputs(
    instrument: Instrument,
    risk_free_rate: float,
    effective_beta_override: Optional[float] = None,
) -> DcfInputs:
    """Estimates all necessary inputs for a DCF valuation from instrument data.

    `effective_beta_override`, when supplied, replaces the reported Yahoo beta
    for CAPM/WACC purposes. Used to correct for non-US stocks whose reported
    beta (regressed against the S&P 500) understates systematic risk.
    """
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
    beta_for_wacc = effective_beta_override if effective_beta_override is not None else info.get("beta")
    wacc = estimate_wacc_dynamic(market_cap, total_debt, beta_for_wacc, risk_free_rate, country)

    return {
        "current_fcf": current_fcf,
        "shares_outstanding": shares,
        "total_cash": total_cash,
        "total_debt": total_debt,
        "wacc": wacc,
        "initial_growth_rate": initial_growth,
        "terminal_growth_rate": TERMINAL_GROWTH_RATE,
    }


def _project_cashflows(
    base_fcf: float,
    g0: float,
    gT: float,
    horizon_years: int,
    plateau_years: int = PLATEAU_YEARS_DEFAULT,
) -> list[float]:
    """Two-stage projection: flat at g0 for plateau_years, then linear fade to gT.

    For a 10-year horizon with plateau_years=3, g0=25%, gT=2.5%:
      years 1–3: 25%  (plateau)
      years 4–10: 21.8%, 18.6%, 15.3%, 12.1%, 8.9%, 5.6%, 2.5%  (linear fade)
    """
    plateau = max(0, min(plateau_years, horizon_years - 1))
    fade = horizon_years - plateau
    g_list = [g0] * plateau
    if fade > 0:
        g_list.extend(g0 + (gT - g0) * (i + 1) / fade for i in range(fade))

    fcf = base_fcf
    projected: list[float] = []
    for g in g_list:
        fcf *= 1.0 + g
        projected.append(fcf)
    return projected


def _compute_equity_value_per_share(
    base_fcf: float,
    shares: float,
    total_cash: float,
    total_debt: float,
    wacc: float,
    g0: float,
    gT: float,
    horizon_years: int,
) -> float:
    """Pure DCF math: given all inputs, return per-share intrinsic value.

    Enforces WACC >= gT + 3pp spread so the terminal-value denominator stays
    positive and reasonable.
    """
    r = _clamp(wacc, 0.05, 0.20)
    min_spread = 0.03
    if r <= gT + min_spread:
        r = gT + min_spread

    projected = _project_cashflows(base_fcf, g0, gT, horizon_years)

    inv_1pr = 1.0 / (1.0 + r)
    pv_fcf = sum(f_t * (inv_1pr ** (t + 1)) for t, f_t in enumerate(projected))

    terminal_fcf = projected[-1] * (1.0 + gT)
    terminal_value = terminal_fcf / (r - gT)
    pv_terminal = terminal_value * (inv_1pr**horizon_years)

    enterprise_value = pv_fcf + pv_terminal
    equity_value = enterprise_value + total_cash - total_debt
    return equity_value / shares


def _solve_implied_growth(
    est: DcfInputs,
    wacc: float,
    gT: float,
    horizon_years: int,
    target_price: float,
    lo: float = IMPLIED_GROWTH_LO,
    hi: float = IMPLIED_GROWTH_HI,
) -> Optional[float]:
    """Reverse DCF: bisect on g0 to find the growth rate that produces target_price.

    Returns the implied initial growth rate, or None if target_price is outside
    the achievable range [price@lo, price@hi] — meaning the market is pricing
    either less than a -10% growth scenario or more than a +50% one.
    """
    if (
        target_price <= 0
        or est["current_fcf"] is None
        or est["current_fcf"] <= 0
        or not est["shares_outstanding"]
        or est["shares_outstanding"] <= 0
    ):
        return None

    def price_at(g: float) -> float:
        return _compute_equity_value_per_share(
            est["current_fcf"],
            est["shares_outstanding"],
            est["total_cash"],
            est["total_debt"],
            wacc,
            g,
            gT,
            horizon_years,
        )

    p_lo = price_at(lo)
    p_hi = price_at(hi)
    # price is monotonically increasing in g0 within the search band
    if target_price < min(p_lo, p_hi) or target_price > max(p_lo, p_hi):
        return None

    for _ in range(IMPLIED_GROWTH_MAX_ITERATIONS):
        mid = 0.5 * (lo + hi)
        p_mid = price_at(mid)
        if abs(p_mid - target_price) / target_price < IMPLIED_GROWTH_TOLERANCE:
            return mid
        if p_mid < target_price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _empty_analysis() -> DcfAnalysis:
    return {"price": None, "low": None, "high": None, "implied_growth": None}


def _compute_dcf_full(
    instrument: Instrument,
    risk_free_rates: dict[str, float],
    years: int = 10,
    wacc_override: Optional[float] = None,
    growth_override: Optional[float] = None,
    terminal_override: Optional[float] = None,
    effective_beta_override: Optional[float] = None,
) -> DcfAnalysis:
    """Compute DCF analysis (point estimate + sensitivity band + reverse-DCF
    implied growth) for a single instrument.
    """
    info = (instrument.yahoo.info or {}) if instrument.yahoo else {}

    # Financial stocks (banks, insurance): FCF has a completely different meaning
    # for companies whose business IS managing cash. Use P/BV or DDM instead.
    if info.get("sector") in SECTORS_WITHOUT_DCF:
        return _empty_analysis()

    # Risk-free rate matched to reporting currency so EUR cash flows discount at
    # EUR rates, GBP at GBP, etc. GBp / GBX both map to GBP.
    currency = info.get("currency") or "USD"
    if currency.upper() in ("GBX", "GBP"):
        currency = "GBP"
    risk_free_rate = risk_free_rates.get(currency, risk_free_rates.get("USD", RISK_FREE_RATE))

    est = _estimate_dcf_inputs(instrument, risk_free_rate, effective_beta_override=effective_beta_override)

    if est["current_fcf"] is None or est["current_fcf"] <= 0:
        return _empty_analysis()
    if not est["shares_outstanding"] or est["shares_outstanding"] <= 0:
        return _empty_analysis()

    wacc = wacc_override if wacc_override is not None else est["wacc"]
    g0 = growth_override if growth_override is not None else est["initial_growth_rate"]
    gT = terminal_override if terminal_override is not None else est["terminal_growth_rate"]
    horizon = max(1, min(30, years))

    def price_at(wacc_v: float, g0_v: float) -> float:
        return _compute_equity_value_per_share(
            est["current_fcf"],
            est["shares_outstanding"],
            est["total_cash"],
            est["total_debt"],
            wacc_v,
            g0_v,
            gT,
            horizon,
        )

    base = price_at(wacc, g0)
    # Pessimistic: higher discount rate + slower growth
    low_wacc = min(0.20, wacc + SENSITIVITY_WACC_PP)
    low_g = max(-0.30, g0 - SENSITIVITY_GROWTH_PP)
    low = price_at(low_wacc, low_g)
    # Optimistic: lower discount rate + faster growth (respects the same +30% cap
    # applied to the base estimate so sensitivity doesn't exceed validated bounds)
    high_wacc = max(0.05, wacc - SENSITIVITY_WACC_PP)
    high_g = min(0.30, g0 + SENSITIVITY_GROWTH_PP)
    high = price_at(high_wacc, high_g)

    current_price = _safe_number(info.get("currentPrice"))
    implied = (
        _solve_implied_growth(est, wacc, gT, horizon, current_price)
        if current_price is not None and current_price > 0
        else None
    )

    def pos_or_none(p: float) -> Optional[float]:
        return p if p > 0 else None

    return {
        "price": pos_or_none(base),
        "low": pos_or_none(low),
        "high": pos_or_none(high),
        "implied_growth": implied,
    }


async def get_dcf_analyses(
    instruments: list[Instrument],
    years: int = 10,
    wacc: Optional[float] = None,
    growth: Optional[float] = None,
    terminal: Optional[float] = None,
    effective_betas: Optional[dict[str, float]] = None,
) -> list[DcfAnalysis]:
    """Calculate DCF analyses (point estimate + sensitivity band + implied
    growth) for a list of instruments.

    Pass `effective_betas` (keyed by `yahoo_symbol`) to override reported Yahoo
    betas for non-US or low-beta stocks. Build the dict with
    :func:`get_effective_betas`.
    """
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=False), raise_for_status=True
    ) as aiohttp_session:
        risk_free_rates = await get_risk_free_rates(aiohttp_session)
    betas = effective_betas or {}
    return [
        _compute_dcf_full(
            instrument,
            risk_free_rates,
            years,
            wacc,
            growth,
            terminal,
            effective_beta_override=betas.get(instrument.yahoo_symbol) if instrument.yahoo_symbol else None,
        )
        for instrument in instruments
    ]


async def get_effective_betas(
    instruments: list[Instrument],
    session: AsyncSession,
    lookback_days: int = EFFECTIVE_BETA_LOOKBACK_DAYS,
) -> dict[str, float]:
    """Compute effective beta overrides, keyed by yahoo_symbol.

    Only symbols needing an override appear in the result. A symbol is
    overridden when either:
      - the stock is non-US (reported beta's S&P 500 regression is the wrong
        benchmark), or
      - the reported beta is < 0.3 (implausibly low for any listed equity).

    In those cases the override is the realised 1-year volatility ratio
    (stock σ / VUAG.L σ), floored by the reported beta when available (so we
    never *reduce* the risk reading below what Yahoo reports). Every override
    is clamped to [0.3, 2.5].

    US stocks with a healthy reported beta are omitted; DCF falls back to the
    Yahoo value in that case.
    """
    if not instruments:
        return {}

    # Decide which instruments need the override using only Yahoo info (no
    # price data required). This lets us skip the PricesDaily query entirely
    # when the whole portfolio is US stocks with healthy reported betas —
    # which is the common case for this app.
    eligible: list[tuple[Instrument, Optional[float]]] = []
    for inst in instruments:
        if not inst.yahoo_symbol:
            continue
        info = (inst.yahoo.info or {}) if inst.yahoo else {}
        reported = _safe_number(info.get("beta"))
        country = (info.get("country") or "").strip()
        is_non_us = bool(country) and country != "United States"
        if is_non_us or (reported is not None and reported < REPORTED_BETA_TRUST_THRESHOLD):
            eligible.append((inst, reported))

    if not eligible:
        return {}

    query_symbols = list({*(inst.yahoo_symbol for inst, _ in eligible), SPY})
    cutoff = datetime.now(TIMEZONE).date() - timedelta(days=lookback_days)

    price_rows = (
        await session.execute(
            select(PricesDaily.symbol, PRICE_COLUMN)
            .filter(PricesDaily.symbol.in_(query_symbols), PricesDaily.date >= cutoff)
            .order_by(PricesDaily.date)
        )
    ).all()

    price_history: dict[str, list[float]] = {}
    for row in price_rows:
        price_history.setdefault(row.symbol, []).append(row.price)

    reference_vol = calculate_annualized_volatility(price_history.get(SPY, []))

    result: dict[str, float] = {}
    for inst, reported in eligible:
        stock_vol = calculate_annualized_volatility(price_history.get(inst.yahoo_symbol, []))
        vol_ratio = (stock_vol / reference_vol) if (stock_vol and reference_vol) else None
        if vol_ratio is None:
            continue  # no usable volatility — leave DCF to use the reported beta

        # Prefer the higher of vol ratio and reported beta: we triggered the
        # override because we suspect the reported figure, so we never want to
        # replace it with something even lower.
        effective = max(vol_ratio, reported) if reported is not None else vol_ratio
        result[inst.yahoo_symbol] = _clamp(effective, EFFECTIVE_BETA_LO, EFFECTIVE_BETA_HI)

    return result
