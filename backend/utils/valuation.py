import asyncio
from typing import Dict, List, Optional, TypedDict, Any

from models import Instrument


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
    return max(lo, min(hi, x))


def estimate_wacc(market_cap: Optional[float], operating_margin: Optional[float]) -> float:
    if not market_cap:
        return 0.10
    if market_cap > 2e11:  # mega cap
        base = 0.075  # ↑ from 0.07
    elif market_cap > 2e10:
        base = 0.09
    elif market_cap > 2e9:
        base = 0.11
    else:
        base = 0.13

    if isinstance(operating_margin, (int, float)):
        if operating_margin < 0.05:
            base += 0.02
        elif operating_margin > 0.20:
            base -= 0.01

    return _clamp(base, 0.07, 0.20)


def _safe_number(x: Any) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) else None


def _median(xs: List[float]) -> float:
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    if n == 0:
        raise ValueError("empty")
    m = n // 2
    return xs_sorted[m] if n % 2 == 1 else 0.5 * (xs_sorted[m - 1] + xs_sorted[m])


def _extract_trailing_fcf(cashflow: Dict[str, Any]) -> Optional[float]:
    """
    Forward-looking FCF smoothing from Yahoo cashflow dict.

    Logic:
      1) Build chronological FCF series (prefer "Free Cash Flow"; else OCF + CapEx).
      2) Keep only positive values (DCF base must be > 0).
      3) Compute:
         - last  = most recent positive FCF
         - med3  = median of the last up to 3 positive FCFs
         - next  = one-step forecast using CAGR across positive history
      4) Blend with dynamic weights (renormalized if any part missing):
         40% last + 40% med3 + 20% next
      5) Final clamp to avoid pathological jumps vs `last`.

    Returns None when there are no usable positive FCF observations.
    """

    def _cagr(first: float, last: float, periods: int) -> Optional[float]:
        if periods <= 0 or first <= 0 or last <= 0:
            return None
        try:
            return (last / first) ** (1.0 / periods) - 1.0
        except Exception:
            return None

    if not isinstance(cashflow, dict) or not cashflow:
        return None

    # 1) Chronological FCF series (CapEx is negative in Yahoo -> OCF + CapEx)
    years = sorted(cashflow.keys())  # e.g. ["2021-06-30", ...]
    series: List[float] = []
    for y in years:
        row = cashflow.get(y) or {}
        fcf = _safe_number(row.get("Free Cash Flow"))
        if fcf is None:
            ocf = _safe_number(row.get("Operating Cash Flow"))
            capex = _safe_number(row.get("Capital Expenditure"))
            if ocf is not None and capex is not None:
                fcf = ocf + capex  # OCF - abs(CapEx)
        if fcf is not None:
            series.append(fcf)

    # 2) Keep only positive observations for DCF base
    pos = [v for v in series if v > 0]
    if not pos:
        return None

    # 3) Components
    last = pos[-1]
    tail = pos[-3:]
    med3 = _median(tail)

    g = _cagr(pos[0], pos[-1], len(pos) - 1) if len(pos) >= 2 else None
    if g is not None:
        # Avoid runaway due to noisy endpoints
        g = _clamp(g, -0.30, 0.40)
        nxt = last * (1.0 + g)
    else:
        nxt = None

    # 4) Blend with dynamic weights (renormalize if missing parts)
    parts: List[tuple[float, float]] = []
    parts.append((0.40, last))
    parts.append((0.40, med3))
    if nxt is not None:
        parts.append((0.20, nxt))

    total_w = sum(w for w, _ in parts)
    if total_w <= 0:
        return None

    smoothed = sum((w / total_w) * v for w, v in parts)

    # 5) Final sanity clamp around `last` to avoid big jumps
    band_lo = last * 0.5
    band_hi = last * 1.8
    return _clamp(smoothed, band_lo, band_hi)


def _derive_shares_if_needed(info: Dict[str, Any], candidate_shares: Optional[float]) -> Optional[float]:
    """Use marketCap/currentPrice to sanity-check shares (important for ADRs)."""
    if not isinstance(candidate_shares, (int, float)) or candidate_shares <= 0:
        candidate_shares = None

    mcap = _safe_number(info.get("marketCap"))
    price = _safe_number(info.get("currentPrice"))
    if mcap is None or price is None or price <= 0:
        return candidate_shares

    derived = mcap / price
    if candidate_shares is None:
        return derived

    # If discrepancy is very large (>50%), prefer derived
    diff = abs(candidate_shares - derived) / max(derived, 1.0)
    return derived if diff > 0.5 else candidate_shares


def _estimate_dcf_inputs(instrument: Instrument) -> DcfInputs:
    """Estimate DCF inputs from Yahoo finance info and cashflow data, robustly."""
    info = instrument.yahoo.info or {}
    cashflow = instrument.yahoo.cashflow or {}
    sector = instrument.sector

    # --- Free Cash Flow (robust trailing) ---
    current_fcf = _extract_trailing_fcf(cashflow)

    # --- Shares, cash, debt ---
    raw_shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
    shares = _derive_shares_if_needed(info, _safe_number(raw_shares))

    total_cash = _safe_number(info.get("totalCash")) or 0.0
    total_debt = _safe_number(info.get("totalDebt")) or 0.0

    # --- Growth rate ---
    rev_g = _safe_number(info.get("revenueGrowth"))
    if rev_g is not None:
        initial_growth = rev_g
    elif sector in SECTOR_GROWTH_DEFAULTS:
        initial_growth = SECTOR_GROWTH_DEFAULTS[sector]
    else:
        initial_growth = 0.08

    # Clamp growths to sane ranges
    initial_growth = _clamp(initial_growth, -0.50, 0.50)
    terminal_growth = _clamp(SECTOR_GROWTH_DEFAULTS.get(sector, 0.03) / 3.0, 0.01, 0.04)

    # --- WACC ---
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
        "terminal_growth_rate": terminal_growth,
    }


async def get_dcf_prices(
    instruments: List[Instrument],
    years: int = 10,
    wacc: float | None = None,
    growth: float | None = None,
    terminal: float | None = None,
) -> List[Optional[float]]:
    return await asyncio.gather(
        *[_get_dcf_price(instrument, years, wacc, growth, terminal) for instrument in instruments]
    )


async def _get_dcf_price(
    instrument: Instrument,
    years: int = 10,
    wacc: float | None = None,
    growth: float | None = None,
    terminal: float | None = None,
    allow_negative: bool = False,  # set True if you want to surface distressed DCFs
) -> Optional[float]:
    """Return DCF-derived intrinsic value per share for a ticker."""
    est = _estimate_dcf_inputs(instrument)

    # Apply overrides
    if wacc is not None:
        est["wacc"] = wacc
    if growth is not None:
        est["initial_growth_rate"] = growth
    if terminal is not None:
        est["terminal_growth_rate"] = terminal

    # Validate minimum inputs (reject non-economic cases)
    if est["current_fcf"] is None or est["current_fcf"] <= 0:
        return None
    if est["shares_outstanding"] is None or est["shares_outstanding"] <= 0:
        return None

    # DCF projection (mid-year convention)
    horizon_years = max(1, min(30, years))
    g0 = _clamp(est["initial_growth_rate"], -0.5, 0.5)
    gT = _clamp(est["terminal_growth_rate"], 0.0, 0.04)
    r = _clamp(est["wacc"], 0.05, 0.20)
    min_spread = 0.03
    if r <= gT + min_spread:
        r = gT + min_spread

    g_step = (gT - g0) / max(1, horizon_years - 1)
    g_list = [g0 + g_step * i for i in range(horizon_years)]

    f = est["current_fcf"]
    fcff: List[float] = []
    for g in g_list:
        f = f * (1.0 + g)
        fcff.append(f)

    inv_1pr = 1.0 / (1.0 + r)
    pv_fcff = sum(f_t * (inv_1pr ** (t - 0.5)) for t, f_t in enumerate(fcff, start=1))

    terminal_fcf = fcff[-1] * (1.0 + gT)
    terminal_value = terminal_fcf / (r - gT)
    pv_terminal = terminal_value * (inv_1pr ** (horizon_years - 0.5))

    terminal_share = pv_terminal / (pv_fcff + pv_terminal) if (pv_fcff + pv_terminal) > 0 else 1.0
    if terminal_share > 0.85:
        gT_adj = max(0.0, gT - 0.005)  # shave 50 bps
        terminal_fcf = fcff[-1] * (1.0 + gT_adj)
        terminal_value = terminal_fcf / max(1e-9, (r - gT_adj))
        pv_terminal = terminal_value * (inv_1pr ** (horizon_years - 0.5))

    enterprise_value = pv_fcff + pv_terminal
    equity_value = enterprise_value + (est["total_cash"] or 0.0) - (est["total_debt"] or 0.0)

    per_share = equity_value / est["shares_outstanding"]

    # Hide negative/absurd per-share unless explicitly allowed
    if not allow_negative and per_share <= 0:
        return None

    return per_share
