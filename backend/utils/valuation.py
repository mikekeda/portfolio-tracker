"""Currency-consistent valuation metrics shared by detail, holdings and features."""

from math import isfinite

from backend.utils.fcf import TrailingFcf

PENCE_CURRENCIES = frozenset({"GBp", "GBX"})

# Cached statements match USD issuer filings; Yahoo info remains in financialCurrency.
# Evidence and scope: docs/reviews/2026-10-12-valuation-currency-fixes.md.
STATEMENT_CURRENCY_OVERRIDES = {"PBR": "USD", "VALE": "USD"}


def number(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value):
        return float(value)
    return None


def statement_currency(symbol: str, info: dict) -> str | None:
    return STATEMENT_CURRENCY_OVERRIDES.get(symbol, info.get("financialCurrency"))


def market_cap_currency(info: dict) -> str | None:
    quote = info.get("currency")
    return "GBP" if quote in PENCE_CURRENCIES else quote


def currency_factor(source: str | None, target: str | None, rates: dict) -> float | None:
    if not source or not target:
        return None
    if source == target:
        return 1.0
    source_rate = 1.0 if source == "GBP" else number(rates.get(source))
    target_rate = 1.0 if target == "GBP" else number(rates.get(target))
    if source_rate is None or target_rate is None or source_rate <= 0 or target_rate <= 0:
        return None
    return source_rate / target_rate


def valuation_metrics(symbol: str, info: dict, rates: dict, fcf: TrailingFcf | None) -> dict:
    """Convert info and statement amounts independently to market-cap major units.

    Aggregate ratios use total market cap, not sharesOutstanding or ADR ratios.
    Missing debt/cash is unknown, never a zero balance.
    """
    currency = market_cap_currency(info)
    info_currency = info.get("financialCurrency")
    cf_currency = statement_currency(symbol, info)
    info_fx = currency_factor(info_currency, currency, rates)
    cf_fx = currency_factor(cf_currency, currency, rates)
    issues = {}

    def convert(value, factor):
        value = number(value)
        return value * factor if value is not None and factor is not None else None

    cap = number(info.get("marketCap"))
    if cap is None or cap <= 0 or not currency:
        cap = None
    raw_debt = number(info.get("totalDebt"))
    raw_cash = number(info.get("totalCash"))
    if raw_debt is not None and raw_debt < 0:
        raw_debt = None
    if raw_cash is not None and raw_cash < 0:
        raw_cash = None
    debt = convert(raw_debt, info_fx)
    cash = convert(raw_cash, info_fx)
    revenue = convert(info.get("totalRevenue"), info_fx)
    ebitda = convert(info.get("ebitda"), info_fx)
    free_cashflow = convert(fcf.fcf, cf_fx) if fcf else None
    net_debt = debt - cash if debt is not None and cash is not None else None
    ev = cap + net_debt if cap is not None and net_debt is not None else None
    raw_net_debt = raw_debt - raw_cash if info_currency and raw_debt is not None and raw_cash is not None else None

    def ratio(numerator, denominator):
        return numerator / denominator if numerator is not None and denominator is not None and denominator > 0 else None

    result = {
        "marketCap": cap,
        "totalDebt": debt,
        "totalCash": cash,
        "totalRevenue": revenue,
        "ebitda": ebitda,
        "freeCashflow": free_cashflow,
        "fcfYield": ratio(free_cashflow, cap),
        "enterpriseValue": ev,
        "enterpriseToEbitda": ratio(ev, ebitda),
        "enterpriseToRevenue": ratio(ev, revenue),
        "priceToSalesTtm": ratio(cap, revenue),
        "priceToBook": None,
        "netDebt": net_debt,
        "netDebtToEbitda": ratio(raw_net_debt, number(info.get("ebitda"))),
        "netCash": raw_net_debt < 0 if raw_net_debt is not None else None,
        "valuationCurrency": currency,
        "statementCurrency": cf_currency,
    }
    if result["fcfYield"] is not None:
        result["fcfYield"] *= 100
    for key in ("totalDebt", "totalCash", "enterpriseValue", "enterpriseToEbitda", "enterpriseToRevenue", "priceToSalesTtm",
                "netDebt", "netDebtToEbitda", "fcfYield"):
        if result[key] is None:
            if key == "netDebtToEbitda":
                issues[key] = (
                    "Unavailable: summary currency is missing."
                    if not info_currency else "Unavailable: debt, cash or positive EBITDA is missing."
                )
                continue
            factor = cf_fx if key == "fcfYield" else info_fx
            issues[key] = (
                "Unavailable: currency or exchange rate is missing."
                if factor is None else "Unavailable: required financial amounts are missing or the denominator is not positive."
            )
    result["valuationIssues"] = issues
    return result
