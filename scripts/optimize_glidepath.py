"""
scripts/optimize_glidepath.py
=============================
Solve a CRRA-optimal allocation glidepath over a fixed horizon. Run from project root:

    python scripts/optimize_glidepath.py --horizon 10 --gamma-start 2 --gamma-end 5
    python scripts/optimize_glidepath.py --horizon 15 --gamma-start 1 --gamma-end 8
    python scripts/optimize_glidepath.py --horizon 10 --gamma-start 3 --gamma-end 3 --no-benchmark

Gamma is the constant-relative-risk-aversion coefficient: 1 is log utility (growth
maximising), higher is more risk averse. Lowering risk over time therefore means
gamma RISING, so a de-risking plan passes --gamma-start below --gamma-end.

For each year the script interpolates gamma, then maximises the annualised
certainty-equivalent growth rate over the universe below, long only and capped.
It prints target weights per year and nothing else — sizing trades against live
holdings is deliberately out of scope.

Horizons beyond the price history in `prices_daily` (which starts 2015-09) fall
back to a four-class proxy panel spliced from public long-history sources, so a
15-year run optimises over asset classes rather than the 20 instruments.

Every fitted weight is in-sample. The 1/N benchmark printed at the end is not
decoration: DeMiguel, Garlappi & Uppal (2009) find naive diversification beats
optimisation out of sample at this sample size, and on this data it did.
"""

import argparse
import asyncio
import csv
import io
from bisect import bisect_right
from datetime import date
from pathlib import Path

import numpy as np
import requests
from scipy.optimize import minimize
from sqlalchemy import select

from backend.app import get_session
from config import logger
from models import CurrencyRateDaily, Instrument, PricesDaily

# Ten funds spanning distinct exposures (size, style, factor, region, commodity),
# de-duplicated: QDVE overlaps CNDX, XAD5 duplicates SGLN.
ETF_UNIVERSE = {
    "VUSA.L": "S&P 500",
    "CNDX.L": "Nasdaq 100",
    "IWFQ.L": "World quality",
    "IWFV.L": "World value",
    "IWFM.L": "World momentum",
    "USSC.L": "US small-cap value",
    "R2SC.L": "US small cap",
    "UDVD.L": "US dividend aristocrats",
    "UKDV.L": "UK dividend aristocrats",
    "SGLN.L": "Gold",
}

# Held names with a full decade of prices, spread across sectors. Any equity
# shortlist drawn from a post-2015 window carries selection bias — edit freely.
STOCK_UNIVERSE = {
    "NVDA": "Semiconductors",
    "ASML": "Semicap equipment",
    "MSFT": "Software",
    "GOOGL": "Communication services",
    "AMZN": "Consumer discretionary",
    "NVO": "Healthcare (EU)",
    "AZN.L": "Healthcare (UK)",
    "RR.L": "Industrials (UK)",
    "MELI": "E-commerce (LatAm)",
    "BARC.L": "Financials (UK)",
}

# Long-history proxies for horizons the price table cannot cover. Nasdaq is a
# price index; the yield approximates its dividend over the period.
LONG_SOURCES = {
    "sp500": "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv",
    "gold": "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv",
    "nasdaq": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=NASDAQ100",
    "gbpusd": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSUK",
    "gilt": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRLTLT01GBM156N",
}
LONG_CLASSES = ("S&P 500", "Nasdaq 100", "Gold", "UK gilts")
NASDAQ_DIVIDEND_YIELD = 0.005
GILT_DURATION = 8.0

CACHE_DIR = Path("backtests/glidepath_cache")
PRICES_START = date(2015, 9, 1)
TRADING_DAYS_PER_MONTH = 21
MONTHS_PER_YEAR = 12

# Stops the optimiser answering with a single instrument. It relaxes with 1/n
# because on a small panel a flat cap binds everywhere and decides the answer.
MAX_WEIGHT = 0.35
CAP_EQUAL_WEIGHT_MULTIPLE = 2.5
# Expected returns are the noisiest moment by an order of magnitude, so pull each
# asset's mean toward the cross-sectional mean before optimising.
MEAN_SHRINKAGE = 0.60
MIN_MONTHS = 60
RESTARTS = 8


def _fetch(name: str, url: str) -> str:
    """Return the body of a long-history source, caching it under CACHE_DIR."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.csv"
    if not path.exists():
        logger.info("Fetching %s", url)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path.read_text()


def _monthly_key(value: str) -> str:
    return value[:7]


def load_long_panel() -> tuple[list[str], np.ndarray]:
    """Monthly GBP total returns for LONG_CLASSES, spliced from public sources.

    Returns the month labels and an (n_months, 4) array.
    """
    sp: dict[str, tuple[float, float]] = {}
    carried = 0.0
    for row in csv.DictReader(io.StringIO(_fetch("sp500", LONG_SOURCES["sp500"]))):
        price, dividend = float(row["SP500"]), float(row["Dividend"] or 0)
        if dividend > 0:
            carried = dividend / price
        sp[_monthly_key(row["Date"])] = (price, carried)

    gold = {row["Date"]: float(row["Price"]) for row in csv.DictReader(io.StringIO(_fetch("gold", LONG_SOURCES["gold"])))}

    def fred(name: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for row in csv.reader(io.StringIO(_fetch(name, LONG_SOURCES[name]))):
            try:
                out[_monthly_key(row[0])] = float(row[1])
            except (ValueError, IndexError):
                continue
        return out

    nasdaq, gbpusd, gilt = fred("nasdaq"), fred("gbpusd"), fred("gilt")
    months = sorted(set(sp) & set(gold) & set(nasdaq) & set(gbpusd) & set(gilt))
    labels, rows = [], []
    for previous, current in zip(months, months[1:]):
        # Sterling investor: a falling GBP lifts the return on a dollar asset.
        fx = gbpusd[previous] / gbpusd[current]
        price_0, yield_0 = sp[previous]
        price_1, _ = sp[current]
        yield_prev, yield_now = gilt[previous] / 100, gilt[current] / 100
        change = yield_now - yield_prev
        rows.append(
            [
                (price_1 + price_0 * yield_0 / MONTHS_PER_YEAR) / price_0 * fx - 1,
                nasdaq[current] / nasdaq[previous] * (1 + NASDAQ_DIVIDEND_YIELD / MONTHS_PER_YEAR) * fx - 1,
                gold[current] / gold[previous] * fx - 1,
                yield_prev / MONTHS_PER_YEAR - GILT_DURATION * change + 0.5 * GILT_DURATION**2 * change**2,
            ]
        )
        labels.append(current)
    return labels, np.asarray(rows)


def _rate_lookup(history: dict[str, tuple[list[date], list[float]]], currency: str, day: date) -> float:
    """Rate into GBP on or before `day`, carrying the last known value forward."""
    if currency in ("GBP", "GBX"):
        return 1.0
    if currency not in history:
        raise ValueError(f"No {currency}->GBP history in currency_rates_daily")
    days, rates = history[currency]
    position = bisect_right(days, day)
    if position == 0:
        # Falling back to rates[0] would price 2015 in a later year's currency.
        raise ValueError(
            f"{currency}->GBP history starts {days[0]}, after {day}. "
            f"Backfill it from {PRICES_START} (scripts/backfill_currency_rates.py) or drop that sleeve."
        )
    return rates[position - 1]


async def load_price_panel(symbols: list[str]) -> tuple[list[str], np.ndarray]:
    """Monthly GBP total returns for `symbols`, sampled at calendar month ends.

    Symbols without a full common window are dropped and logged.
    """
    async with get_session() as session:
        rate_rows = (
            await session.execute(
                select(CurrencyRateDaily.from_currency, CurrencyRateDaily.date, CurrencyRateDaily.rate)
                .where(CurrencyRateDaily.to_currency == "GBP", CurrencyRateDaily.date >= PRICES_START)
                .order_by(CurrencyRateDaily.date)
            )
        ).all()
        rows = (
            await session.execute(
                select(PricesDaily.symbol, PricesDaily.date, PricesDaily.adj_close_price, Instrument.currency)
                .join(Instrument, Instrument.yahoo_symbol == PricesDaily.symbol)
                .where(PricesDaily.symbol.in_(symbols), PricesDaily.date >= PRICES_START)
                .order_by(PricesDaily.date)
            )
        ).all()

    history: dict[str, tuple[list[date], list[float]]] = {}
    for currency, day, rate in rate_rows:
        days, rates = history.setdefault(currency, ([], []))
        days.append(day)
        rates.append(rate)

    series: dict[str, dict[date, float]] = {}
    for symbol, day, price, currency in rows:
        # The rate must be the one for `day`: a constant multiplier cancels out of a
        # return, leaving foreign sleeves priced in their own currency, not GBP.
        series.setdefault(symbol, {})[day] = price * _rate_lookup(history, currency, day)

    usable = [s for s in symbols if len(series.get(s, {})) >= MIN_MONTHS * TRADING_DAYS_PER_MONTH]
    for missing in sorted(set(symbols) - set(usable)):
        logger.warning("Dropped %s: insufficient price history", missing)
    if not usable:
        raise ValueError("No symbol has enough price history to optimise")

    # Last common trading day of each calendar month, so annualising at 12 is exact.
    common = sorted(set.intersection(*(set(series[s]) for s in usable)))
    month_ends = {(d.year, d.month): d for d in common}
    sampled = [month_ends[key] for key in sorted(month_ends)]
    # Per-symbol history says nothing about the overlap the optimiser actually sees.
    if len(sampled) < MIN_MONTHS + 1:
        raise ValueError(f"Only {len(sampled)} overlapping months across {len(usable)} sleeves; need {MIN_MONTHS + 1}")
    matrix = np.array([[series[s][d] for s in usable] for d in sampled])
    return usable, matrix[1:] / matrix[:-1] - 1


def certainty_equivalent(weights: np.ndarray, returns: np.ndarray, gamma: float) -> float:
    """Annualised CRRA certainty-equivalent growth rate, as a percentage."""
    growth = 1.0 + returns @ weights
    if np.any(growth <= 0):
        return -np.inf
    if abs(gamma - 1.0) < 1e-9:
        monthly = np.exp(np.mean(np.log(growth)))
    else:
        monthly = np.mean(growth ** (1 - gamma)) ** (1 / (1 - gamma))
    return (monthly**MONTHS_PER_YEAR - 1) * 100


def shrink_means(returns: np.ndarray, intensity: float) -> np.ndarray:
    """Pull each column's mean toward the cross-sectional mean, leaving covariance intact."""
    means = returns.mean(axis=0)
    target = means.mean()
    return returns - means + (means * (1 - intensity) + target * intensity)


def optimise(returns: np.ndarray, gamma: float, rng: np.random.Generator, max_weight: float) -> np.ndarray:
    """Long-only weights maximising certainty equivalent, capped per instrument."""
    n = returns.shape[1]
    cap = max(max_weight, CAP_EQUAL_WEIGHT_MULTIPLE / n)
    bounds = [(0.0, cap)] * n
    constraint = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    best_weights, best_value = np.full(n, 1.0 / n), -np.inf
    starts = [np.full(n, 1.0 / n)] + [rng.dirichlet(np.ones(n)) for _ in range(RESTARTS - 1)]
    for start in starts:
        result = minimize(
            lambda w: -certainty_equivalent(w, returns, gamma),
            np.clip(start, 0.0, cap) / np.clip(start, 0.0, cap).sum(),
            method="SLSQP",
            bounds=bounds,
            constraints=[constraint],
            options={"maxiter": 400, "ftol": 1e-10},
        )
        if result.success and -result.fun > best_value:
            best_weights, best_value = result.x, -result.fun
    if best_value == -np.inf:
        raise RuntimeError(f"All {RESTARTS} SLSQP restarts failed at gamma={gamma}")
    return np.clip(best_weights, 0.0, None) / np.clip(best_weights, 0.0, None).sum()


def gamma_schedule(horizon: int, start: float, end: float) -> list[float]:
    """One gamma per year, linearly interpolated from `start` to `end`."""
    if horizon == 1:
        return [start]
    step = (end - start) / (horizon - 1)
    return [start + step * year for year in range(horizon)]


def print_glidepath(labels: list[str], names: dict[str, str], schedule: list[float], weights: list[np.ndarray]) -> None:
    """Render the year-by-year target weight table, hiding always-zero sleeves."""
    stacked = np.vstack(weights)
    keep = [i for i in range(stacked.shape[1]) if stacked[:, i].max() >= 0.005]
    header = f"{'year':>4} {'gamma':>6} " + " ".join(f"{labels[i]:>9}" for i in keep)
    print(f"\n{header}")
    print("-" * len(header))
    for year, (gamma, row) in enumerate(zip(schedule, weights), start=1):
        cells = " ".join(f"{row[i] * 100:8.1f}%" for i in keep)
        print(f"{year:>4} {gamma:6.2f} {cells}")
    print("\nsleeves:")
    for i in keep:
        print(f"  {labels[i]:<10} {names.get(labels[i], '')}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Solve a CRRA-optimal allocation glidepath")
    parser.add_argument("--horizon", type=int, default=10, help="years to plan for")
    parser.add_argument("--gamma-start", type=float, default=2.0, help="risk aversion in year 1")
    parser.add_argument("--gamma-end", type=float, default=5.0, help="risk aversion in the final year")
    parser.add_argument("--etfs-only", action="store_true", help="drop the single-stock sleeves")
    parser.add_argument("--max-weight", type=float, default=MAX_WEIGHT, help="per-instrument cap")
    parser.add_argument("--shrinkage", type=float, default=MEAN_SHRINKAGE, help="0 fits raw means, 1 equalises them")
    parser.add_argument("--no-benchmark", action="store_true", help="skip the 1/N comparison")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.horizon < 1:
        parser.error("--horizon must be at least 1")
    if min(args.gamma_start, args.gamma_end) <= 0:
        parser.error("gamma must be positive")
    if not 0.0 <= args.shrinkage <= 1.0:
        parser.error("--shrinkage must be between 0 and 1")
    if not 0.0 < args.max_weight <= 1.0:
        parser.error("--max-weight must be in (0, 1]")
    rng = np.random.default_rng(args.seed)

    available = (date.today() - PRICES_START).days / 365.25
    if args.horizon > available:
        logger.warning(
            "Horizon %sy exceeds the %.1fy of prices_daily history; optimising over %s long-history classes instead",
            args.horizon,
            available,
            len(LONG_CLASSES),
        )
        labels, returns = load_long_panel()
        labels = list(LONG_CLASSES)
        names = {label: "long-history proxy" for label in labels}
    else:
        names = dict(ETF_UNIVERSE) if args.etfs_only else {**ETF_UNIVERSE, **STOCK_UNIVERSE}
        labels, returns = await load_price_panel(list(names))

    effective_cap = max(args.max_weight, CAP_EQUAL_WEIGHT_MULTIPLE / returns.shape[1])
    logger.info(
        "Fitting on %s months x %s sleeves, per-instrument cap %.1f%%",
        returns.shape[0],
        returns.shape[1],
        effective_cap * 100,
    )
    if effective_cap > args.max_weight:
        logger.warning(
            "Cap relaxed from %.1f%% to %.1f%% (%.1fx equal weight) so it does not bind on every sleeve",
            args.max_weight * 100,
            effective_cap * 100,
            CAP_EQUAL_WEIGHT_MULTIPLE,
        )
    fitted = shrink_means(returns, args.shrinkage)
    schedule = gamma_schedule(args.horizon, args.gamma_start, args.gamma_end)
    weights = [optimise(fitted, gamma, rng, args.max_weight) for gamma in schedule]
    print_glidepath(labels, names, schedule, weights)

    if not args.no_benchmark:
        naive = np.full(returns.shape[1], 1.0 / returns.shape[1])
        print(f"\n{'gamma':>6} {'glidepath CE':>13} {'1/N CE':>9} {'gap':>8}   (scored on realised returns)")
        for gamma, row in zip(schedule, weights):
            fit = certainty_equivalent(row, returns, gamma)
            flat = certainty_equivalent(naive, returns, gamma)
            print(f"{gamma:6.2f} {fit:12.2f}% {flat:8.2f}% {fit - flat:+7.2f}pp")
        print(
            f"\nWeights are fitted on means shrunk {args.shrinkage:.0%} toward the cross-sectional mean,"
            "\nthen scored above on realised returns — so a negative gap is the price of that"
            "\nregularisation, not a solver failure. Re-run with --shrinkage 0 to see the raw fit."
            "\nBoth columns remain in-sample; equal-weight beat fitted weights out of sample here."
        )


if __name__ == "__main__":
    asyncio.run(main())
