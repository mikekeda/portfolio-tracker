"""Point-in-time backtest engine.

Decision at close of date d (features use only data ≤ d), fills at the next
trading day's price. The constraint layer runs inside the loop exactly as in
the live path, so caps shape backtested behaviour the same way they will shape
real suggestions. Fills use GBP-normalized adjusted closes (see data.py).

The first rebalance runs with the turnover cap lifted: the book starts as cash
and the strategy's deployment mode builds the initial portfolio in one step
instead of trickling in 5% a day for a month.
"""

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from backend.agent.constraints import apply_constraints
from backend.agent.strategy import Strategy
from backend.agent.types import AgentLimits, PortfolioState, TradeOrder, is_fund, trade_fee_rate
from backend.agent.backtest.data import (
    MarketData,
    features_for_date,
    risk_columns,
    tradable_universe,
    use_sec_universe,
)


@dataclass
class BacktestResult:
    strategy: str
    equity: pd.Series  # daily GBP portfolio value
    trades: pd.DataFrame  # one row per fill
    metrics: dict = field(default_factory=dict)
    benchmarks: dict[str, pd.Series] = field(default_factory=dict)


def rebalance_schedule(dates: pd.Index, mode: str) -> set:
    """Decision dates: every day, or the last trading day of each ISO week."""
    if mode == "daily":
        return set(dates)
    if mode == "weekly":
        s = pd.Series(dates, index=dates)
        # One string key per date: a list of tuples would be read as multiple
        # grouper columns, not as labels (KeyError: (2018, 1)).
        iso_weeks = pd.Index([f"{d.isocalendar()[0]}-{d.isocalendar()[1]:02d}" for d in dates])
        return set(s.groupby(iso_weeks).last())
    raise ValueError(f"unknown rebalance mode: {mode}")


def run_backtest(
    strategy: Strategy,
    md: MarketData,
    start: date,
    end: date | None = None,
    initial_cash: float = 100_000.0,
    rebalance: str = "weekly",
    limits: AgentLimits | None = None,
) -> BacktestResult:
    limits = limits or AgentLimits.from_config()
    window = md.gbp_prices.loc[start:end] if end else md.gbp_prices.loc[start:]
    dates = window.index
    if len(dates) < 2:
        raise ValueError("backtest window has fewer than 2 trading days")
    rebs = rebalance_schedule(dates, rebalance)
    valuation_prices = md.gbp_prices.ffill()

    cash = initial_cash
    positions: dict[str, float] = {}
    pending: list[tuple[date, TradeOrder]] = []
    first_rebalance_done = False
    equity_vals: list[float] = []
    trade_log: list[dict] = []

    for i, d in enumerate(dates):
        cash, still_pending = _fill(pending, d, md, positions, cash, limits, trade_log)
        pending = still_pending

        prices_today = valuation_prices.loc[d]
        total_value = cash + sum(q * prices_today[s] for s, q in positions.items() if pd.notna(prices_today[s]))
        equity_vals.append(total_value)

        if d not in rebs or i == len(dates) - 1:
            continue

        universe = tradable_universe(md, d, sec_only=use_sec_universe(md, d))
        if not universe:
            continue
        weights = {
            s: q * prices_today[s] / total_value
            for s, q in positions.items()
            if q > 0 and pd.notna(prices_today[s])
        }
        features = features_for_date(md, d, universe)
        features = features.join(risk_columns(md, d, weights))

        state = PortfolioState(
            date=d,
            total_value_gbp=total_value,
            cash_gbp=cash,
            weights=weights,
            quantities={s: q for s, q in positions.items() if q > 0},
            currencies=md.currencies,
            tags=md.tags,
            etf_symbols=md.etf_symbols,
        )
        intents = strategy.propose(d, features, state)
        if not intents:
            first_rebalance_done = True
            continue

        decision_prices = {s: float(md.gbp_prices.loc[d, s]) for s in {i.symbol for i in intents}}
        eff_limits = limits
        if not first_rebalance_done:
            eff_limits = AgentLimits(**{**limits.__dict__, "max_daily_turnover": 1.0})
        orders = apply_constraints(intents, state, decision_prices, eff_limits)
        pending = [(d, o) for o in orders if o.executable]
        first_rebalance_done = True

    equity = pd.Series(equity_vals, index=dates, name=strategy.name)
    trades = pd.DataFrame(trade_log)
    return BacktestResult(strategy=strategy.name, equity=equity, trades=trades)


def _fill(
    pending: list[tuple[date, "TradeOrder"]],
    d: date,
    md: MarketData,
    positions: dict[str, float],
    cash: float,
    limits: AgentLimits,
    trade_log: list[dict],
) -> tuple[float, list]:
    """Fill pending orders at today's price; keep orders whose symbol didn't trade."""
    still_pending = []
    for decision_date, order in pending:
        assert d > decision_date, "fill must be strictly after the decision date"
        price = md.gbp_prices.loc[d, order.symbol] if order.symbol in md.gbp_prices.columns else None
        if price is None or pd.isna(price) or price <= 0:
            still_pending.append((decision_date, order))
            continue

        side = "sell" if order.action in ("exit", "trim") else "buy"
        fee_rate = trade_fee_rate(
            order.symbol,
            md.currencies.get(order.symbol),
            is_fund(order.symbol, md.etf_symbols),
            side,
            limits,
        )
        if order.action in ("exit", "trim"):
            quantity = positions.get(order.symbol, 0.0) if order.action == "exit" else order.value_gbp / price
            quantity = min(quantity, positions.get(order.symbol, 0.0))
            value = quantity * price
            positions[order.symbol] = positions.get(order.symbol, 0.0) - quantity
            if positions[order.symbol] <= 1e-9:
                positions.pop(order.symbol, None)
            cash += value * (1 - fee_rate)
        else:
            value = min(order.value_gbp, cash / (1 + fee_rate))
            if value <= 0:
                continue
            quantity = value / price
            positions[order.symbol] = positions.get(order.symbol, 0.0) + quantity
            cash -= value * (1 + fee_rate)

        trade_log.append(
            {
                "decision_date": decision_date,
                "fill_date": d,
                "symbol": order.symbol,
                "action": order.action,
                "value_gbp": round(value, 2),
                "quantity": quantity,
                "fee_gbp": round(value * fee_rate, 4),
                "score": order.score,
                "adjustments": "; ".join(order.adjustments),
            }
        )
    return cash, still_pending


def buy_and_hold_curve(
    md: MarketData, symbol: str, dates: pd.Index, initial_cash: float, limits: AgentLimits
) -> pd.Series:
    """Benchmark equity curve: everything into `symbol` on day one, hold."""
    prices = md.gbp_prices[symbol].reindex(dates).ffill()
    first_priced = prices.first_valid_index()
    if first_priced is None:
        raise ValueError(f"{symbol}: no prices in backtest window")
    fee_rate = trade_fee_rate(
        symbol, md.currencies.get(symbol), is_fund(symbol, md.etf_symbols), "buy", limits
    )
    quantity = initial_cash * (1 - fee_rate) / prices.loc[first_priced]
    curve = quantity * prices
    curve.loc[:first_priced] = initial_cash
    return curve.rename(symbol)
