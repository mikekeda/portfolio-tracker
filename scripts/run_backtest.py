"""
scripts/run_backtest.py
=======================
Backtest a trade-agent strategy on point-in-time data. Run from project root:

    python scripts/run_backtest.py --strategy rules --start 2018-01-01
    python scripts/run_backtest.py --strategy rules --start 2018-01-01 --verify-invariance
    python scripts/run_backtest.py --strategy buyhold:VUAG.L --start 2018-01-01   # known-answer check
    python scripts/run_backtest.py --strategy oracle --start 2022-01-01           # leakage canary

Prints a metrics table vs the BENCHES benchmarks and writes equity/trade CSVs
to backtests/. The oracle strategy deliberately cheats (scores = next week's
returns); it exists to prove the harness would expose leakage — its Sharpe
should be absurd, and a clean strategy's result is only meaningful because of that.
"""

import argparse
import asyncio
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backend.agent.backtest.data import MarketData, load_market_data
from backend.agent.backtest.engine import BacktestResult, buy_and_hold_curve, run_backtest
from backend.agent.backtest.metrics import summarize
from backend.agent.rules_strategy import RulesStrategy
from backend.agent.types import AgentLimits, PortfolioState, TradeIntent
from backend.app import get_session
from config import BENCHES, logger

OUT_DIR = Path("backtests")


class BuyAndHoldStrategy:
    """Deploys fully into one symbol at the first rebalance, then holds.

    Running it through the whole engine must reproduce the analytic
    buy-and-hold curve within fee tolerance — the engine's known-answer test.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.name = f"buyhold:{symbol}"

    def propose(self, d, features, state: PortfolioState) -> list[TradeIntent]:
        if state.quantities.get(self.symbol, 0) > 0:
            return []
        return [TradeIntent(self.symbol, "buy", 0.99, score=1.0, rationale={"trigger": "buy and hold"})]


class OracleStrategy:
    """Deliberately leaking canary: scores each symbol by its NEXT 5-day return."""

    name = "oracle"

    def __init__(self, md: MarketData):
        self._forward = md.gbp_prices.shift(-5) / md.gbp_prices - 1

    def propose(self, d, features, state: PortfolioState) -> list[TradeIntent]:
        fwd = self._forward.loc[d].reindex(features.index).dropna()
        if fwd.empty:
            return []
        top = fwd.sort_values(ascending=False).head(5)
        intents = [
            TradeIntent(s, "add" if state.quantities.get(s, 0) > 0 else "buy", 0.15, score=float(v),
                        rationale={"forward_5d": float(v)})
            for s, v in top.items()
            if v > 0
        ]
        for s, q in state.quantities.items():
            if q > 0 and s in fwd.index and fwd[s] < 0 and len(state.quantities) > 5:
                intents.append(TradeIntent(s, "exit", None, score=float(fwd[s]), rationale={"forward_5d": float(fwd[s])}))
        return intents


def build_strategy(spec: str, md: MarketData, limits: AgentLimits):
    if spec == "rules":
        return RulesStrategy(limits)
    if spec == "oracle":
        return OracleStrategy(md)
    if spec.startswith("buyhold:"):
        return BuyAndHoldStrategy(spec.split(":", 1)[1])
    raise SystemExit(f"unknown strategy: {spec}")


def print_report(result: BacktestResult, md: MarketData, initial_cash: float, limits: AgentLimits) -> None:
    rows = {result.strategy: summarize(result.equity, result.trades)}
    for bench in BENCHES:
        if bench in md.gbp_prices.columns:
            curve = buy_and_hold_curve(md, bench, result.equity.index, initial_cash, limits)
            rows[f"hold {bench}"] = summarize(curve, pd.DataFrame())
    print("\n" + pd.DataFrame(rows).to_string())


def verify_invariance(strategy_spec: str, md: MarketData, args, limits: AgentLimits) -> None:
    """Rerun with data truncated at the window midpoint; history up to the
    cutoff must be identical, otherwise future data leaked into past decisions."""
    full = run_backtest(build_strategy(strategy_spec, md, limits), md, args.start, args.end,
                        args.initial_cash, args.rebalance, limits)
    dates = full.equity.index
    cutoff = dates[len(dates) // 2]
    md_cut = md.truncated(cutoff)
    part = run_backtest(build_strategy(strategy_spec, md_cut, limits), md_cut, args.start, cutoff,
                        args.initial_cash, args.rebalance, limits)

    eq_full = full.equity.loc[:cutoff]
    if not np.allclose(eq_full.values, part.equity.values, rtol=0, atol=1e-9):
        raise SystemExit(f"INVARIANCE FAILED: equity curves diverge before {cutoff} — future data is leaking")
    t_full = full.trades[full.trades["fill_date"] <= cutoff].reset_index(drop=True) if not full.trades.empty else full.trades
    t_part = part.trades[part.trades["fill_date"] <= cutoff].reset_index(drop=True) if not part.trades.empty else part.trades
    if not t_full.equals(t_part):
        raise SystemExit(f"INVARIANCE FAILED: trade logs diverge before {cutoff} — future data is leaking")
    print(f"invariance OK: {len(t_full)} trades and {len(eq_full)} equity points identical up to {cutoff}")


def check_known_answer(
    result: BacktestResult, md: MarketData, symbol: str, initial_cash: float, limits: AgentLimits
) -> None:
    analytic = buy_and_hold_curve(md, symbol, result.equity.index, initial_cash, limits)
    # Engine holds ~1% cash (target weight 0.99) and fills one day after the
    # first rebalance, so allow a small tolerance on the final value.
    ratio = float(result.equity.iloc[-1] / analytic.iloc[-1])
    status = "OK" if 0.97 <= ratio <= 1.03 else "FAILED"
    print(f"known-answer {status}: engine={result.equity.iloc[-1]:.0f} analytic={analytic.iloc[-1]:.0f} ratio={ratio:.4f}")
    if status == "FAILED":
        raise SystemExit(1)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest a trade-agent strategy")
    parser.add_argument("--strategy", default="rules", help="rules | oracle | buyhold:<SYMBOL>")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2018, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--rebalance", choices=("daily", "weekly"), default="weekly")
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--verify-invariance", action="store_true")
    args = parser.parse_args()

    limits = AgentLimits.from_config()
    if args.strategy.startswith("buyhold:"):
        # The known-answer strategy must be able to put ~100% into one symbol;
        # the position/cluster caps would otherwise block full deployment.
        limits = AgentLimits(
            max_daily_turnover=1.0,
            min_holdings=1,
            max_position_weight=1.0,
            min_trade_gbp=0.0,
            fx_fee=limits.fx_fee,
            cluster_caps={},
            stamp_duty=limits.stamp_duty,
            french_ftt=limits.french_ftt,
        )
    async with get_session() as session:
        md = await load_market_data(session, args.start, args.end)
    logger.info("Loaded %d symbols × %d days", md.gbp_prices.shape[1], md.gbp_prices.shape[0])

    if args.verify_invariance:
        verify_invariance(args.strategy, md, args, limits)
        return

    strategy = build_strategy(args.strategy, md, limits)
    result = run_backtest(strategy, md, args.start, args.end, args.initial_cash, args.rebalance, limits)
    print_report(result, md, args.initial_cash, limits)

    if args.strategy.startswith("buyhold:"):
        check_known_answer(result, md, args.strategy.split(":", 1)[1], args.initial_cash, limits)

    OUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = args.strategy.replace(":", "_")
    result.equity.to_csv(OUT_DIR / f"{slug}_{stamp}_equity.csv")
    if not result.trades.empty:
        result.trades.to_csv(OUT_DIR / f"{slug}_{stamp}_trades.csv", index=False)
    print(f"\nwrote {OUT_DIR}/{slug}_{stamp}_equity.csv and trades CSV")


if __name__ == "__main__":
    asyncio.run(main())
