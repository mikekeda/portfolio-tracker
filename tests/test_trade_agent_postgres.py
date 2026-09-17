"""Migration, real proposal replacement and rollback behavior on PostgreSQL."""

import asyncio
from datetime import date, datetime

import pandas as pd
import pytest
from sqlalchemy import select
from types import SimpleNamespace
from unittest.mock import AsyncMock

import backend.app as app_module
import scripts.run_trade_agent as runner
from backend.views.agent import get_suggestions
from models import HoldingDaily, Instrument, PortfolioDaily, TradeAgentRun, TradeSuggestion
from postgres_helpers import postgres_database


def test_no_action_rerun_and_failed_replacement_are_transactional(monkeypatch):
    async def check():
        async with postgres_database() as factory:
            monkeypatch.setattr(app_module, "_get_session_factory", lambda: factory)
            monkeypatch.setattr(runner, "datetime", SimpleNamespace(now=lambda tz: datetime(2026, 9, 16, 7, tzinfo=tz)))
            d = date(2026, 9, 15)
            md = SimpleNamespace(
                gbp_prices=pd.DataFrame({"TEST": [100.0]}, index=[d]), currencies={}, tags={}, etf_symbols=set()
            )
            monkeypatch.setattr(runner, "load_market_data", AsyncMock(return_value=md))
            monkeypatch.setattr(runner, "tradable_universe", lambda *args: ["TEST"])
            monkeypatch.setattr(runner, "features_for_date", lambda *args, **kwargs: pd.DataFrame(index=["TEST"]))
            monkeypatch.setattr(runner, "risk_columns", lambda *args: pd.DataFrame(index=["TEST"]))
            monkeypatch.setattr(runner.RulesStrategy, "propose", lambda *args: [])
            async with factory.begin() as session:
                session.add_all(
                    [
                        Instrument(
                            id=i,
                            t212_code=str(i),
                            yahoo_symbol="TEST" if i == 1 else str(i),
                            name=str(i),
                            currency="USD",
                            created_at=datetime(2026, 9, 16),
                            updated_at=datetime(2026, 9, 16),
                        )
                        for i in (1, 2, 3)
                    ]
                )
                await session.flush()
                session.add(
                    HoldingDaily(
                        instrument_id=1,
                        date=d,
                        quantity=1,
                        avg_price=100,
                        current_price=100,
                        ppl=0,
                        updated_at=datetime(2026, 9, 16),
                    )
                )
                session.add(PortfolioDaily(
                    date=d, value=100, cash=0, invested=100, unrealised_profit=0, realised_profit=0,
                    updated_at=datetime(2026, 9, 16),
                ))
                for i, status in enumerate(("proposed", "accepted", "dismissed"), 1):
                    session.add(
                        TradeSuggestion(
                            date=d, instrument_id=i, strategy="rules", action="buy", value_gbp=100, status=status
                        )
                    )
            await runner.run_trade_agent()
            async with factory() as session:
                result = await get_suggestions(session=session)
                assert result["date"] == str(d)
                assert result["run"]["status"] == "success" and result["run"]["order_count"] == 0
                assert {s["status"] for s in result["suggestions"]} == {"accepted", "dismissed"}
            # Put a pending proposal back, then fail after its deletion.
            async with factory.begin() as session:
                session.add(
                    TradeSuggestion(
                        date=d, instrument_id=1, strategy="rules", action="buy", value_gbp=100, status="proposed"
                    )
                )
            generate = runner._generate_suggestions

            async def fail_after_delete(session, limits, run):
                await generate(session, limits, run)
                raise RuntimeError("failure after DELETE")

            monkeypatch.setattr(runner, "_generate_suggestions", fail_after_delete)
            with pytest.raises(RuntimeError, match="failure after DELETE"):
                await runner.run_trade_agent()
            async with factory() as session:
                result = await get_suggestions(session=session)
                assert result["run"]["status"] == "success" and result["latest_run"]["status"] == "failed"
                assert len(result["suggestions"]) == 3
                runs = (await session.execute(select(TradeAgentRun))).scalars().all()
                assert len(runs) == 2

    asyncio.run(check())


def test_missing_held_price_persists_skip_and_keeps_previous_proposals(monkeypatch):
    async def check():
        async with postgres_database() as factory:
            monkeypatch.setattr(app_module, '_get_session_factory', lambda: factory)
            monkeypatch.setattr(runner, 'datetime', SimpleNamespace(now=lambda tz: datetime(2026, 9, 16, 7, tzinfo=tz)))
            d = date(2026, 9, 15)
            md = SimpleNamespace(gbp_prices=pd.DataFrame({'SMALL': [100.]}, index=[d]))
            monkeypatch.setattr(runner, 'load_market_data', AsyncMock(return_value=md))
            async with factory.begin() as session:
                for iid, symbol in [(1, 'SMALL'), (2, 'LARGE')]:
                    session.add(Instrument(id=iid, t212_code=symbol, yahoo_symbol=symbol, name=symbol, currency='GBP',
                                           created_at=datetime(2026, 9, 16), updated_at=datetime(2026, 9, 16)))
                await session.flush()
                for iid, qty in [(1, 1), (2, 1000)]:
                    session.add(HoldingDaily(instrument_id=iid, date=d, quantity=qty, avg_price=100, current_price=100,
                                             ppl=0, updated_at=datetime(2026, 9, 16)))
                session.add(PortfolioDaily(date=d, value=100100, cash=0, invested=100100,
                                           unrealised_profit=0, realised_profit=0, updated_at=datetime(2026, 9, 16)))
                session.add(TradeSuggestion(date=d, instrument_id=1, strategy='rules', action='buy', value_gbp=100,
                                            status='proposed'))
            await runner.run_trade_agent()
            async with factory() as session:
                result = await get_suggestions(session=session)
                assert result['latest_run']['status'] == 'skipped'
                assert 'LARGE' in result['latest_run']['reason']
                assert len(result['suggestions']) == 1
                assert result['suggestions'][0]['value_gbp'] == 100
    asyncio.run(check())
