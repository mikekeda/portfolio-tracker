"""Exercise the critical updater's real writes with external APIs replaced."""

import asyncio
from datetime import date, datetime, timezone
from unittest.mock import Mock

import pandas as pd
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from models import HoldingDaily, Instrument, InstrumentYahoo, PricesDaily
from postgres_helpers import postgres_database
from scripts import update_data as module


class Clock(datetime):
    day = 17

    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 9, cls.day, 8, tzinfo=timezone.utc).astimezone(tz)


def run_in_database(check):
    async def run():
        async with postgres_database() as factory:
            async with factory() as session:
                schema = await session.scalar(text("SELECT current_schema()"))
                url = session.bind.url.set(drivername="postgresql+psycopg2")
            engine = create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
            try:
                check(engine)
            finally:
                engine.dispose()
    asyncio.run(run())


def price(symbol, day, value=10):
    return PricesDaily(symbol=symbol, date=date(2026, 9, day), updated_at=datetime(2026, 9, day, 14),
                       open_price=value, high_price=value, low_price=value, close_price=value,
                       adj_close_price=value, volume=100)


def frame(symbol, days, value=20):
    return pd.DataFrame({(symbol, field): [value] * len(days) for field in
                         ("Open", "High", "Low", "Close", "Adj Close", "Volume")},
                        index=pd.to_datetime([f"2026-09-{day:02}" for day in days]))


def test_real_price_writes_repair_then_skip_and_catch_up_only_lagging_symbol(monkeypatch):
    monkeypatch.setattr(module, "datetime", Clock)
    monkeypatch.setattr(Clock, "day", 17)

    def check(engine):
        monkeypatch.setattr(module, "_get_session_factory", lambda: sessionmaker(
            engine, autoflush=False, expire_on_commit=False))
        with Session(engine) as session:
            session.add_all([price("CURRENT", 16), price("LAGGING", 15)])
            session.commit()

        available = {"CURRENT": [15, 16, 17], "LAGGING": [15]}

        def download(**kwargs):
            return pd.concat([frame(symbol, available[symbol]) for symbol in kwargs["tickers"]], axis=1, sort=True)

        fetch = Mock(side_effect=download)
        monkeypatch.setattr(module.yf, "download", fetch)
        module.update_prices(set())
        assert set(fetch.call_args.kwargs["tickers"]) == {"CURRENT", "LAGGING"}
        assert fetch.call_args.kwargs["start"] == "2026-09-10"
        with Session(engine) as session:
            rows = list(session.scalars(select(PricesDaily)))
            assert len(rows) == 3
            assert all(row.date < date(2026, 9, 17) and row.close_price == 20 for row in rows)

        available["LAGGING"] = [16]
        module.update_prices(set())
        assert fetch.call_args.kwargs["tickers"] == ["LAGGING"]
        assert fetch.call_args.kwargs["start"] == "2026-09-16"
        module.update_prices(set())
        assert fetch.call_count == 2
        with Session(engine) as session:
            assert len(list(session.scalars(select(PricesDaily)))) == 4

        monkeypatch.setattr(Clock, "day", 18)
        module.update_prices(set())
        assert fetch.call_count == 3
        assert set(fetch.call_args.kwargs["tickers"]) == {"CURRENT", "LAGGING"}
        assert fetch.call_args.kwargs["start"] == "2026-09-11"

    run_in_database(check)


def test_late_insert_failure_rolls_back_whole_batch_and_keeps_prior_commit(monkeypatch):
    monkeypatch.setattr(module, "datetime", Clock)
    monkeypatch.setattr(module, "DB_CHUNK_ROWS", 1)
    fetch = Mock(return_value=frame("GOOD", [16]))
    monkeypatch.setattr(module.yf, "download", fetch)

    def check(engine):
        with Session(engine) as session:
            assert module._update_prices(session, ["GOOD"], date(2026, 9, 10)) == {"GOOD"}
            session.commit()
            fetch.return_value = frame("FAILED", [15, 16])
            execute = session.execute
            calls = 0

            def fail_second_insert(statement, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("Simulated failure after the first chunk was written")
                return execute(statement, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(session, "execute", fail_second_insert)
                assert module._update_prices(session, ["FAILED"], date(2026, 9, 10)) == set()
            session.commit()
            assert list(session.scalars(select(PricesDaily.symbol))) == ["GOOD"]
            assert module._update_prices(session, ["FAILED"], date(2026, 9, 10)) == {"FAILED"}
            session.commit()
            assert len(list(session.scalars(select(PricesDaily)))) == 3

    run_in_database(check)


def test_quarter_refresh_preserves_failed_module_and_updates_broker_holding(monkeypatch):
    monkeypatch.setattr(module, "datetime", Clock)
    current, previous = "2026-06-30", "2026-03-31"

    def check(engine):
        monkeypatch.setattr(module, "_get_session_factory", lambda: sessionmaker(
            engine, autoflush=False, expire_on_commit=False))
        with Session(engine) as session:
            session.add(Instrument(id=1, t212_code="TEST_US_EQ", yahoo_symbol="TEST", name="Test", currency="USD"))
            session.flush()
            session.add(InstrumentYahoo(instrument_id=1, news=[], splits={},
                info={"quoteType": "EQUITY"}, quarterly_income_stmt={current: {"Total Revenue": 100}},
                quarterly_balance_sheet={current: {"Total Assets": 200}},
                quarterly_cashflow={previous: {"Operating Cash Flow": 30}},
                profile_fetched_at=datetime(2026, 9, 1), **{key: {} for key in (
                    "cashflow", "earnings", "recommendations", "analyst_price_targets", "pes",
                    "balance_sheet", "income_stmt",
                )}))
            session.commit()

        payload = {key: {} for key in ("cashflow", "balance_sheet", "income_stmt", "earnings",
            "recommendations", "analyst_price_targets", "splits", "quarterly_cashflow",
            "quarterly_balance_sheet")}
        payload.update(info={"quoteType": "EQUITY"}, news=[], estimates=None,
                       quarterly_income_stmt={"2026-09-30": {"Total Revenue": 110}})
        fetch = Mock(return_value={"TEST": payload})
        monkeypatch.setattr(module, "get_yahoo_ticker_data", fetch)
        monkeypatch.setattr(module, "fetch_holdings", Mock(return_value={"TEST_US_EQ": {
            "quantity": 3, "averagePrice": 10, "currentPrice": 12, "ppl": 6, "fxPpl": 0,
        }}))
        module.update_holdings()
        assert fetch.call_args.args[1] == {"TEST": date(2026, 3, 31)}
        with Session(engine) as session:
            cached = session.get(InstrumentYahoo, 1)
            assert cached.quarterly_cashflow == {previous: {"Operating Cash Flow": 30}}
            assert cached.quarterly_balance_sheet == {current: {"Total Assets": 200}}
            assert set(cached.quarterly_income_stmt) == {current, "2026-09-30"}
            holding = session.scalar(select(HoldingDaily))
            assert holding.quantity == 3 and holding.current_price == 12
            assert holding.date == date(2026, 9, 17)

    run_in_database(check)
