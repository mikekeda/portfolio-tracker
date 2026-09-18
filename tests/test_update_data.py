"""Recent price repair, quarterly refresh and update-run cleanup."""
from contextlib import contextmanager
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from scripts import update_data as module


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 9, 17, 8, tzinfo=timezone.utc).astimezone(tz)


def test_price_download_refetches_completed_bars_and_excludes_today(monkeypatch):
    monkeypatch.setattr(module, "datetime", Clock)
    frame = pd.DataFrame({("TEST", field): values for field, values in {
        "Open": [1, 2], "High": [2, 3], "Low": [1, 2], "Close": [2, 3],
        "Adj Close": [2, 3], "Volume": [10, 20],
    }.items()}, index=pd.to_datetime(["2026-09-16", "2026-09-17"]))
    download = Mock(return_value=frame)
    monkeypatch.setattr(module.yf, "download", download)
    session = Mock()
    assert module._update_prices(session, ["TEST"], date(2026, 9, 10)) == {"TEST"}
    assert download.call_args.kwargs["end"] == "2026-09-17"
    params = session.execute.call_args.args[0].compile().params
    assert params["date_m0"] == date(2026, 9, 16)
    assert params["close_price_m0"] == 2
    assert "date_m1" not in params


@pytest.fixture(autouse=True)
def no_split_refetch(monkeypatch):
    original = module._full_history_starts
    monkeypatch.setattr(module, "_full_history_starts", Mock(return_value={}))
    return original


def test_scheduler_overlaps_even_when_newest_bar_is_today(monkeypatch):
    monkeypatch.setattr(module, "datetime", Clock)
    session = Mock()
    session.scalars.return_value.all.return_value = ["TEST"]
    session.query.return_value.where.return_value.group_by.return_value.order_by.return_value.all.return_value = [
        SimpleNamespace(symbol="TEST", max_date=date(2026, 9, 17), refreshed_at=datetime(2026, 9, 16, 22))]
    @contextmanager
    def db():
        yield session
    monkeypatch.setattr(module, "get_session", db)
    download = Mock()
    monkeypatch.setattr(module, "_update_prices", download)
    module.update_prices(set())
    assert download.call_args_list[0].args == (session, ["TEST"], date(2026, 9, 10))


@pytest.mark.parametrize("latest, refreshed, today, expected", [
    (date(2026, 9, 16), datetime(2026, 9, 16, 23, 59), date(2026, 9, 17), date(2026, 9, 10)),
    (date(2026, 9, 16), datetime(2026, 9, 17, 0, 1), date(2026, 9, 17), None),
    (date(2026, 9, 15), datetime(2026, 9, 17, 0, 1), date(2026, 9, 17), date(2026, 9, 16)),
    (date(2026, 9, 18), datetime(2026, 9, 20, 0, 1), date(2026, 9, 20), None),
    (date(2026, 9, 18), datetime(2026, 9, 21, 0, 1), date(2026, 9, 21), None),
    (date(2026, 9, 16), datetime(2026, 9, 17, 23), date(2026, 9, 18), date(2026, 9, 11)),
    (date(2026, 8, 1), None, date(2026, 9, 17), date(2026, 8, 1)),
])
def test_price_overlap_once_per_utc_day_and_catch_up(latest, refreshed, today, expected):
    assert module._price_fetch_start(latest, refreshed, today) == expected


def test_intraday_run_retries_only_failed_or_lagging_symbols(monkeypatch):
    monkeypatch.setattr(module, "datetime", Clock)
    rows = [SimpleNamespace(symbol=symbol, max_date=date(2026, 9, 16),
                            refreshed_at=datetime(2026, 9, 16, 22))
            for symbol in ("GOOD", "FAILED", "LAGGING")]
    rows[2].max_date = date(2026, 9, 15)
    session = Mock()
    session.scalars.return_value.all.return_value = [row.symbol for row in rows]
    session.query.return_value.where.return_value.group_by.return_value.order_by.return_value.all.return_value = rows

    @contextmanager
    def db():
        yield session

    def store(session, symbols, start):
        for row in rows:
            if row.symbol in symbols and row.symbol != "FAILED":
                row.refreshed_at = datetime(2026, 9, 17, 0, 30)
        return set(symbols) - {"FAILED"}

    monkeypatch.setattr(module, "get_session", db)
    download = Mock(side_effect=store)
    monkeypatch.setattr(module, "_update_prices", download)
    module.update_prices(set())
    assert set(download.call_args_list[0].args[1]) == {"GOOD", "FAILED", "LAGGING"}
    download.reset_mock()
    module.update_prices(set())
    calls = [(call.args[1], call.args[2]) for call in download.call_args_list[:2]]
    assert calls == [(["FAILED"], date(2026, 9, 10)), (["LAGGING"], date(2026, 9, 16))]


@pytest.mark.parametrize("splits, first_date, first_written, expected", [
    # AVB in production: oldest row written 2025-10-12, 2026-08-17 split, recent rows refetched on the new scale.
    ({"2026-08-17": 2.793}, date(2015, 10, 5), datetime(2025, 10, 12, 9), "2026-08-17"),
    # Already refetched after the split, or the split predates stored history.
    ({"2026-08-17": 2.793}, date(2015, 10, 5), datetime(2026, 8, 18, 0, 5), None),
    ({"2024-06-10": 10.0}, date(2024, 7, 1), datetime(2024, 7, 2), None),
    # Written on the ex-date may predate Yahoo's rescale: refetch once more.
    ({"2026-09-14": 2.0}, date(2016, 1, 4), datetime(2026, 9, 14, 0, 1), "2026-09-14"),
    ({"not-a-date": 2.0, "2026-09-14": None}, date(2016, 1, 4), datetime(2020, 1, 1), None),
    (None, date(2016, 1, 4), datetime(2020, 1, 1), None),
])
def test_split_after_oldest_write_requires_full_history(splits, first_date, first_written, expected):
    assert module._split_stale_history(splits, first_date, first_written) == expected


def test_split_day_refetches_history_but_future_event_does_not(monkeypatch):
    monkeypatch.setattr(module, "datetime", Clock)
    # Yahoo can already restate prior closes on the ex-date even though today's bar is excluded.
    assert module._split_stale_history({"2026-09-17": 2}, date(2015, 1, 1), datetime(2025, 1, 1)) == "2026-09-17"
    assert module._split_stale_history({"2026-09-18": 2}, date(2015, 1, 1), datetime(2025, 1, 1)) is None


def test_refreshed_first_row_cannot_hide_stale_split_history_in_postgres(monkeypatch, no_split_refetch):
    import asyncio

    from sqlalchemy import update
    from models import Instrument, InstrumentYahoo, PricesDaily
    from postgres_helpers import postgres_database

    monkeypatch.setattr(module, "datetime", Clock)

    async def check():
        async with postgres_database() as factory:
            async with factory.begin() as session:
                session.add(Instrument(id=1, t212_code="TEST", yahoo_symbol="TEST", name="Test", currency="USD",
                                       created_at=datetime(2026, 9, 16), updated_at=datetime(2026, 9, 16)))
                await session.flush()
                session.add(InstrumentYahoo(instrument_id=1, splits={"2026-09-14": 2}, news=[],
                    updated_at=datetime(2026, 9, 16), **{key: {} for key in (
                        "info", "cashflow", "earnings", "recommendations", "analyst_price_targets", "pes",
                        "balance_sheet", "income_stmt",
                    )}))
                for day, written in [(date(2020, 1, 2), datetime(2026, 9, 16)),
                                     (date(2020, 1, 3), datetime(2026, 9, 10))]:
                    session.add(PricesDaily(symbol="TEST", date=day, updated_at=written,
                        open_price=1, high_price=1, low_price=1, close_price=1, adj_close_price=1, volume=1))
            async with factory.begin() as session:
                assert await session.run_sync(no_split_refetch, ["TEST"]) == {"TEST": date(2020, 1, 2)}
                await session.execute(update(PricesDaily).values(updated_at=datetime(2026, 9, 16)))
                assert await session.run_sync(no_split_refetch, ["TEST"]) == {}

    asyncio.run(check())


def test_split_refetch_gets_its_own_full_history_batch(monkeypatch):
    monkeypatch.setattr(module, "datetime", Clock)
    rows = [SimpleNamespace(symbol=symbol, max_date=date(2026, 9, 16), refreshed_at=datetime(2026, 9, 16, 22))
            for symbol in ("AVB", "CURRENT")]
    session = Mock()
    session.scalars.return_value.all.return_value = [row.symbol for row in rows]
    session.query.return_value.where.return_value.group_by.return_value.order_by.return_value.all.return_value = rows

    @contextmanager
    def db():
        yield session

    split_check = Mock(return_value={"AVB": date(2015, 10, 5)})
    monkeypatch.setattr(module, "_full_history_starts", split_check)
    monkeypatch.setattr(module, "get_session", db)
    download = Mock(return_value=set())
    monkeypatch.setattr(module, "_update_prices", download)
    module.update_prices(set())
    assert split_check.call_args.args[1] == ["AVB", "CURRENT"]
    calls = [(call.args[1], call.args[2]) for call in download.call_args_list[:2]]
    assert calls == [(["AVB"], date(2015, 10, 5)), (["CURRENT"], date(2026, 9, 10))]


def test_split_check_skips_symbols_already_refreshed_today(monkeypatch):
    monkeypatch.setattr(module, "datetime", Clock)
    rows = [SimpleNamespace(symbol="DONE", max_date=date(2026, 9, 16), refreshed_at=datetime(2026, 9, 17, 0, 30))]
    session = Mock()
    session.scalars.return_value.all.return_value = ["DONE"]
    session.query.return_value.where.return_value.group_by.return_value.order_by.return_value.all.return_value = rows

    @contextmanager
    def db():
        yield session

    split_check = Mock(return_value={})
    monkeypatch.setattr(module, "_full_history_starts", split_check)
    monkeypatch.setattr(module, "get_session", db)
    monkeypatch.setattr(module, "_update_prices", Mock(return_value=set()))
    module.update_prices(set())
    assert split_check.call_args.args[1] == []


def test_lagging_statement_drives_quarter_retry():
    current = {"2026-06-30": {"value": 1}}
    old = {"2026-03-31": {"value": 1}}
    stored = module._oldest_statement_period(current, old, current)
    assert module._quarterly_due(stored, date(2026, 6, 30))
    assert module._oldest_statement_period({}, current, None) == date(2026, 6, 30)
    assert module._oldest_statement_period({}, None, {}) is None


def test_failed_quarterly_module_does_not_discard_other_statements(monkeypatch):
    frame = pd.DataFrame({pd.Timestamp("2026-06-30"): {"value": 1}})
    class Ticker:
        ticker = "TEST"
        info = {"mostRecentQuarter": 1782777600}
        cashflow = balance_sheet = income_stmt = frame
        quarterly_balance_sheet = quarterly_income_stmt = frame
        @property
        def quarterly_cashflow(self):
            raise RuntimeError("provider failed")
    _, payload = module.fetch_profile_for_ticker(Ticker(), {}, set())
    assert payload["quarterly_cashflow"] == {}
    assert "2026-06-30" in payload["quarterly_balance_sheet"]
    assert "2026-06-30" in payload["quarterly_income_stmt"]


def test_update_failure_clears_cached_holdings(monkeypatch):
    monkeypatch.setattr(module, "update_currency_rates", Mock(side_effect=RuntimeError("offline")))
    holdings = Mock()
    monkeypatch.setattr(module, "fetch_holdings", holdings)
    with pytest.raises(RuntimeError, match="offline"):
        module.update_data()
    holdings.cache_clear.assert_called_once()


@pytest.mark.parametrize("failed_stage", [None, "rates", "instruments", "holdings", "prices", "portfolio"])
def test_pipeline_order_and_cache_cleanup_at_every_stage(monkeypatch, failed_stage):
    calls = []
    stages = [("rates", "update_currency_rates"), ("instruments", "update_instruments"),
              ("holdings", "update_holdings"), ("prices", "update_prices"), ("portfolio", "update_portfolio")]

    def stage(name):
        def run(*args):
            calls.append(name)
            if name == failed_stage:
                raise RuntimeError(name)
        return run

    for name, function in stages:
        monkeypatch.setattr(module, function, stage(name))
    holdings = Mock()
    holdings.cache_clear.side_effect = lambda: calls.append("clear")
    monkeypatch.setattr(module, "fetch_holdings", holdings)
    if failed_stage:
        with pytest.raises(RuntimeError, match=failed_stage):
            module.update_data()
        expected = [name for name, _ in stages]
        expected = expected[:expected.index(failed_stage) + 1]
    else:
        module.update_data()
        expected = [name for name, _ in stages]
    assert calls == expected + ["clear"]
