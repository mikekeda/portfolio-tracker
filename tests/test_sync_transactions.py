"""History fetch failures must not masquerade as successful empty imports."""
from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest
import requests

from scripts import sync_transactions as module


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(module.time, "sleep", lambda _: None)


@pytest.mark.parametrize("page", [1, 2])
def test_404_on_any_page_is_a_failed_fetch(monkeypatch, page):
    ok = Mock(status_code=200)
    ok.json.return_value = {"items": [{"reference": "first"}], "nextPagePath": "?cursor=next"}
    missing = Mock(status_code=404)
    missing.raise_for_status.side_effect = requests.HTTPError("404")
    monkeypatch.setattr(module.requests, "get", Mock(side_effect=[ok] * (page - 1) + [missing]))
    with pytest.raises(requests.HTTPError):
        module._fetch_all_pages("/api/v0/equity/history/orders")


def test_successful_empty_page_is_valid_but_malformed_response_is_not(monkeypatch):
    monkeypatch.setattr(module, "_get", Mock(return_value={"items": [], "nextPagePath": None}))
    assert module._fetch_all_pages("/history") == []
    monkeypatch.setattr(module, "_get", Mock(return_value={"error": "unavailable"}))
    with pytest.raises(ValueError, match="Invalid history"):
        module._fetch_all_pages("/history")


def prepare_sync(monkeypatch):
    session = Mock()
    empty = Mock()
    empty.scalars.return_value.all.return_value = []
    latest = datetime(2026, 9, 17, 7)
    dated = Mock()
    dated.scalar.return_value = latest
    session.execute.side_effect = [empty, empty, dated, dated, dated]
    @contextmanager
    def db():
        yield session
    monkeypatch.setattr(module, "get_session", db)
    return session, latest


def test_incremental_sync_uses_seven_day_overlap(monkeypatch):
    _, latest = prepare_sync(monkeypatch)
    fetch = Mock(return_value=[])
    monkeypatch.setattr(module, "_fetch_all_pages", fetch)
    module.sync_transactions(dry_run=True)
    assert len(fetch.call_args_list) == 3
    assert all(call.kwargs["stop_before"] == latest - timedelta(days=7) for call in fetch.call_args_list)


def test_fetch_failure_propagates_without_import_or_commit(monkeypatch):
    session, _ = prepare_sync(monkeypatch)
    monkeypatch.setattr(module, "_fetch_all_pages", Mock(side_effect=requests.HTTPError("down")))
    importer = Mock()
    monkeypatch.setattr(module, "_import_orders", importer)
    with pytest.raises(requests.HTTPError):
        module.sync_transactions()
    importer.assert_not_called()
    session.commit.assert_not_called()


def test_failed_item_rolls_back_its_feed_but_commits_independent_feeds(monkeypatch):
    session, _ = prepare_sync(monkeypatch)
    monkeypatch.setattr(module, "_fetch_all_pages", Mock(return_value=[]))
    def bad_import(*args):
        args[-1]["errors"] = 1
    monkeypatch.setattr(module, "_import_orders", bad_import)
    with pytest.raises(RuntimeError, match="orders rolled back; successful feeds committed"):
        module.sync_transactions()
    assert session.commit.call_count == 2


def test_overlap_recovers_late_cash_and_skips_existing_reference(monkeypatch):
    monkeypatch.setattr(module, "_find_semantic_duplicate", Mock(return_value=None))
    stats = {"cash_imported": 0, "cash_skipped": 0, "dedup_skipped": 0, "errors": 0}
    session = Mock()
    ids = {"already-imported"}
    existing = {"reference": "already-imported", "dateTime": "2026-09-16T07:00:00Z",
                "type": "DEPOSIT", "amount": 100}
    late = {**existing, "reference": "late", "dateTime": "2026-09-13T07:00:00Z"}
    module._import_cash([existing, late, late], ids, session, stats)
    assert stats == {"cash_imported": 1, "cash_skipped": 2, "dedup_skipped": 0, "errors": 0}
    assert session.add.call_args.args[0].timestamp == datetime(2026, 9, 13, 7)


def test_since_mode_preserves_explicit_cutoff(monkeypatch):
    prepare_sync(monkeypatch)
    fetch = Mock(return_value=[])
    monkeypatch.setattr(module, "_fetch_all_pages", fetch)
    since = datetime(2025, 1, 1)
    module.sync_transactions(dry_run=True, since=since)
    assert all(call.kwargs["stop_before"] == since for call in fetch.call_args_list)


def test_failed_feed_retains_cursor_and_other_feeds_commit_in_postgres(monkeypatch):
    import asyncio

    from sqlalchemy import create_engine, select, text
    from sqlalchemy.orm import Session

    from models import TransactionAction, TransactionHistory
    from postgres_helpers import postgres_database

    raw = {
        "orders": [
            {"order": {"id": 10, "side": "BUY", "ticker": "TEST_US_EQ"},
             "fill": {"id": 11, "filledAt": "2026-09-17T07:00:00Z", "quantity": 1,
                      "price": 20, "walletImpact": {"netValue": 20}}},
            {"order": {"id": 12, "side": "BUY"}},  # No date: reject this feed.
        ],
        "transactions": [{"reference": "cash-1", "dateTime": "2026-09-17T07:00:00Z",
                          "type": "DEPOSIT", "amount": 100}],
        "dividends": [{"reference": "div-1", "paidOn": "2026-09-17T07:00:00Z",
                       "ticker": "TEST_US_EQ", "amount": 5}],
    }
    fetch = Mock(side_effect=lambda path, **kwargs: raw[path.rsplit("/", 1)[-1]])
    monkeypatch.setattr(module, "_fetch_all_pages", fetch)

    def exercise(engine):
        @contextmanager
        def db():
            with Session(engine) as session:
                try:
                    yield session
                    session.commit()
                except Exception:
                    session.rollback()
                    raise

        monkeypatch.setattr(module, "get_session", db)
        old_date = datetime(2026, 9, 1, 7)
        with db() as session:
            session.add(TransactionHistory(csv_id="old-order", timestamp=old_date,
                        action=TransactionAction.MARKET_BUY, ticker="TEST", quantity=1, total=10))

        with pytest.raises(RuntimeError, match="orders rolled back"):
            module.sync_transactions()
        with db() as session:
            assert set(session.scalars(select(TransactionHistory.csv_id))) == {"old-order", "cash-1", "div-1"}

        # An unchanged order watermark retries the rejected interval even after the other feeds advance.
        fetch.reset_mock()
        with pytest.raises(RuntimeError, match="orders rolled back"):
            module.sync_transactions()
        assert fetch.call_args_list[0].kwargs["stop_before"] == old_date - timedelta(days=7)
        with db() as session:
            assert len(list(session.scalars(select(TransactionHistory.csv_id)))) == 3

        raw["orders"].pop()
        stats = module.sync_transactions()
        assert stats["order_imported"] == 1 and stats["cash_imported"] == stats["div_imported"] == 0
        with db() as session:
            assert len(list(session.scalars(select(TransactionHistory.csv_id)))) == 4

    async def run():
        async with postgres_database() as factory:
            async with factory() as session:
                schema = await session.scalar(text("SELECT current_schema()"))
                url = session.bind.url.set(drivername="postgresql+psycopg2")
            engine = create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
            try:
                exercise(engine)
            finally:
                engine.dispose()

    asyncio.run(run())
