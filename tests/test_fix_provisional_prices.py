"""Repair selection for price history holding intraday bars stored as closes."""
from contextlib import contextmanager
from datetime import date
from unittest.mock import Mock

import pytest

from scripts import fix_provisional_prices as module


@pytest.fixture
def repair(monkeypatch):
    session = Mock()

    @contextmanager
    def db():
        yield session

    monkeypatch.setattr(module, "get_session", db)
    download = Mock(return_value={"AAA", "BBB"})
    monkeypatch.setattr(module, "_update_prices", download)
    return session, download


def results(session, *batches):
    session.execute.side_effect = [Mock(all=Mock(return_value=list(batch))) for batch in batches]


def test_affected_symbols_are_refetched_from_their_earliest_stored_row(repair):
    session, download = repair
    provisional = [("AAA", date(2026, 9, 8)), ("BBB", date(2026, 9, 1))]
    results(session, provisional, [("AAA", date(2016, 1, 4)), ("BBB", date(2020, 3, 2))], [])
    assert module.fix_provisional_prices() == 2
    # Earliest stored date, not the provisional bar's date: a mid-history refetch
    # would leave the older rows on a different adjustment basis.
    assert download.call_args.args[1:] == (["AAA", "BBB"], date(2016, 1, 4))
    session.commit.assert_called_once()


def test_clean_history_downloads_nothing(repair):
    session, download = repair
    results(session, [])
    assert module.fix_provisional_prices() == 0
    download.assert_not_called()


def test_dry_run_reports_without_downloading(repair):
    session, download = repair
    results(session, [("AAA", date(2026, 9, 8))], [("AAA", date(2016, 1, 4))])
    assert module.fix_provisional_prices(dry_run=True) == 0
    download.assert_not_called()


def test_missing_download_fails_instead_of_claiming_repair(repair):
    session, download = repair
    download.return_value = set()
    results(session, [("AAA", date(2026, 9, 8))], [("AAA", date(2016, 1, 4))],
            [("AAA", date(2026, 9, 8))], [("AAA", date(2016, 1, 4))])
    with pytest.raises(RuntimeError, match="Price repair incomplete.*AAA"):
        module.fix_provisional_prices()


def test_partial_download_with_unrepaired_old_bar_fails(repair):
    session, download = repair
    download.return_value = {"AAA"}
    results(session, [("AAA", date(2026, 9, 8))], [("AAA", date(2016, 1, 4))],
            [("AAA", date(2026, 9, 8))], [("AAA", date(2016, 1, 4))])
    with pytest.raises(RuntimeError, match="Price repair incomplete.*AAA"):
        module.fix_provisional_prices()
