"""Target selection for the 13F repair script."""
from contextlib import contextmanager
from unittest.mock import Mock

import pytest

from scripts import fix_13f_backfill as module

INVESTORS = [
    {"name": "Healthy", "cik": "1"},
    {"name": "Empty", "cik": "2"},
    {"name": "MultiCik", "cik": "3", "ciks": ["3", "4"]},
]


@pytest.fixture
def scraped(monkeypatch):
    session = Mock()

    @contextmanager
    def db():
        yield session

    monkeypatch.setattr(module, "get_session", db)
    monkeypatch.setattr(module, "INVESTORS", INVESTORS)
    monkeypatch.setattr(module, "investor_ciks", lambda inv: inv.get("ciks", [inv["cik"]]))
    monkeypatch.setattr(module, "_damaged_manager_names", Mock(return_value={"Empty"}))
    scrape = Mock(side_effect=lambda inv, **kwargs: [{
        "investor": inv["name"], "reportDate": "2026-06-30", "holdingsCount": 1, "totalValue": 100,
    }])
    monkeypatch.setattr(module, "scrape_investor", scrape)
    return scrape


def test_default_run_targets_only_detected_damage(scraped):
    module.fix_13f_backfill(dry_run=True)
    assert [call.args[0]["name"] for call in scraped.call_args_list] == ["Empty", "MultiCik"]


def test_named_investor_is_rescraped_without_detected_damage(scraped):
    # A duplicated amendment table has holdings and a total value, so the detector cannot see it.
    module.fix_13f_backfill(only="Healthy", dry_run=True)
    assert [call.args[0]["name"] for call in scraped.call_args_list] == ["Healthy"]
    assert scraped.call_args.kwargs["existing_dates"] == set()


def test_unknown_investor_exits(scraped):
    with pytest.raises(SystemExit, match="not a tracked investor"):
        module.fix_13f_backfill(only="Missing", dry_run=True)


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("failure", ["exception", "empty"])
def test_failed_scrape_never_reports_success_or_saves_partial_repair(scraped, monkeypatch, dry_run, failure):
    scraped.side_effect = [
        [{"investor": "Empty", "reportDate": "2026-06-30", "holdingsCount": 1, "totalValue": 100}],
        RuntimeError("SEC unavailable") if failure == "exception" else [],
    ]
    save = Mock()
    monkeypatch.setattr(module, "_save_to_db", save)
    with pytest.raises(RuntimeError, match="no changes saved.*MultiCik"):
        module.fix_13f_backfill(dry_run=dry_run)
    save.assert_not_called()
