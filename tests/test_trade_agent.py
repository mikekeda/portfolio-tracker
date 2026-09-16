"""Run outcomes, transaction boundaries and default evaluation selection."""

import asyncio
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from backend.views.agent import get_suggestions
from models import TradeAgentRun


class Result:
    def __init__(self, value=None, rows=()):
        self.value, self.rows = value, rows

    def scalar(self):
        return self.value

    def all(self):
        return self.rows


def run_record(status="success", as_of=date(2026, 9, 15), orders=0):
    return TradeAgentRun(
        as_of_date=as_of,
        strategy="rules",
        ran_at=datetime(2026, 9, 16, 7, tzinfo=timezone.utc),
        status=status,
        intent_count=orders,
        order_count=orders,
        executable_count=orders,
        reason=None if status == "success" else "No price data loaded",
    )


def test_newer_prices_cannot_hide_latest_successful_zero_action_run():
    session = SimpleNamespace(execute=AsyncMock(side_effect=[Result(run_record()), Result()]))
    result = asyncio.run(get_suggestions(session=session))
    assert result["date"] == "2026-09-15"
    assert result["run"]["status"] == "success" and result["run"]["order_count"] == 0
    assert result["suggestions"] == []
    assert all("prices_daily" not in str(c.args[0]) for c in session.execute.call_args_list)


@pytest.mark.parametrize("status", ["failed", "skipped"])
def test_unsuccessful_attempt_keeps_previous_success_visible(status):
    latest, successful = run_record(status, date(2026, 9, 16)), run_record()
    session = SimpleNamespace(execute=AsyncMock(side_effect=[Result(latest), Result(successful), Result()]))
    result = asyncio.run(get_suggestions(session=session))
    assert result["date"] == "2026-09-15"
    assert result["run"]["status"] == "success"
    assert result["latest_run"]["status"] == status


def test_legacy_suggestions_are_used_before_run_history_exists():
    session = SimpleNamespace(execute=AsyncMock(side_effect=[Result(), Result(), Result(date(2026, 9, 14)), Result()]))
    result = asyncio.run(get_suggestions(session=session))
    assert result["date"] == "2026-09-14" and result["run"] is None
    assert "max(trade_suggestions.date)" in str(session.execute.call_args_list[2].args[0])


def test_explicit_date_does_not_substitute_a_different_run():
    session = SimpleNamespace(execute=AsyncMock(side_effect=[Result(), Result(), Result()]))
    result = asyncio.run(get_suggestions(session=session, for_date=date(2026, 8, 1)))
    assert result["date"] == "2026-08-01" and result["run"] is None
    sql = str(session.execute.call_args_list[0].args[0].compile(compile_kwargs={"literal_binds": True}))
    assert "2026-08-01" in sql


def make_session(results=()):
    return SimpleNamespace(
        execute=AsyncMock(side_effect=results),
        add=Mock(),
        commit=AsyncMock(),
        rollback=AsyncMock(),
        close=AsyncMock(),
    )


@pytest.fixture
def agent_inputs(monkeypatch):
    import scripts.run_trade_agent as runner

    d = date(2026, 9, 15)
    md = SimpleNamespace(
        gbp_prices=pd.DataFrame({"TEST": [100.0]}, index=[d]), currencies={}, tags={}, etf_symbols=set()
    )
    monkeypatch.setattr(runner, "load_market_data", AsyncMock(return_value=md))
    monkeypatch.setattr(runner, "tradable_universe", lambda *args: ["TEST"])
    monkeypatch.setattr(runner, "features_for_date", lambda *args, **kwargs: pd.DataFrame(index=["TEST"]))
    monkeypatch.setattr(runner, "risk_columns", lambda *args: pd.DataFrame(index=["TEST"]))
    monkeypatch.setattr(runner.RulesStrategy, "propose", lambda *args: [])
    return runner, md


def test_zero_intent_rerun_commits_deletion_and_run_together(monkeypatch, agent_inputs):
    import backend.app as app_module

    runner, md = agent_inputs
    d = md.gbp_prices.index[-1]
    session = make_session(
        [Result(d), Result(rows=[SimpleNamespace(yahoo_symbol="TEST", quantity=1)]), Result(0), Result()]
    )
    monkeypatch.setattr(app_module, "_get_session_factory", lambda: lambda: session)
    asyncio.run(runner.run_trade_agent())
    sql = str(session.execute.call_args_list[-1].args[0].compile(compile_kwargs={"literal_binds": True}))
    assert "DELETE FROM trade_suggestions" in sql
    assert "status = 'proposed'" in sql and "2026-09-15" in sql and "strategy = 'rules'" in sql
    run = session.add.call_args.args[0]
    assert run.status == "success" and run.as_of_date == d
    assert (run.intent_count, run.order_count, run.executable_count) == (0, 0, 0)
    session.commit.assert_awaited_once()
    session.rollback.assert_not_awaited()


@pytest.mark.parametrize("reason", ["prices", "portfolio"])
def test_missing_inputs_record_skipped_without_deleting_proposals(monkeypatch, agent_inputs, reason):
    import backend.app as app_module

    runner, md = agent_inputs
    if reason == "prices":
        md.gbp_prices = pd.DataFrame()
        session = make_session()
    else:
        session = make_session([Result(date(2026, 9, 15)), Result(), Result(0)])
    monkeypatch.setattr(app_module, "_get_session_factory", lambda: lambda: session)
    asyncio.run(runner.run_trade_agent())
    run = session.add.call_args.args[0]
    assert run.status == "skipped" and run.intent_count is None
    assert all("DELETE" not in str(c.args[0]) for c in session.execute.call_args_list)
    session.commit.assert_awaited_once()


def test_failure_rolls_back_proposals_then_records_failure_separately(monkeypatch, agent_inputs):
    import backend.app as app_module

    runner, _ = agent_inputs
    first, second = make_session(), make_session()
    sessions = iter([first, second])
    monkeypatch.setattr(app_module, "_get_session_factory", lambda: lambda: next(sessions))
    error = RuntimeError("private failure detail")

    async def fail(session, limits, run):
        run.update(as_of_date=date(2026, 9, 15), intent_count=2)
        raise error

    monkeypatch.setattr(runner, "_generate_suggestions", fail)
    with pytest.raises(RuntimeError) as exc:
        asyncio.run(runner.run_trade_agent())
    assert exc.value is error
    first.rollback.assert_awaited_once()
    first.commit.assert_not_awaited()
    second.commit.assert_awaited_once()
    run = second.add.call_args.args[0]
    assert run.status == "failed" and run.as_of_date == date(2026, 9, 15)
    assert "private" not in run.reason


def test_all_vetoed_intents_still_record_success_and_order_counts(monkeypatch, agent_inputs):
    import backend.app as app_module
    from backend.agent.types import TradeIntent

    runner, md = agent_inputs
    session = make_session(
        [
            Result(md.gbp_prices.index[-1]),
            Result(rows=[SimpleNamespace(yahoo_symbol="TEST", quantity=1)]),
            Result(0),
            Result(),
            Result(rows=[("TEST", 7)]),
            Result(),
        ]
    )
    monkeypatch.setattr(app_module, "_get_session_factory", lambda: lambda: session)
    monkeypatch.setattr(runner.RulesStrategy, "propose", lambda *args: [TradeIntent("TEST", "exit", None, score=-1)])
    asyncio.run(runner.run_trade_agent())
    run = session.add.call_args.args[0]
    assert run.status == "success"
    assert (run.intent_count, run.order_count, run.executable_count) == (1, 1, 0)
    assert "INSERT INTO trade_suggestions" in str(session.execute.call_args_list[-1].args[0])


def test_unknown_action_fails_instead_of_becoming_an_empty_order_batch():
    from backend.agent.constraints import apply_constraints
    from backend.agent.types import TradeIntent

    with pytest.raises(ValueError, match="unknown action"):
        apply_constraints([TradeIntent("TEST", "typo", None, score=1)], SimpleNamespace(), {"TEST": 100})
