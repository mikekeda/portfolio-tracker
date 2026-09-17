import asyncio
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.utils.form13f import _compute_form13f_change, _compute_form13f_signal_score, split_adjusted_shares
from backend.utils.form13f_actions import quantity_split_history
from backend.views.form13f import _adjust_previous_holdings, _aggregate_holdings_by_cusip, _build_portfolio_and_moves


def test_price_rally_does_not_count_as_buying():
    assert _compute_form13f_change(100, 100) == "+0.0%"
    assert _compute_form13f_signal_score(100, 100, 15000, 10000, 1000000) == 0
    assert _compute_form13f_change(50000, 49500) == "+1.0%"
    assert _compute_form13f_signal_score(295406, 496234, 100000, 111483, 1000000) == -1


def test_split_window_excludes_prior_date_and_future_splits():
    splits = {"2026-03-31": 10, "2026-04-10T00:00:00": 2, "2026-06-30": 3, "2026-07-01": 10}
    adjusted = split_adjusted_shares(100, splits, date(2026, 3, 31), date(2026, 6, 30))
    assert adjusted == 600
    assert _compute_form13f_change(600, adjusted) == "+0.0%"
    assert _compute_form13f_signal_score(600, adjusted, 100000, 10000, 1000000) == 0
    assert _compute_form13f_change(660, adjusted) == "+10.0%"
    assert _compute_form13f_signal_score(660, adjusted, 100000, 10000, 1000000) == 1


def test_reverse_split_and_new_closed_positions():
    adjusted = split_adjusted_shares(100, {"2026-04-10": 0.1}, date(2026, 3, 31), date(2026, 6, 30))
    assert _compute_form13f_change(10, adjusted) == "+0.0%"
    assert _compute_form13f_change(10, 0) == "New"
    assert _compute_form13f_change(0, 10) == "Closed"
    assert _compute_form13f_change(10, None) == "—"


@pytest.mark.parametrize(
    "symbol, feed, previous, current, expected",
    [
        ("FDX", {"2026-06-01": 1.241}, 9983449, 7233269, 9983449),
        ("HON", {"2026-06-29": 0.9535}, 8019, 4010, 4009.5),
        ("SCCO", {"2026-05-13": 1.01}, 10000, 10100, 10100),
        ("BKNG", {"2026-04-06": 25}, 997498, 25536950, 24937450),
        ("KLAC", {"2026-06-12": 10}, 100, 1000, 1000),
    ],
)
def test_verified_corporate_actions(symbol, feed, previous, current, expected):
    adjusted = split_adjusted_shares(
        previous, quantity_split_history(feed, symbol), date(2026, 3, 31), date(2026, 6, 30)
    )
    assert adjusted == expected
    assert _compute_form13f_signal_score(current, adjusted, value=100000, filing_total_value=1000000) == 0


def test_verified_override_is_applied_when_feed_omits_event():
    adjusted = split_adjusted_shares(8000, quantity_split_history({}, "HON"), date(2026, 3, 31), date(2026, 6, 30))
    assert adjusted == 4000


@pytest.mark.parametrize("factor", [1.241, 0.9535, 1.01, 0, -2, True, float("nan"), float("inf"), "2"])
def test_unverified_or_invalid_adjustments_withhold_comparison(factor):
    adjusted = split_adjusted_shares(
        100, quantity_split_history({"2026-05-01": factor}), date(2026, 3, 31), date(2026, 6, 30)
    )
    assert adjusted is None
    assert _compute_form13f_change(150, adjusted) == "—"
    assert _compute_form13f_signal_score(150, adjusted, value=100000, filing_total_value=1000000) == 0


def test_hon_cusip_transition_does_not_create_fake_new_and_closed_positions():
    from backend.utils.form13f import load_form13f_holdings

    previous = SimpleNamespace(
        filing_id=1, cusip="438516106", issuer="Honeywell", instrument_id=7, shares=8019, value=2000000
    )
    current = SimpleNamespace(
        filing_id=2, cusip="438516205", issuer="Honeywell", instrument_id=None, shares=4010, value=1000000
    )
    session = SimpleNamespace(execute=AsyncMock(return_value=Result(rows=[(previous, 7), (current, 7)])))
    resolved = asyncio.run(load_form13f_holdings(session, [1, 2]))
    assert current.instrument_id is None
    prior = _adjust_previous_holdings(
        _aggregate_holdings_by_cusip(resolved[:1]),
        {7: quantity_split_history({"2026-06-29": 0.9535}, "HON")},
        date(2026, 3, 31),
        date(2026, 6, 30),
    )
    portfolio, moves = _build_portfolio_and_moves(_aggregate_holdings_by_cusip(resolved[1:]), prior, 1000000, {})
    assert not moves["new_positions"] and not moves["closed_positions"]
    assert portfolio[0]["cusip"] == "438516205"
    assert portfolio[0]["shares_prev"] == 8019 and portfolio[0]["shares_prev_adjusted"] == 4009.5
    assert portfolio[0]["change"] == "+0.0%"
    sql = str(session.execute.call_args.args[0].compile(compile_kwargs={"literal_binds": True}))
    assert "438516205" in sql and "HON" in sql


def test_unverified_adjustment_is_not_ranked_as_a_manager_buy():
    previous = {"test": {"instrument_id": 1, "shares": 100, "value": 10000, "issuer": "Test"}}
    current = {"test": {"instrument_id": 1, "shares": 200, "value": 20000, "issuer": "Test"}}
    prior = _adjust_previous_holdings(
        previous, {1: quantity_split_history({"2026-05-01": 1.241})}, date(2026, 3, 31), date(2026, 6, 30)
    )
    portfolio, moves = _build_portfolio_and_moves(current, prior, 20000, {})
    assert portfolio[0]["shares_prev"] == 100 and portfolio[0]["shares_prev_adjusted"] is None
    assert portfolio[0]["estimated_flow"] is None
    assert moves["top_buys"] == [] and moves["top_sells"] == []


def test_manager_moves_use_trading_not_market_value_change():
    def holding(shares, value):
        return {"instrument_id": 1, "shares": shares, "value": value, "issuer": "Test"}

    previous = {"split": holding(100, 10000), "rally": holding(100, 10000), "sale": holding(100, 10000)}
    # A 2:1 split applies to all synthetic positions in this security.
    adjusted = _adjust_previous_holdings(previous, {1: {"2026-04-10": 2}}, date(2026, 3, 31), date(2026, 6, 30))
    current = {"split": holding(200, 10000), "rally": holding(200, 15000), "sale": holding(120, 12000)}
    rows, moves = _build_portfolio_and_moves(current, adjusted, 37000, {})
    assert moves["top_buys"] == []
    assert [r["cusip"] for r in moves["top_sells"]] == ["sale"]
    sale = next(r for r in rows if r["cusip"] == "sale")
    assert sale["shares_prev"] == 100 and sale["shares_prev_adjusted"] == 200
    assert sale["value_change"] == 2000 and sale["estimated_flow"] == -8000
    assert previous["sale"]["shares"] == 100  # raw filings are never modified


class Result:
    def __init__(self, value=None, rows=()):
        self.value = value
        self.rows = rows

    def scalar(self):
        return self.value

    def all(self):
        return self.rows


def test_holdings_13f_query_preserves_filed_shares_but_scores_split_basis():
    from backend.utils.form13f import _get_form13f_for_instruments

    manager = SimpleNamespace(id=1, name="Manager", cik="1")
    current = SimpleNamespace(id=2, manager_id=1, report_date=date(2026, 6, 30), total_value=1000000, accession_number="accession")
    previous = SimpleNamespace(id=1, manager_id=1, report_date=date(2026, 3, 31), total_value=1000000, accession_number="accession")
    rows = [
        (SimpleNamespace(filing_id=2, shares=200, value=20000), 7),
        (SimpleNamespace(filing_id=1, shares=100, value=10000), 7),
    ]
    session = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                Result(rows=[(7, {"2026-04-20": 2}, "TEST")]),
                Result(rows=[(current, manager), (previous, manager)]),
                Result(rows=rows),
            ]
        )
    )
    result = asyncio.run(_get_form13f_for_instruments(session, [7]))[7]
    assert result["score"] is None  # No directional trade, despite market value doubling.
    holder = result["holders"][0]
    assert holder["change"] == "+0.0%"
    assert holder["shares_prev"] == 100 and holder["shares_prev_adjusted"] == 200


def test_latest_manager_periods_handle_exits_reentry_and_old_positions():
    from backend.utils.form13f import _get_form13f_for_instruments

    manager = SimpleNamespace(id=1, name='Manager', cik='1')
    filings = [SimpleNamespace(id=i, manager_id=1, report_date=d, total_value=1000000, accession_number=str(i))
               for i, d in [(3, date(2026, 6, 30)), (2, date(2026, 3, 31)), (1, date(2025, 12, 31))]]
    # Only holdings in the latest two manager filings should be queried. Stock 8
    # existed in Q4 only: it must not reappear as a current holder. Stock 9
    # re-entered in Q2; its old Q4 holding must not become its comparison base.
    rows = [(SimpleNamespace(filing_id=fid, shares=qty, value=value), iid)
            for fid, iid, qty, value in [(2, 7, 100, 100000), (3, 9, 10, 10000),
                                         (2, 10, 100, 10000), (3, 10, 200, 20000)]]
    session = SimpleNamespace(execute=AsyncMock(side_effect=[
        Result(rows=[(iid, {}, str(iid)) for iid in [7, 8, 9, 10]]),
        Result(rows=[(f, manager) for f in filings]), Result(rows=rows),
    ]))
    result = asyncio.run(_get_form13f_for_instruments(session, [7, 8, 9, 10]))
    closed = result[7]['holders'][0]
    assert closed['change'] == 'Closed' and closed['scored']
    assert closed['shares'] == closed['value'] == 0
    assert closed['shares_prev'] == 100 and closed['value_prev'] == 100000
    assert closed['report_date'] == '2026-06-30' and closed['report_date_prev'] == '2026-03-31'
    assert closed['sec_filing_url'].endswith('/3/')
    assert result[7]['score'] == -2
    assert 8 not in result
    assert result[9]['holders'][0]['change'] == 'New'
    assert result[10]['holders'][0]['change'] == '+100.0%'
    sql = str(session.execute.call_args.args[0].compile(compile_kwargs={'literal_binds': True}))
    assert 'filing_id IN (3, 2)' in sql


def test_exit_uses_prior_conviction_in_mixed_signal_score():
    from backend.utils.form13f import _get_form13f_for_instruments

    sold = SimpleNamespace(id=1, name='Sold', cik='1')
    bought = SimpleNamespace(id=2, name='Bought', cik='2')
    def filing(fid, mid, d):
        return SimpleNamespace(id=fid, manager_id=mid, report_date=d, total_value=1000000, accession_number=str(fid))
    session = SimpleNamespace(execute=AsyncMock(side_effect=[
        Result(rows=[(7, {}, 'TEST')]),
        Result(rows=[(filing(2, 1, date(2026, 6, 30)), sold),
                     (filing(4, 2, date(2026, 6, 30)), bought),
                     (filing(1, 1, date(2026, 3, 31)), sold),
                     (filing(3, 2, date(2026, 3, 31)), bought)]),
        Result(rows=[(SimpleNamespace(filing_id=1, shares=100, value=100000), 7),
                     (SimpleNamespace(filing_id=4, shares=10, value=10000), 7)]),
    ]))
    result = asyncio.run(_get_form13f_for_instruments(session, [7]))[7]
    assert result['score'] == -1.6  # (-2 * 10% + 2 * 1%) / 11%, not +2.


@pytest.mark.parametrize('adjusted', [100, 2500, None])
def test_exit_flow_uses_prior_value_without_mixing_split_price_bases(adjusted):
    from backend.utils.form13f import estimated_holder_flow
    assert estimated_holder_flow({'shares': 0, 'value': 0, 'shares_prev': 100,
                                  'shares_prev_adjusted': adjusted, 'value_prev': 10000}) == -10000
    assert estimated_holder_flow({'shares': 0, 'value': 0, 'shares_prev': 100}) is None


def test_holder_flow_matches_split_adjusted_buy_example():
    from backend.utils.form13f import estimated_holder_flow
    assert estimated_holder_flow({'shares': 25536950, 'value': 4551705968, 'shares_prev': 997498,
                                  'shares_prev_adjusted': 24937450}) == pytest.approx(106854880)


def test_ingestion_maps_both_honeywell_cusips_to_existing_instrument():
    from unittest.mock import Mock
    from scripts.scrape_13f import _build_cusip_to_instrument_map
    session = SimpleNamespace(execute=Mock(return_value=Result(rows=[SimpleNamespace(id=7, isin='US4385161066')])))
    mapping = _build_cusip_to_instrument_map(session)
    assert mapping['438516106'] == mapping['438516205'] == 7
