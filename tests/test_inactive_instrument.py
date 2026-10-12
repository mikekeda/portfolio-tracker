"""Inactive price feeds retain history without exposing current investment signals."""
from copy import deepcopy

from backend.views.instrument import _apply_inactive_status


def test_inactive_response_withholds_signals_without_mutating_cached_inputs():
    detail = {'instrument': {'symbol': 'AVB'}, 'fundamentals': {'currentPrice': 184.06},
              'analyst_price_targets': {'median': 201}, 'dcf_diff': -.68, 'screener_score': -2,
              'prices': [{'date': '2026-07-16', 'close': 69.9964}],
              'earnings_reports': [{'summary': 'Historical report'}], 'cashflow': {'2025': {}},
              'position_review': {'position_action': 'buy'}, 'agent_suggestion': {'action': 'buy'}}
    cached = detail['fundamentals']
    history = {k: deepcopy(detail[k]) for k in ('prices', 'earnings_reports', 'cashflow')}
    _apply_inactive_status(detail)
    assert detail['instrument']['is_active'] is False
    assert detail['fundamentals'] == {} and cached == {'currentPrice': 184.06}
    assert detail['analyst_price_targets'] == {}
    for key in ('dcf_diff', 'screener_score', 'position_review', 'agent_suggestion'):
        assert detail[key] is None
    assert {k: detail[k] for k in history} == history


def test_active_instrument_keeps_current_signals():
    detail = {'instrument': {'symbol': 'MSFT'}, 'fundamentals': {'currentPrice': 420}, 'dcf_diff': .2}
    _apply_inactive_status(detail)
    assert detail['instrument']['is_active'] is True
    assert detail['fundamentals']['currentPrice'] == 420 and detail['dcf_diff'] == .2
