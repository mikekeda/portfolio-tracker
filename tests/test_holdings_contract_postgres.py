"""Screener and thesis rules may only read fields that real Holdings rows carry."""

import asyncio
from datetime import date, datetime
from unittest.mock import AsyncMock

from backend.schemas.instrument_thesis import THESIS_RULE_FIELDS
from backend.screener_config import get_screener_config
from backend.views import portfolio as holdings_view
from models import HoldingDaily, Instrument, InstrumentYahoo
from postgres_helpers import postgres_database

# The allocation band comes from the thesis itself at evaluation time, not from the row.
THESIS_BAND_FIELDS = {"target_weight_min_pct", "target_weight_max_pct"}


def test_holdings_rows_carry_every_rule_field(monkeypatch):
    monkeypatch.setattr(holdings_view, "get_rates", AsyncMock(return_value={"USD": 0.75, "GBP": 1.0, "GBX": 0.01}))
    monkeypatch.setattr(holdings_view, "get_effective_betas", AsyncMock(return_value={}))
    monkeypatch.setattr(holdings_view, "get_dcf_analyses", AsyncMock(return_value=[]))
    monkeypatch.setattr(holdings_view, "calculate_technical_indicators_for_symbols", AsyncMock(return_value=({}, {})))

    async def first_row():
        async with postgres_database() as factory:
            async with factory.begin() as session:
                session.add(Instrument(id=1, t212_code="ACME_US_EQ", yahoo_symbol="ACME", name="Acme",
                                       currency="USD", created_at=datetime(2026, 10, 13),
                                       updated_at=datetime(2026, 10, 13)))
                await session.flush()
                session.add(InstrumentYahoo(
                    instrument_id=1,
                    info={"currency": "USD", "financialCurrency": "USD"},
                    **{key: {} for key in ("cashflow", "income_stmt", "balance_sheet", "earnings",
                                           "recommendations", "analyst_price_targets", "splits", "pes")},
                    news=[], updated_at=datetime(2026, 10, 13),
                ))
                session.add(HoldingDaily(instrument_id=1, date=date(2026, 10, 13), quantity=1, avg_price=10,
                                         current_price=10, ppl=0, fx_ppl=0, updated_at=datetime(2026, 10, 13)))
            async with factory() as session:
                return (await holdings_view.build_portfolio(session))["holdings"][0]

    row = asyncio.run(first_row())
    rule_fields = set(get_screener_config().get_available_fields()) | (THESIS_RULE_FIELDS - THESIS_BAND_FIELDS)
    assert rule_fields - set(row) == set()
