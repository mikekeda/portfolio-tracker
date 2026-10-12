"""Detail, holdings and persisted feature rows must use the same valuation basis."""

import asyncio
from datetime import date, datetime
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from backend.utils.dcf import _compute_dcf_full
from backend.views import instrument as detail_view
from backend.views import portfolio as holdings_view
from models import FeaturesDaily, HoldingDaily, Instrument, InstrumentYahoo
from postgres_helpers import postgres_database
from scripts.update_features import _row_from_holding


def test_shared_valuation_reaches_detail_holdings_and_features(monkeypatch):
    rates = {"USD": .75, "EUR": .9, "BRL": .15, "GBP": 1, "GBX": .01}

    async def dcf(instruments, session, **kwargs):
        return [_compute_dcf_full(i, {}, rates) for i in instruments]

    for view in (detail_view, holdings_view):
        monkeypatch.setattr(view, "get_rates", AsyncMock(return_value=rates))
        monkeypatch.setattr(view, "get_effective_betas", AsyncMock(return_value={}))
        monkeypatch.setattr(view, "get_dcf_analyses", dcf)
    monkeypatch.setattr(holdings_view, "calculate_technical_indicators_for_symbols", AsyncMock(return_value=({}, {})))

    async def check():
        async with postgres_database() as factory:
            async with factory.begin() as session:
                for id_, symbol, financial_currency in [(1, "ASML", "EUR"), (2, "PBR", "BRL")]:
                    session.add(Instrument(id=id_, t212_code=symbol + "_US_EQ", yahoo_symbol=symbol,
                                           name=symbol, currency="USD",
                                           created_at=datetime(2026, 10, 12), updated_at=datetime(2026, 10, 12)))
                    await session.flush()
                    session.add(InstrumentYahoo(
                        instrument_id=id_,
                        info={"currency": "USD", "financialCurrency": financial_currency,
                              "marketCap": 1000, "currentPrice": 10, "sharesOutstanding": 100,
                              "totalDebt": 200, "totalCash": 50, "ebitda": 100, "totalRevenue": 500,
                              "enterpriseValue": 999999, "enterpriseToEbitda": 999, "priceToBook": 999},
                        cashflow={"2025-12-31": {"Free Cash Flow": 100}},
                        income_stmt={"2025-12-31": {"Total Revenue": 500}},
                        **{key: {} for key in ("balance_sheet", "earnings", "recommendations",
                                              "analyst_price_targets", "splits", "pes")},
                        news=[], updated_at=datetime(2026, 10, 12),
                    ))
                    session.add(HoldingDaily(instrument_id=id_, date=date.today(), quantity=1,
                                             avg_price=10, current_price=10, ppl=0, fx_ppl=0,
                                             updated_at=datetime(2026, 10, 12)))

            async with factory() as session:
                holdings = (await holdings_view.build_portfolio(session, show_all=True))["holdings"]
                by_symbol = {h["yahoo_symbol"]: h for h in holdings}
                for id_, symbol, expected_yield in [(1, "ASML", 12), (2, "PBR", 10)]:
                    detail = await detail_view.get_instrument(symbol, session=session)
                    holding = by_symbol[symbol]
                    assert detail["fundamentals"]["fcfYield"] == pytest.approx(expected_yield)
                    assert holding["free_cashflow_yield"] == pytest.approx(expected_yield)
                    assert holding["ps_ratio"] == detail["fundamentals"]["priceToSalesTtm"]
                    assert holding["net_debt_to_ebitda"] == detail["fundamentals"]["netDebtToEbitda"]
                    assert detail["fundamentals"]["priceToBook"] is None
                    assert detail["fundamentals"]["enterpriseValue"] != 999999
                    if symbol == "PBR":
                        assert detail["dcf_price"] is None and holding["dcf_price"] is None
                        assert "currencies differ" in detail["dcf_unavailable_reason"]
                    session.add(FeaturesDaily(**_row_from_holding(holding, id_)))
                await session.commit()
                rows = (await session.scalars(select(FeaturesDaily).order_by(FeaturesDaily.instrument_id))).all()
                assert rows[0].fcf_yield == pytest.approx(12)
                assert rows[1].fcf_yield == pytest.approx(10)
                assert rows[1].dcf_price is None
                assert "currencies differ" in rows[1].extras["dcf_unavailable_reason"]
                assert rows[0].extras["valuation_issues"] == {}
                assert rows[1].extras["valuation_issues"] == {}

    asyncio.run(check())
