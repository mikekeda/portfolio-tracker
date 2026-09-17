"""Exercise real joins, CUSIP resolution and all manager views on PostgreSQL."""

import asyncio
from datetime import date, datetime

import pytest
from sqlalchemy import select

from backend.utils.form13f import _get_form13f_for_instruments
from backend.views.form13f import (
    get_form13f_manager_detail,
    get_form13f_managers_list,
    get_13f_not_in_portfolio,
    get_form13f_highlights,
)
from models import Form13FFiling, Form13FHolding, Form13FManager, Instrument, InstrumentYahoo
from postgres_helpers import postgres_database


@pytest.mark.parametrize(
    "symbol, old_cusip, new_cusip, old_shares, shares, factor, adjusted",
    [
        ("HON", "438516106", "438516205", 8019, 4010, 0.9535, 4009.5),
        ("ODD", "123456789", "123456789", 100, 200, 1.241, None),
    ],
)
def test_corporate_actions_in_real_queries(symbol, old_cusip, new_cusip, old_shares, shares, factor, adjusted):
    async def check():
        async with postgres_database() as factory:
            async with factory.begin() as session:
                session.add(
                    Instrument(
                        id=7,
                        t212_code=symbol,
                        yahoo_symbol=symbol,
                        name=symbol,
                        currency="USD",
                        created_at=datetime(2026, 9, 16),
                        updated_at=datetime(2026, 9, 16),
                    )
                )
                session.add(Form13FManager(id=1, name="Dodge & Cox", cik="1", created_at=datetime(2026, 9, 16)))
                await session.flush()
                session.add(
                    InstrumentYahoo(
                        instrument_id=7,
                        splits={"2026-06-29": factor},
                        **{
                            key: {}
                            for key in (
                                "info",
                                "cashflow",
                                "earnings",
                                "recommendations",
                                "analyst_price_targets",
                                "news",
                                "pes",
                                "balance_sheet",
                                "income_stmt",
                            )
                        },
                    )
                )
                session.add_all(
                    [
                        Form13FFiling(
                            id=1,
                            manager_id=1,
                            report_date=date(2026, 3, 31),
                            form="13F-HR",
                            accession_number="1",
                            total_value=2000000,
                            created_at=datetime(2026, 9, 16),
                        ),
                        Form13FFiling(
                            id=2,
                            manager_id=1,
                            report_date=date(2026, 6, 30),
                            form="13F-HR",
                            accession_number="2",
                            total_value=1000000,
                            created_at=datetime(2026, 9, 16),
                        ),
                    ]
                )
                await session.flush()
                session.add_all(
                    [
                        Form13FHolding(
                            filing_id=1,
                            instrument_id=7,
                            issuer="Honeywell",
                            cusip=old_cusip,
                            shares=old_shares,
                            value=2000000,
                        ),
                        Form13FHolding(
                            filing_id=2,
                            instrument_id=None if symbol == "HON" else 7,
                            issuer=symbol,
                            cusip=new_cusip,
                            shares=shares,
                            value=1000000,
                        ),
                    ]
                )
            async with factory() as session:
                holdings = await _get_form13f_for_instruments(session, [7])
                assert holdings[7]["score"] is None
                assert holdings[7]["holders"][0]["shares_prev_adjusted"] == adjusted
                detail = await get_form13f_manager_detail(1, session=session)
                assert detail["portfolio"][0]["yahoo_symbol"] == symbol
                assert detail["moves"]["new_positions"] == detail["moves"]["closed_positions"] == []
                managers = await get_form13f_managers_list(session=session)
                assert managers[0]["activity"] == {"new": 0, "closed": 0, "increased": 0, "trimmed": 0, "stable": 1}
                discovery = await get_13f_not_in_portfolio(min_managers=1, session=session)
                assert discovery[0]["yahoo_symbol"] == symbol
                assert discovery[0]["buy_count"] == discovery[0]["sell_count"] == 0
                raw = (await session.execute(select(Form13FHolding).where(Form13FHolding.filing_id == 2))).scalar_one()
                assert raw.instrument_id == (None if symbol == "HON" else 7)
                assert raw.cusip == new_cusip
                highlights = await get_form13f_highlights(session=session)
                assert highlights["most_bought"] == highlights["most_sold"] == []

    asyncio.run(check())


def test_closed_and_reentered_holdings_match_stock_detail():
    from backend.views.instrument import get_instrument

    async def check():
        async with postgres_database() as factory:
            async with factory.begin() as session:
                for iid, symbol in [(7, 'EXIT'), (8, 'OLD'), (9, 'REENTER')]:
                    session.add(Instrument(id=iid, t212_code=symbol, yahoo_symbol=symbol, name=symbol,
                                           currency='GBP', created_at=datetime(2026, 9, 16), updated_at=datetime(2026, 9, 16)))
                session.add(Form13FManager(id=1, name='Manager', cik='1', created_at=datetime(2026, 9, 16)))
                session.add(Form13FManager(id=2, name='Buyer', cik='2', created_at=datetime(2026, 9, 16)))
                await session.flush()
                for fid, period in [(1, date(2025, 12, 31)), (2, date(2026, 3, 31)), (3, date(2026, 6, 30))]:
                    session.add(Form13FFiling(id=fid, manager_id=1, report_date=period, form='13F-HR',
                                             accession_number=str(fid), total_value=1000000, created_at=datetime(2026, 9, 16)))
                for fid, period in [(4, date(2026, 3, 31)), (5, date(2026, 6, 30))]:
                    session.add(Form13FFiling(id=fid, manager_id=2, report_date=period, form='13F-HR',
                                             accession_number=str(fid), total_value=1000000, created_at=datetime(2026, 9, 16)))
                await session.flush()
                session.add(Form13FHolding(filing_id=5, instrument_id=7, issuer='EXIT', cusip='7', shares=1, value=1000))
                for fid, iid, qty in [(1, 7, 1), (2, 7, 100), (1, 8, 100), (1, 9, 1), (3, 9, 100)]:
                    session.add(Form13FHolding(filing_id=fid, instrument_id=iid, issuer=str(iid),
                                               cusip=str(iid), shares=qty, value=qty * 1000))
            async with factory() as session:
                holdings = await _get_form13f_for_instruments(session, [7, 8, 9])
                assert 8 not in holdings  # Absence in both recent quarters, not an eternal exit.
                assert holdings[7]['score'] == -2
                assert holdings[9]['holders'][0]['change'] == 'New'
                for iid, symbol in [(7, 'EXIT'), (8, 'OLD'), (9, 'REENTER')]:
                    stock = await get_instrument(symbol, session=session)
                    expected = holdings.get(iid, {'score': None, 'holders': []})
                    assert stock['form13f_score'] == expected['score']
                    assert stock['form13f_holdings'] == expected['holders']
                # The $100k exit sorts before the $1k new buy, despite zero current value.
                closed = holdings[7]['holders'][0]
                assert closed['report_date'] == '2026-06-30' and closed['shares'] == 0
                assert closed['report_date_prev'] == '2026-03-31' and closed['shares_prev'] == 100
                detail = await get_form13f_manager_detail(1, session=session)
                assert detail['moves']['closed_positions'][0]['shares_prev'] == closed['shares_prev']

    asyncio.run(check())
