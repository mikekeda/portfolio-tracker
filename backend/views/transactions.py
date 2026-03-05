"""Transaction history and income summary endpoint."""
from collections import defaultdict
from typing import Any, Optional

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app import get_db_session
from models import Instrument, TransactionAction, TransactionHistory

router = APIRouter()


def _fees_gbp(fees: list | None, exchange_rate: float | None) -> float:
    """
    Sum all fees converted to GBP.

    Fee records may carry an explicit 'currency' field added by the import script.
    When absent (old records), we infer: WITHHOLDING_TAX is in the stock's trading
    currency; everything else is already in GBP.
    exchange_rate is stored as (GBP per unit of stock currency), so:
        fee_gbp = fee_foreign * exchange_rate
    """
    total = 0.0
    for f in (fees or []):
        amount = abs(f.get("quantity", 0.0))
        currency = f.get("currency")
        # Determine whether this fee needs FX conversion
        is_foreign = (
            currency not in (None, "GBP", "GBX")           # explicit non-GBP
            or (currency is None                             # old record: infer by name
                and f.get("name") == "WITHHOLDING_TAX")
        )
        if is_foreign and exchange_rate:
            amount *= exchange_rate
        total += amount
    return total


_DIVIDEND_ACTIONS = frozenset({
    TransactionAction.DIVIDEND,
    TransactionAction.DIVIDEND_PROPERTY,
    TransactionAction.DIVIDEND_TAX_EXEMPT,
})
_SELL_ACTIONS = frozenset({TransactionAction.MARKET_SELL, TransactionAction.LIMIT_SELL})


@router.get("/api/transactions")
async def get_transactions(
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """
    Return all transactions plus pre-computed summary stats and chart data.

    Summary and chart data always cover the full history — filtering is left
    to the frontend so the user can switch years without a round-trip.
    """
    # Build yahoo_symbol → (name, currency) lookup directly — the ORM relationship
    # joins on t212_code, but TransactionHistory.ticker stores the Yahoo symbol, so
    # a direct query by yahoo_symbol is more reliable.
    instr_rows = await session.execute(
        select(Instrument.yahoo_symbol, Instrument.name, Instrument.currency)
        .where(Instrument.yahoo_symbol.isnot(None))
    )
    instrument_map: dict[str, dict] = {
        row.yahoo_symbol: {"name": row.name, "currency": row.currency}
        for row in instr_rows
    }

    result = await session.execute(
        select(TransactionHistory)
        .order_by(TransactionHistory.timestamp.desc())
    )
    all_txns = result.scalars().all()

    # ── Aggregate summary (full history, unfiltered) ──────────────────────────
    total_dividends = 0.0
    total_realized_gains = 0.0
    total_fees = 0.0
    total_deposited = 0.0
    total_interest = 0.0

    dividends_by_month: dict[str, float] = defaultdict(float)
    dividends_by_ticker: dict[str, dict] = {}
    years_set: set[int] = set()

    for txn in all_txns:
        years_set.add(txn.timestamp.year)
        fees = _fees_gbp(txn.fees, txn.exchange_rate)
        total_fees += fees

        if txn.action in _DIVIDEND_ACTIONS:
            total_dividends += txn.total
            dividends_by_month[txn.timestamp.strftime("%Y-%m")] += txn.total

            if txn.ticker:
                if txn.ticker not in dividends_by_ticker:
                    dividends_by_ticker[txn.ticker] = {
                        "ticker": txn.ticker,
                        "name": instrument_map.get(txn.ticker, {}).get("name") or txn.ticker,
                        "total": 0.0,
                        "count": 0,
                    }
                dividends_by_ticker[txn.ticker]["total"] += txn.total
                dividends_by_ticker[txn.ticker]["count"] += 1

        elif txn.action == TransactionAction.DEPOSIT:
            total_deposited += txn.total

        elif txn.action == TransactionAction.INTEREST:
            total_interest += txn.total

        elif txn.action in _SELL_ACTIONS and txn.result is not None:
            total_realized_gains += txn.result

    # Build dividend chart — all months in ascending order (frontend trims to 24)
    dividends_chart = [
        {"month": m, "amount": round(v, 2)}
        for m, v in sorted(dividends_by_month.items())
    ]

    # Top payers sorted by cumulative total
    top_dividend_payers = sorted(
        [{"ticker": t, "name": d["name"], "total": round(d["total"], 2), "count": d["count"]}
         for t, d in dividends_by_ticker.items()],
        key=lambda x: x["total"],
        reverse=True,
    )

    # ── Serialize transactions ────────────────────────────────────────────────
    txn_list = []
    for txn in all_txns:
        fees = _fees_gbp(txn.fees, txn.exchange_rate)
        instr = instrument_map.get(txn.ticker) if txn.ticker else None

        txn_list.append({
            "id": txn.id,
            "date": txn.timestamp.isoformat(),
            "ticker": txn.ticker,
            "name": instr["name"] if instr else None,
            "action": txn.action.value,
            "quantity": txn.quantity,
            "price": txn.price,
            "currency": instr["currency"] if instr else None,  # native stock currency
            "total": round(txn.total, 2),     # always GBP
            "exchange_rate": txn.exchange_rate,
            "result": round(txn.result, 2) if txn.result is not None else None,
            "fees": round(fees, 2),           # always GBP (converted via _fees_gbp)
            "notes": txn.notes,
        })

    return {
        "summary": {
            "total_dividends": round(total_dividends, 2),
            "total_realized_gains": round(total_realized_gains, 2),
            "total_fees": round(total_fees, 2),
            "total_deposited": round(total_deposited, 2),
            "total_interest": round(total_interest, 2),
        },
        "dividends_chart": dividends_chart,
        "top_dividend_payers": top_dividend_payers,
        "transactions": txn_list,
        "years": sorted(years_set, reverse=True),
        "total_count": len(txn_list),
    }
