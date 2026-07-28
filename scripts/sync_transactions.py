"""
Sync transaction history from Trading212 API.

Fetches paginated history from three T212 REST endpoints and upserts into the
TransactionHistory table. Uses the API's ``reference`` field as the dedup key
(stored as csv_id), so re-runs are fully idempotent.

T212 history endpoints:
  GET /api/v0/equity/history/orders        — buy / sell orders (fills)
  GET /api/v0/equity/history/transactions  — cash movements (deposit/withdraw/fees/transfers)
  GET /api/v0/equity/history/dividends     — dividend payments

All three endpoints return items newest-first. Pagination stops once items
older than the configured cutoff are encountered, keeping routine syncs to
1–2 pages per endpoint.

This supplements (not replaces) the CSV import; CSV-imported records keep
their existing csv_ids and are not touched.

Schedule: nightly at 05:00 UTC via Celery beat (see celery_tasks/).
"""

import argparse
import logging
import requests
import time
from datetime import datetime, timezone
from typing import Optional

from config import TRADING212_API_BASE, TRADING212_API_KEY, logger
from models import Instrument, TransactionAction, TransactionHistory
from scripts.update_data import get_session
from sqlalchemy import func, select

# ── T212 API config ────────────────────────────────────────────────────────────

_HEADERS = {"Authorization": TRADING212_API_KEY}
_TIMEOUT = 20

# Proactive delay between page requests. T212 history endpoints enforce a
# rate limit; Retry-After headers show 5–7s in practice.
_PAGE_DELAY = 10.0

# Minimum floor for 429 Retry-After to avoid tight loops when header says 0s.
_MIN_RETRY_WAIT = 5

# Prefix used for csv_id values written by this script. Rows originating
# elsewhere (CSV importer, manual inserts) carry any other reference string or
# NULL. Content-based dedup skips matches with this prefix so legitimate
# multi-fill orders (same ticker/timestamp/qty/total, distinct fill.id) are
# kept — fill uniqueness within the API source is handled by uq_transaction_csv_id.
_API_CSV_ID_PREFIX = "api:"

# Tolerances for float comparison in the semantic-duplicate check. T212
# re-exports the same fill with tiny quantity-rounding drift between export
# dates (observed 1.004257 vs 1.004029 shares for a single Stellantis buy).
_QTY_TOLERANCE = 1e-4  # ~4 decimals
_TOTAL_TOLERANCE = 1e-2  # ~2 decimals (pence)

# ── Action mappings (by endpoint) ──────────────────────────────────────────────

# /api/v0/equity/history/transactions — cash movements (no ticker/quantity/price, use ``amount`` for GBP amount):
_CASH_ACTION_MAP: dict[str, TransactionAction] = {
    # Trading212 docs currently use: WITHDRAW, DEPOSIT, FEE, TRANSFER, ...
    "DEPOSIT": TransactionAction.DEPOSIT,
    "WITHDRAW": TransactionAction.WITHDRAWAL,
    # Observed / legacy variants
    "WITHDRAWAL": TransactionAction.WITHDRAWAL,
    "INTEREST_ON_CASH": TransactionAction.INTEREST,
    "INTEREST": TransactionAction.INTEREST,
}

# /api/v0/equity/history/dividends — dividend type field
_DIVIDEND_TYPE_MAP: dict[str, TransactionAction] = {
    "ORDINARY": TransactionAction.DIVIDEND,
    "PROPERTY_INCOME_DISTRIBUTION": TransactionAction.DIVIDEND_PROPERTY,
    "TAX_EXEMPT": TransactionAction.DIVIDEND_TAX_EXEMPT,
}

_ORDER_ACTIONS = (
    TransactionAction.MARKET_BUY,
    TransactionAction.LIMIT_BUY,
    TransactionAction.MARKET_SELL,
    TransactionAction.LIMIT_SELL,
)
_CASH_ACTIONS = (
    TransactionAction.DEPOSIT,
    TransactionAction.WITHDRAWAL,
    TransactionAction.INTEREST,
)
_DIVIDEND_ACTIONS = tuple(_DIVIDEND_TYPE_MAP.values())

# The history API has no interest type — interest-on-cash arrives as "DEPOSIT".
# T212's minimum deposit is £1, so anything smaller is interest, not a subscription.
_MIN_DEPOSIT_GBP = 1.0

# ── Helpers ────────────────────────────────────────────────────────────────────


def _get(url: str) -> Optional[dict]:
    """GET a T212 URL with retry on 429 rate-limit.

    Returns None on 404 (expired/invalid cursor — treated as end of
    pagination). Raises on all other HTTP errors.
    """
    for attempt in range(4):
        r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if r.status_code == 429:
            wait = max(int(r.headers.get("Retry-After", _MIN_RETRY_WAIT)), _MIN_RETRY_WAIT)
            logger.warning("T212 rate limited — waiting %ds before retry %d/4", wait, attempt + 1)
            time.sleep(wait)
            continue
        if r.status_code == 404:
            logger.warning("T212 404 for %s — cursor expired, stopping pagination", url)
            return None
        r.raise_for_status()
        return r.json()
    raise requests.HTTPError(f"Still rate-limited after 4 attempts: {url}")


def _parse_dt(s: str) -> datetime:
    """Parse a T212 ISO-8601 timestamp to a naive UTC datetime."""
    s = s.replace("+0000", "+00:00").replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except ValueError:
        return datetime.fromisoformat(s[:19])


def _item_date(item: dict) -> Optional[str]:
    """Return the date string from a T212 item, trying all known field names.

    orders       → ``fill.filledAt`` or ``order.createdAt``
    transactions → ``dateTime``
    dividends    → ``paidOn``
    """
    fill = item.get("fill") if isinstance(item.get("fill"), dict) else None
    order = item.get("order") if isinstance(item.get("order"), dict) else None
    return (
        (fill.get("filledAt") if fill else None)
        or (order.get("createdAt") if order else None)
        or item.get("dateTime")
        or item.get("paidOn")
    )


def _build_fees(item: dict, instrument_currency: Optional[str]) -> Optional[list[dict]]:
    """Convert T212 API taxes/fees objects into our standard JSONB fee format.

    The newer equity/history endpoints place costs under:
      fill.walletImpact.taxes: [{name, chargedAt, quantity? / amount? / value?}, ...]

    Convention (mirrors CSV import):
      WITHHOLDING_TAX → instrument native currency (e.g. USD)
      all others      → GBP
    """
    all_fee_objs: list[dict] = []

    fill = item.get("fill") if isinstance(item.get("fill"), dict) else None
    wallet = fill.get("walletImpact") if (fill and isinstance(fill.get("walletImpact"), dict)) else None
    taxes = wallet.get("taxes") if (wallet and isinstance(wallet.get("taxes"), list)) else None
    if taxes:
        all_fee_objs.extend([t for t in taxes if isinstance(t, dict)])

    fees = []
    for f in all_fee_objs:
        name = str(f.get("name") or "UNKNOWN_FEE")
        raw_qty = f.get("quantity")
        if raw_qty is None:
            raw_qty = f.get("amount")
        if raw_qty is None:
            raw_qty = f.get("value")
        try:
            quantity = float(raw_qty) if raw_qty is not None else 0.0
        except (TypeError, ValueError):
            quantity = 0.0
        if quantity == 0.0:
            continue

        fee_currency = instrument_currency if (name == "WITHHOLDING_TAX" and instrument_currency) else "GBP"
        fees.append(
            {
                "name": name,
                "quantity": -abs(quantity),  # always negative (cost)
                "currency": fee_currency,
                "timeCharged": str(f.get("chargedAt") or f.get("timeCharged") or ""),
            }
        )

    return fees if fees else None


def _find_semantic_duplicate(
    session,
    *,
    action: TransactionAction,
    ticker: Optional[str],
    timestamp: datetime,
    quantity: float,
    total: float,
) -> Optional[TransactionHistory]:
    """Look for a pre-existing non-API row representing the same transaction.

    T212 returns the same event with different reference strings across export
    sources, so the ``csv_id`` unique constraint alone can't catch cross-source
    duplicates. We additionally match on (action, ticker, timestamp) with a
    loose tolerance on quantity and total to absorb per-export rounding drift.

    Rows whose ``csv_id`` starts with ``api:`` are deliberately skipped:
    legitimate multi-fill orders arrive with distinct ``fill.id`` values at the
    same second and must both be kept; the ``uq_transaction_csv_id`` constraint
    already guarantees uniqueness within the API source.
    """
    stmt = select(TransactionHistory).where(
        TransactionHistory.action == action,
        TransactionHistory.timestamp == timestamp,
    )
    stmt = stmt.where(TransactionHistory.ticker.is_(None) if ticker is None else TransactionHistory.ticker == ticker)

    for row in session.execute(stmt).scalars():
        if row.csv_id and row.csv_id.startswith(_API_CSV_ID_PREFIX):
            continue
        if abs((row.quantity or 0.0) - quantity) > _QTY_TOLERANCE:
            continue
        if abs((row.total or 0.0) - total) > _TOTAL_TOLERANCE:
            continue
        return row
    return None


# ── Pagination ─────────────────────────────────────────────────────────────────


def _fetch_all_pages(path: str, stop_before: Optional[datetime] = None) -> list[dict]:
    """Paginate through a T212 history endpoint and return all collected items.

    T212 returns items newest-first. When ``stop_before`` is provided, pagination
    stops as soon as a page contains an item older than that datetime; only items
    newer than the cutoff from that page are kept. This limits routine syncs to
    1–2 pages per endpoint after the initial full import.

    T212's ``nextPagePath`` comes in three observed formats:
      Absolute URL  "https://live.trading212.com/api/v0/…"  → use as-is
      Root-relative "/api/v0/history/orders?cursor=…"       → prepend T212_BASE
      Query-only    "limit=20&cursor=…"                     → append to endpoint URL
    """
    endpoint_url = TRADING212_API_BASE + path
    current_url: Optional[str] = endpoint_url
    items: list[dict] = []
    page_num = 0

    while current_url:
        page_num += 1
        if page_num > 1:
            time.sleep(_PAGE_DELAY)

        data = _get(current_url)
        if data is None:
            break  # 404 — cursor expired, stop gracefully

        page_items = data.get("items") or []
        logger.debug("%s page %d: %d items", path, page_num, len(page_items))

        if stop_before and page_items:
            fresh = []
            reached_cutoff = False
            first_cutoff_item_dt: Optional[datetime] = None
            for item in page_items:
                date_str = _item_date(item)
                if date_str:
                    try:
                        item_dt = _parse_dt(date_str)
                        if item_dt < stop_before:
                            reached_cutoff = True
                            first_cutoff_item_dt = item_dt
                            break
                    except Exception:
                        pass  # keep item if date unparseable
                fresh.append(item)
            items.extend(fresh)
            if reached_cutoff:
                if len(items) == 0:
                    newest_dt: Optional[datetime] = None
                    newest_str = _item_date(page_items[0])
                    if newest_str:
                        try:
                            newest_dt = _parse_dt(newest_str)
                        except Exception:
                            newest_dt = None
                    logger.info(
                        "%s: no new items (newest=%s, cutoff=%s)",
                        path,
                        newest_dt.isoformat(sep=" ") if newest_dt else (newest_str or "unknown"),
                        stop_before.isoformat(sep=" "),
                    )
                else:
                    logger.info(
                        "%s: reached cutoff %s at item=%s after %d items",
                        path,
                        stop_before.isoformat(sep=" "),
                        first_cutoff_item_dt.isoformat(sep=" ") if first_cutoff_item_dt else "unknown",
                        len(items),
                    )
                break
        else:
            items.extend(page_items)

        next_page = data.get("nextPagePath") or None
        if not next_page:
            current_url = None
        elif next_page.startswith("http"):
            current_url = next_page
        elif next_page.startswith("/"):
            current_url = TRADING212_API_BASE + next_page
        else:
            # Bare query string — append to the endpoint path
            current_url = endpoint_url.split("?")[0] + "?" + next_page

    return items


# ── Import helpers ─────────────────────────────────────────────────────────────


def _resolve_ticker(
    t212_ticker: str,
    t212_to_yahoo: dict[str, str],
) -> Optional[str]:
    """Map a T212 API ticker code to a Yahoo symbol. Falls back to raw code."""
    if not t212_ticker:
        return None
    yahoo = t212_to_yahoo.get(t212_ticker)
    if not yahoo:
        logger.debug("No yahoo_symbol for T212 ticker %r — storing raw", t212_ticker)
        return t212_ticker
    return yahoo


def _order_action(order: dict) -> Optional[TransactionAction]:
    """Map an order (BUY/SELL + limit/stop fields) into our TransactionAction."""
    side = str(order.get("side") or "").upper()
    if side not in {"BUY", "SELL"}:
        return None

    # Prefer explicit type if present (observed in payloads).
    raw_type = str(order.get("type") or "").upper()
    if "LIMIT" in raw_type or "STOP" in raw_type:
        is_limit_like = True
    elif "MARKET" in raw_type:
        is_limit_like = False
    else:
        # Fallback heuristic: if any of limit/stop prices exist, treat it as LIMIT_*; else MARKET_*.
        is_limit_like = any(order.get(k) is not None for k in ("limitPrice", "stopPrice"))
    if side == "BUY":
        return TransactionAction.LIMIT_BUY if is_limit_like else TransactionAction.MARKET_BUY
    return TransactionAction.LIMIT_SELL if is_limit_like else TransactionAction.MARKET_SELL


def _import_orders(
    raw: list[dict],
    existing_ids: set[str],
    t212_to_yahoo: dict[str, str],
    t212_to_currency: dict[str, str],
    session,
    stats: dict,
) -> None:
    """Import buy/sell orders from /api/v0/equity/history/orders.

    Each item contains nested ``order`` and (optionally) ``fill``. We store fills
    (not the original order ticket) because that's the cash-impacting event.
    """
    for item in raw:
        try:
            order = item.get("order") if isinstance(item.get("order"), dict) else {}
            fill = item.get("fill") if isinstance(item.get("fill"), dict) else {}

            order_id = order.get("id")
            fill_id = fill.get("id")
            if order_id is None and fill_id is None:
                logger.debug("Order item missing order.id/fill.id — keys: %s", list(item.keys()))
                stats["errors"] += 1
                continue

            # Dedup key: stable and unique even for multi-fill orders
            csv_id = f"api:order:{order_id}:fill:{fill_id}" if fill_id is not None else f"api:order:{order_id}"
            if not csv_id:
                logger.warning("Order missing reference: %s", item)
                stats["errors"] += 1
                continue
            if csv_id in existing_ids:
                stats["order_skipped"] += 1
                continue

            action = _order_action(order)
            if action is None:
                logger.debug("Unknown order side — skipping. order keys: %s", list(order.keys()))
                stats["order_skipped"] += 1
                continue

            date_str = _item_date(item)
            if not date_str:
                logger.debug("Order %s missing date field — keys: %s", csv_id, list(item.keys()))
                stats["errors"] += 1
                continue

            t212_ticker = (order.get("ticker") or "").strip()
            yahoo_ticker = _resolve_ticker(t212_ticker, t212_to_yahoo)
            instrument_currency = t212_to_currency.get(t212_ticker) if t212_ticker else None

            is_sell = action in (TransactionAction.MARKET_SELL, TransactionAction.LIMIT_SELL)
            wallet = fill.get("walletImpact") if isinstance(fill.get("walletImpact"), dict) else {}
            exchange_rate = wallet.get("fxRate")
            realised = wallet.get("realisedProfitLoss")

            # Prefer walletImpact.netValue (account currency, typically GBP). Fall back to filledValue/filledQuantity.
            net_value = wallet.get("netValue")
            if net_value is None:
                net_value = order.get("filledValue")
            price = fill.get("price")
            qty = fill.get("quantity") or order.get("filledQuantity") or 0.0

            ts = _parse_dt(date_str)
            qty_abs = abs(qty or 0.0)
            total_abs = abs(float(net_value or 0.0))

            dup = _find_semantic_duplicate(
                session, action=action, ticker=yahoo_ticker, timestamp=ts, quantity=qty_abs, total=total_abs
            )
            if dup is not None:
                logger.info(
                    "Semantic duplicate skipped: %s %s at %s matches existing id=%d (csv_id=%s)",
                    action.value,
                    yahoo_ticker,
                    ts,
                    dup.id,
                    dup.csv_id,
                )
                stats["dedup_skipped"] += 1
                continue

            session.add(
                TransactionHistory(
                    csv_id=csv_id,
                    timestamp=ts,
                    ticker=yahoo_ticker,
                    isin=(order.get("instrument") or {}).get("isin")
                    if isinstance(order.get("instrument"), dict)
                    else None,
                    action=action,
                    quantity=qty_abs,
                    price=price if price is not None else None,
                    total=total_abs,
                    exchange_rate=exchange_rate,
                    result=float(realised) if (is_sell and realised is not None) else None,
                    fees=_build_fees(item, instrument_currency),
                    notes=None,
                )
            )
            existing_ids.add(csv_id)
            stats["order_imported"] += 1

        except Exception as exc:
            logger.error("Error importing order %s: %s", item.get("reference", "?"), exc)
            stats["errors"] += 1


def _import_cash(
    raw: list[dict],
    existing_ids: set[str],
    session,
    stats: dict,
) -> None:
    """Import cash movements from /api/v0/equity/history/transactions.

    Cash items have no ticker/quantity/price. Amount is in the ``amount`` field
    (not ``total``); currency is always GBP for a UK ISA account.
    """
    for item in raw:
        try:
            csv_id = item.get("reference", "")
            if not csv_id:
                logger.warning("Cash transaction missing reference: %s", item)
                stats["errors"] += 1
                continue
            if csv_id in existing_ids:
                stats["cash_skipped"] += 1
                continue

            api_type = item.get("type", "")
            action = _CASH_ACTION_MAP.get(api_type)
            if action is None:
                # Keep visibility into unhandled types without breaking the sync.
                logger.debug("Unknown cash transaction type %r — skipping. keys=%s", api_type, list(item.keys()))
                stats["cash_skipped"] += 1
                continue

            date_str = _item_date(item)
            if not date_str:
                logger.debug("Cash %s missing date field — keys: %s", csv_id, list(item.keys()))
                stats["errors"] += 1
                continue

            ts = _parse_dt(date_str)
            total_abs = abs(item.get("amount") or 0.0)  # cash uses 'amount', not 'total'

            if action == TransactionAction.DEPOSIT and total_abs < _MIN_DEPOSIT_GBP:
                action = TransactionAction.INTEREST

            dup = _find_semantic_duplicate(
                session, action=action, ticker=None, timestamp=ts, quantity=0.0, total=total_abs
            )
            if dup is not None:
                logger.info(
                    "Semantic duplicate skipped: %s at %s matches existing id=%d (csv_id=%s)",
                    action.value,
                    ts,
                    dup.id,
                    dup.csv_id,
                )
                stats["dedup_skipped"] += 1
                continue

            session.add(
                TransactionHistory(
                    csv_id=csv_id,
                    timestamp=ts,
                    ticker=None,
                    isin=None,
                    action=action,
                    quantity=0.0,
                    price=None,
                    total=total_abs,
                    exchange_rate=None,
                    result=None,
                    fees=None,
                    notes=None,
                )
            )
            existing_ids.add(csv_id)
            stats["cash_imported"] += 1

        except Exception as exc:
            logger.error("Error importing cash transaction %s: %s", item.get("reference", "?"), exc)
            stats["errors"] += 1


def _import_dividends(
    raw: list[dict],
    existing_ids: set[str],
    t212_to_yahoo: dict[str, str],
    t212_to_currency: dict[str, str],
    session,
    stats: dict,
) -> None:
    """Import dividend payments from /api/v0/equity/history/dividends."""
    for item in raw:
        try:
            csv_id = item.get("reference", "")
            if not csv_id:
                logger.warning("Dividend missing reference: %s", item)
                stats["errors"] += 1
                continue
            if csv_id in existing_ids:
                stats["div_skipped"] += 1
                continue

            date_str = _item_date(item)
            if not date_str:
                logger.debug("Dividend %s missing date field — keys: %s", csv_id, list(item.keys()))
                stats["errors"] += 1
                continue

            div_type = item.get("type", "ORDINARY")
            action = _DIVIDEND_TYPE_MAP.get(div_type, TransactionAction.DIVIDEND)

            instrument = item.get("instrument") if isinstance(item.get("instrument"), dict) else {}
            t212_ticker = (instrument.get("ticker") or item.get("ticker") or "").strip()
            yahoo_ticker = _resolve_ticker(t212_ticker, t212_to_yahoo)
            instrument_currency = t212_to_currency.get(t212_ticker) if t212_ticker else None

            # ``amount`` is the net GBP amount paid (after withholding tax).
            total_gbp = abs(item.get("amount") or item.get("paidQuantity") or 0.0)
            qty_abs = abs(item.get("quantity") or 0.0)
            ts = _parse_dt(date_str)

            dup = _find_semantic_duplicate(
                session, action=action, ticker=yahoo_ticker, timestamp=ts, quantity=qty_abs, total=total_gbp
            )
            if dup is not None:
                logger.info(
                    "Semantic duplicate skipped: %s %s at %s matches existing id=%d (csv_id=%s)",
                    action.value,
                    yahoo_ticker,
                    ts,
                    dup.id,
                    dup.csv_id,
                )
                stats["dedup_skipped"] += 1
                continue

            session.add(
                TransactionHistory(
                    csv_id=csv_id,
                    timestamp=ts,
                    ticker=yahoo_ticker,
                    isin=instrument.get("isin") or item.get("isin") or None,
                    action=action,
                    quantity=qty_abs,
                    price=item.get("grossAmountPerShare") or None,
                    total=total_gbp,
                    exchange_rate=None,
                    result=None,
                    fees=_build_fees(item, instrument_currency),
                    notes=None,
                )
            )
            existing_ids.add(csv_id)
            stats["div_imported"] += 1

        except Exception as exc:
            logger.error("Error importing dividend %s: %s", item.get("reference", "?"), exc)
            stats["errors"] += 1


# ── Main sync function ─────────────────────────────────────────────────────────


def sync_transactions(
    *,
    dry_run: bool = False,
    since: Optional[datetime] = None,
    full: bool = False,
) -> dict[str, int]:
    """Sync order, cash, and dividend history from T212 API into the DB."""
    logger.info("Starting T212 transaction sync from API")

    with get_session() as session:
        instruments = session.execute(select(Instrument)).scalars().all()
        t212_to_yahoo: dict[str, str] = {i.t212_code: i.yahoo_symbol for i in instruments if i.yahoo_symbol}
        t212_to_currency: dict[str, str] = {i.t212_code: i.currency for i in instruments}
        logger.info("Loaded %d instruments for ticker normalisation", len(t212_to_yahoo))

        existing_ids: set[str] = set(
            session.execute(select(TransactionHistory.csv_id).where(TransactionHistory.csv_id.isnot(None)))
            .scalars()
            .all()
        )
        logger.info("Pre-loaded %d existing csv_ids", len(existing_ids))

        latest_orders_ts: Optional[datetime] = session.execute(
            select(func.max(TransactionHistory.timestamp)).where(TransactionHistory.action.in_(_ORDER_ACTIONS))
        ).scalar()
        latest_cash_ts: Optional[datetime] = session.execute(
            select(func.max(TransactionHistory.timestamp)).where(TransactionHistory.action.in_(_CASH_ACTIONS))
        ).scalar()
        latest_div_ts: Optional[datetime] = session.execute(
            select(func.max(TransactionHistory.timestamp)).where(TransactionHistory.action.in_(_DIVIDEND_ACTIONS))
        ).scalar()

    if full:
        logger.info("Full import enabled — paginating until end")
        stop_before_orders = None
        stop_before_cash = None
        stop_before_divs = None
    elif since:
        logger.info("Since mode enabled — paginating until %s", since.date())
        stop_before_orders = since
        stop_before_cash = since
        stop_before_divs = since
    else:
        stop_before_orders = latest_orders_ts
        stop_before_cash = latest_cash_ts
        stop_before_divs = latest_div_ts
        logger.info(
            "Most recent DB records — orders: %s | cash: %s | dividends: %s",
            latest_orders_ts,
            latest_cash_ts,
            latest_div_ts,
        )

    # ── Fetch ──────────────────────────────────────────────────────────────────
    def _fetch(label: str, path: str, stop_before: Optional[datetime]) -> list[dict]:
        logger.info("Fetching %s", path)
        time.sleep(_PAGE_DELAY)
        try:
            items = _fetch_all_pages(path, stop_before=stop_before)
            logger.info("Fetched %d %s items", len(items), label)
            if items:
                logger.debug("%s sample keys: %s", label, list(items[0].keys()))
                if label == "orders":
                    o = items[0].get("order") if isinstance(items[0].get("order"), dict) else {}
                    f = items[0].get("fill") if isinstance(items[0].get("fill"), dict) else {}
                    logger.debug("orders sample order keys: %s", list(o.keys()))
                    logger.debug("orders sample fill keys: %s", list(f.keys()))
            return items
        except Exception as exc:
            logger.error("Failed to fetch %s: %s", path, exc)
            return []

    raw_orders = _fetch("orders", "/api/v0/equity/history/orders", stop_before_orders)
    raw_cash = _fetch("cash", "/api/v0/equity/history/transactions", stop_before_cash)
    raw_divs = _fetch("dividends", "/api/v0/equity/history/dividends", stop_before_divs)

    # ── Import ─────────────────────────────────────────────────────────────────
    stats: dict[str, int] = {
        "order_imported": 0,
        "order_skipped": 0,
        "cash_imported": 0,
        "cash_skipped": 0,
        "div_imported": 0,
        "div_skipped": 0,
        "dedup_skipped": 0,
        "errors": 0,
    }

    if dry_run:
        logger.info(
            "Dry-run enabled — would attempt to import: orders=%d cash=%d divs=%d",
            len(raw_orders),
            len(raw_cash),
            len(raw_divs),
        )
        return stats

    with get_session() as session:
        _import_orders(raw_orders, existing_ids, t212_to_yahoo, t212_to_currency, session, stats)
        _import_cash(raw_cash, existing_ids, session, stats)
        _import_dividends(raw_divs, existing_ids, t212_to_yahoo, t212_to_currency, session, stats)
        session.commit()

    total_imported = stats["order_imported"] + stats["cash_imported"] + stats["div_imported"]
    total_skipped = stats["order_skipped"] + stats["cash_skipped"] + stats["div_skipped"]
    logger.info(
        "T212 sync complete — imported: %d (orders: %d, cash: %d, divs: %d) | skipped: %d | dedup-skipped: %d | errors: %d",
        total_imported,
        stats["order_imported"],
        stats["cash_imported"],
        stats["div_imported"],
        total_skipped,
        stats["dedup_skipped"],
        stats["errors"],
    )
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Trading212 history from API into TransactionHistory.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch only; do not write to DB.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Fetch history back to (and including) this date (YYYY-MM-DD). Ignores latest-DB cutoff.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Fetch the full available history (can be slow due to rate limits).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    since_dt: Optional[datetime] = None
    if args.since:
        try:
            since_dt = datetime.strptime(args.since.strip(), "%Y-%m-%d")
        except ValueError as exc:
            raise SystemExit(f"Invalid --since date {args.since!r}, expected YYYY-MM-DD") from exc

    result = sync_transactions(
        dry_run=args.dry_run,
        since=since_dt,
        full=args.full,
    )
    logger.info("Sync result: %s", result)
