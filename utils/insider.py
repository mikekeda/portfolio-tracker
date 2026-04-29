"""Insider transaction aggregation helper.

Pure function that condenses a yfinance ``ticker.insider_transactions`` payload
into three trailing-90-day numbers persisted on
:class:`models.InstrumentMetricsDaily`. No DB or network I/O — callers pass the
already-scrubbed list of records.

yfinance column shape (as of 0.2.x): ``Shares, Value, URL, Text, Insider,
Position, Transaction, Start Date``. The buy/sell description lives in ``Text``
(e.g. ``"Sale at price 435.00 per share."``); the dedicated ``Transaction``
column is currently empty for every row, so we classify off ``Text`` and only
fall back to ``Transaction`` for older yfinance releases.
"""

from datetime import date, datetime, timedelta
from typing import Any, Optional, TypedDict


# Window length in days. Aligns with the Lakonishok-Lee insider-cluster studies
# (60-90d is the canonical "recent" window beyond which the signal decays).
WINDOW_DAYS = 90

# Net buyer/seller in this run isn't meaningful for the screener gates, but
# defining the threshold here keeps the per-row classification readable.
ZERO_TOLERANCE = 1e-6

# Yahoo's ``Text`` strings are inconsistent across tickers and over time.
# Match by lowercased substring rather than exact equality so we accept e.g.
# "Sale at price 12.34 per share.", "Purchase at price ...", "Sale (Multiple
# Prices)" without an explosion of literals. Anything not matched is treated
# as a non-open-market event (stock award/grant, option exercise, gift,
# vesting, conversion, disposition) and ignored.
_BUY_KEYWORDS = ("purchase", "buy", "acquired open market")
_SELL_KEYWORDS = ("sale", "sell", "sold open market")


class InsiderSignal(TypedDict):
    buy_count: int
    sell_count: int
    net_value: float


def _classify(transaction: Any) -> Optional[str]:
    """Return ``"buy"``, ``"sell"`` or ``None`` for non-open-market events."""
    if not isinstance(transaction, str):
        return None
    s = transaction.lower()
    # Order matters: "sale" appears in some "Stock Sale (Open Market)" strings,
    # check sell keywords before buy to avoid a "purchase" substring inside an
    # award description being classified as a buy.
    if any(k in s for k in _SELL_KEYWORDS):
        return "sell"
    if any(k in s for k in _BUY_KEYWORDS):
        return "buy"
    return None


def _parse_date(value: Any) -> Optional[date]:
    """Parse the ``Start Date`` column. yfinance returns it as a Pandas Timestamp;
    after ``scrub_for_json`` it becomes an ISO 8601 string. Be lenient about both."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            try:
                return datetime.strptime(value[:10], "%Y-%m-%d").date()
            except ValueError:
                return None
    return None


def _safe_float(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    if not isinstance(x, (int, float)):
        return None
    if x != x or x == float("inf") or x == float("-inf"):
        return None
    return float(x)


def _get(row: dict[str, Any], *keys: str) -> Any:
    """Return the first present value across a list of column-name synonyms."""
    for key in keys:
        if key in row:
            return row[key]
    return None


def compute_insider_signal(
    transactions: Optional[list[dict[str, Any]]],
    as_of: date,
) -> InsiderSignal:
    """Aggregate yfinance insider transactions over the trailing 90 days.

    Returns counts of distinct insiders who *net-bought* and *net-sold* over the
    window and the signed net value of all open-market trades (positive => net
    buying). Empty/malformed input returns zeros — callers can distinguish "no
    activity" from "no coverage" via NULL-vs-0 in the column itself, since
    ingestion writes NULL when the yfinance fetch fails.
    """
    cutoff = as_of - timedelta(days=WINDOW_DAYS)

    # Per-insider net value (sum of signed buy/sell values within window).
    per_insider: dict[str, float] = {}
    total_net = 0.0

    if not transactions:
        return {"buy_count": 0, "sell_count": 0, "net_value": 0.0}

    for row in transactions:
        if not isinstance(row, dict):
            continue

        # ``Text`` is the human-readable description (e.g. "Sale at price ...");
        # ``Transaction`` is currently empty in yfinance 0.2.x but we keep it as
        # a fallback in case a future release moves the description back.
        side = _classify(_get(row, "Text", "Transaction", "text", "transaction"))
        if side is None:
            continue

        tx_date = _parse_date(_get(row, "Start Date", "start_date", "Date Reported", "date"))
        if tx_date is None or tx_date < cutoff or tx_date > as_of:
            continue

        value = _safe_float(_get(row, "Value", "value"))
        if value is None:
            continue
        # Yahoo reports ``Value`` as a positive gross dollar amount; the sign
        # comes from ``Transaction``. Take abs() defensively in case a future
        # yfinance release switches to signed values.
        signed = abs(value) if side == "buy" else -abs(value)

        insider = _get(row, "Insider", "insider")
        # Group by insider name; fall back to a synthetic key per row so that
        # anonymous filings still contribute to ``net_value`` but don't share
        # a bucket and skew the distinct-buyer count.
        key = insider.strip() if isinstance(insider, str) and insider.strip() else f"__row_{id(row)}"

        per_insider[key] = per_insider.get(key, 0.0) + signed
        total_net += signed

    buy_count = sum(1 for v in per_insider.values() if v > ZERO_TOLERANCE)
    sell_count = sum(1 for v in per_insider.values() if v < -ZERO_TOLERANCE)

    return {
        "buy_count": buy_count,
        "sell_count": sell_count,
        "net_value": total_net,
    }
