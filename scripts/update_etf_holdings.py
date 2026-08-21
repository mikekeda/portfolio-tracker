"""
scripts/update_etf_holdings.py
==================================
Refresh the etf_holdings table from issuer holdings files.

The look-through risk model needs to know what sits inside each held ETF. Only
some issuers publish that machine-readably, so each fund declares its source in
`ETF_HOLDING_SOURCES` and this script fetches accordingly:

  dws      Xtrackers' product-page API, the one the site's own holdings table
           calls. ISIN-keyed.
  ishares  BlackRock's product-data API. Also usable for another issuer's fund
           on the same index, where that issuer publishes nothing.
  ssga     State Street's daily holdings XLSX. ISIN-keyed.
  sp500    No issuer file exists for VUAG, so weights are derived by
           cap-weighting the SP500 list. Valid only for an uncapped
           full-replication tracker — see the note in data.py.

Every published holding is stored, whether or not it maps to a tracked
instrument; `instrument_id` is re-resolved on each run, so coverage improves as
instruments are added without re-fetching. A fund whose fetch fails keeps its
previous snapshot. Runs weekly from Celery, or by hand from the project root:

    python scripts/update_etf_holdings.py
"""

import asyncio
import re
import zipfile
from datetime import date
from collections import namedtuple
from io import BytesIO
from xml.etree import ElementTree as ET

import requests
from sqlalchemy import delete, func, or_, select, update

from backend.app import get_session
from config import logger
from data import ETF_HOLDING_SOURCES, SP500
from models import EtfHolding, Instrument, InstrumentYahoo

# One published holding, before it is matched to anything we track.
Row = namedtuple("Row", "key name weight_pct")

ISHARES_URL = (
    "https://www.blackrock.com/varnish-api/uk-retail01-product-data/product-data/api/v1/"
    "get-product-data?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=ishares-uk"
    "&locale=en_GB&userType=individual&component=holdings&portfolioId={portfolio_id}"
)
DWS_URL = "https://etf.dws.com/api/pdp/en-gb/etf/{slug}/holdings"
# Xtrackers also offers the same list as a spreadsheet, if the API ever moves:
# https://etf.dws.com/etfdata/export/GBR/ENG/excel/product/constituent/{isin}/
SSGA_URL = (
    "https://www.ssga.com/library-content/products/fund-data/etfs/emea/"
    "holdings-daily-emea-en-{ticker}.xlsx"
)
# Both issuers 403 a default user agent.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    )
}
TIMEOUT = 60
OTHER_KEY = "Other"
# Below this a line is rounding error in a fund that is itself a few percent of
# the book; risk.py applies its own, stricter materiality floor when modelling.
MIN_WEIGHT_PCT = 0.01
# Yahoo repeats the whole-company marketCap on every share class, so a dual
# listing would double-count. Weights are the classes' published index split.
DUAL_CLASS_SPLIT = {("GOOGL", "GOOG"): (3.26, 2.59), ("FOXA", "FOX"): (0.55, 0.45),
                    ("NWSA", "NWS"): (0.6, 0.4)}
DROP_TICKERS = {"BRK-A"}
XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}\d$")
FETCHERS = {
    "dws": lambda spec, caps: fetch_dws(spec["slug"], caps),
    "ishares": lambda spec, caps: fetch_ishares(spec["portfolio_id"], caps),
    "ssga": lambda spec, caps: fetch_ssga(spec["ticker"], caps),
    "sp500": lambda spec, caps: derive_sp500(caps),
}
# Depository receipts are equity exposure; cash and futures are not.
DWS_EQUITY_CLASSES = {"Equities", "Depository Receipts"}


def _dedupe(rows) -> list[Row]:
    """Collapse repeated issuer keys, which the unique constraint would reject.

    No issuer currently repeats one, but a second line for the same security is
    an ordinary way to express a partial position.
    """
    merged: dict[str, Row] = {}
    for r in rows:
        prior = merged.get(r.key)
        merged[r.key] = Row(r.key, prior.name if prior else r.name,
                            (prior.weight_pct if prior else 0.0) + r.weight_pct)
    return list(merged.values())


def _get(url: str) -> requests.Response:
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    return response


def fetch_ishares(portfolio_id: str, _isins: dict[str, str]) -> list[Row]:
    """Equity lines from BlackRock's product-data API, keyed by ISIN.

    Cash, futures and collateral rows are dropped: they are not look-through
    exposures and belong in the residual.
    """
    columns = _get(ISHARES_URL.format(portfolio_id=portfolio_id)).json()["productData"]["all"]
    out = []
    for isin, ticker, name, weight, asset_class in zip(
        columns["isin"]["value"],
        columns["ticker"]["value"],
        columns["issueName"]["value"],
        columns["holdingPercent"]["value"],
        columns["assetClass"]["value"],
    ):
        if asset_class == "Equity" and (isin or ticker):
            out.append(Row(isin or ticker, name, float(weight)))
    return out


def fetch_dws(slug: str, _isins: dict[str, str]) -> list[Row]:
    """Equity lines from the Xtrackers product-page API, keyed by ISIN.

    Column keys are positional (`column_1` is the weight), and `sortValue`
    carries the unrounded float that the rendered "18.708%" is formatted from.
    """
    table = _get(DWS_URL.format(slug=slug)).json()["tables"][0]
    return [
        Row(row["header"]["value"], row["column_0"]["value"], float(row["column_1"]["sortValue"]))
        for row in table["values"]
        if row.get("column_5", {}).get("value") in DWS_EQUITY_CLASSES
    ]


def _xlsx_rows(content: bytes) -> list[dict[str, str | None]]:
    """Column-letter-keyed rows from the first sheet, without openpyxl."""
    archive = zipfile.ZipFile(BytesIO(content))
    shared: list[str] = []
    if "xl/sharedStrings.xml" in archive.namelist():
        for item in ET.fromstring(archive.read("xl/sharedStrings.xml")):
            shared.append("".join(t.text or "" for t in item.iter(XLSX_NS + "t")))
    rows = []
    for row in ET.fromstring(archive.read("xl/worksheets/sheet1.xml")).iter(XLSX_NS + "row"):
        cells: dict[str, str | None] = {}
        for cell in row.iter(XLSX_NS + "c"):
            column = re.match(r"([A-Z]+)", cell.get("r") or "A").group(1)
            value = cell.find(XLSX_NS + "v")
            if value is None:
                cells[column] = None
            elif cell.get("t") == "s":
                cells[column] = shared[int(value.text)]
            else:
                cells[column] = value.text
        rows.append(cells)
    return rows


def fetch_ssga(ticker: str, _isins: dict[str, str]) -> list[Row]:
    """Equity lines from State Street's daily XLSX (A=ISIN, C=name, F=percent)."""
    out = []
    for cells in _xlsx_rows(_get(SSGA_URL.format(ticker=ticker)).content):
        isin, percent = cells.get("A"), cells.get("F")
        if not isin or not percent or not ISIN_RE.match(str(isin)):
            continue
        try:
            out.append(Row(isin, cells.get("C"), float(percent)))
        except ValueError:
            continue
    return out


def derive_sp500(market_caps: dict[str, float]) -> list[Row]:
    """Cap-weighted S&P 500, standing in for an unpublished holdings file.

    The only ticker-keyed source: there is no issuer file, so rows carry the
    Yahoo symbol as their key rather than an ISIN.
    """
    caps = {t: market_caps[t] for t in SP500 if t not in DROP_TICKERS and market_caps.get(t)}
    for (primary, secondary), (a, b) in DUAL_CLASS_SPLIT.items():
        if primary in caps and secondary in caps:
            whole = caps[primary]
            caps[primary], caps[secondary] = whole * a / (a + b), whole * b / (a + b)
    total = sum(caps.values())
    return [Row(t, None, 100.0 * v / total) for t, v in caps.items()]


async def _resolve(session, rows: list[Row]) -> dict[str, int]:
    """Map issuer keys to instrument ids, by ISIN first and then Yahoo symbol.

    Re-derived on every run, so a constituent added to `instruments` between
    refreshes resolves itself without anyone re-fetching the issuer.
    """
    keys = {r.key for r in rows}
    found = (
        await session.execute(
            select(Instrument.id, Instrument.isin, Instrument.yahoo_symbol).where(
                or_(Instrument.isin.in_(keys), Instrument.yahoo_symbol.in_(keys))
            )
        )
    ).all()
    out: dict[str, int] = {}
    for iid, isin, symbol in found:
        if isin in keys:
            out[isin] = iid
        if symbol in keys:
            out.setdefault(symbol, iid)
    return out


async def _latest_snapshot(session, etf_instrument_id: int) -> tuple[date | None, dict[str, float]]:
    """The most recent stored composition for a fund, for change detection."""
    latest = (
        await session.execute(
            select(func.max(EtfHolding.date)).where(
                EtfHolding.etf_instrument_id == etf_instrument_id
            )
        )
    ).scalar_one_or_none()
    if latest is None:
        return None, {}
    rows = (
        await session.execute(
            select(EtfHolding.source_key, EtfHolding.weight_pct).where(
                EtfHolding.etf_instrument_id == etf_instrument_id,
                EtfHolding.date == latest,
            )
        )
    ).all()
    return latest, {k: w for k, w in rows}


async def _touch_and_reresolve(session, etf_instrument_id: int, when: date,
                               resolved: dict[str, int]) -> int:
    """Re-point instrument_id on an existing snapshot, and stamp updated_at.

    An unchanged index still needs this: a constituent added to `instruments`
    since the last reconstitution would otherwise stay unresolved until the fund
    next rebalances. Every row is written, not just the changed ones, so
    updated_at is a reliable "last time we checked" for the staleness audit.
    """
    rows = (
        await session.execute(
            select(EtfHolding.id, EtfHolding.source_key, EtfHolding.instrument_id).where(
                EtfHolding.etf_instrument_id == etf_instrument_id, EtfHolding.date == when
            )
        )
    ).all()
    payload = [{"id": rid, "instrument_id": resolved.get(key)} for rid, key, _ in rows]
    if payload:
        await session.execute(update(EtfHolding), payload)
    return sum(1 for _, key, iid in rows if resolved.get(key) != iid)


async def _market_caps(session) -> dict[str, float]:
    rows = (
        await session.execute(
            select(Instrument.yahoo_symbol, InstrumentYahoo.info["marketCap"].astext).join(
                InstrumentYahoo, InstrumentYahoo.instrument_id == Instrument.id
            )
        )
    ).all()
    return {s: float(m) for s, m in rows if s and m}


async def refresh() -> int:
    today = date.today()
    failures = 0

    async with get_session() as session:
        funds = {
            s: i
            for i, s in (
                await session.execute(
                    select(Instrument.id, Instrument.yahoo_symbol).where(
                        Instrument.yahoo_symbol.in_(list(ETF_HOLDING_SOURCES))
                    )
                )
            ).all()
        }
        caps = await _market_caps(session)

        for symbol, spec in ETF_HOLDING_SOURCES.items():
            if spec is None:
                logger.info("%s: no issuer source, skipped", symbol)
                continue
            if symbol not in funds:
                failures += 1
                logger.error("%s: no Instrument row, cannot store constituents", symbol)
                continue
            try:
                fetch = FETCHERS[spec["kind"]]
                rows = fetch(spec, caps)
            except Exception as e:  # noqa: BLE001 — one dead issuer must not block the rest
                failures += 1
                logger.error("%s: fetch failed (%s), previous snapshot kept", symbol, e)
                continue

            rows = _dedupe(r for r in rows if r.weight_pct >= MIN_WEIGHT_PCT and r.key)
            listed = sum(r.weight_pct for r in rows)
            if not rows or listed > 100.5:
                failures += 1
                logger.error("%s: implausible weights (%d rows, %.2f%%), kept", symbol, len(rows), listed)
                continue

            resolved = await _resolve(session, rows)
            matched = sum(r.weight_pct for r in rows if r.key in resolved)
            logger.info(
                "%s: %d rows %.2f%% published, %d resolved to instruments (%.2f%%)",
                symbol, len(rows), listed, sum(1 for r in rows if r.key in resolved), matched,
            )

            # Weekly runs on a quarterly-rebalancing index would otherwise store the
            # same composition over and over; only a real change earns a snapshot.
            incoming = {r.key: round(r.weight_pct, 4) for r in rows}
            previous_date, previous = await _latest_snapshot(session, funds[symbol])
            if incoming == previous:
                changed = await _touch_and_reresolve(session, funds[symbol], previous_date, resolved)
                await session.commit()
                logger.info("%s: unchanged since %s, %d newly resolved", symbol, previous_date, changed)
                continue
            if previous:
                added, dropped = set(incoming) - set(previous), set(previous) - set(incoming)
                logger.info(
                    "%s: %d added, %d dropped, %d reweighted vs %s",
                    symbol, len(added), len(dropped),
                    sum(1 for k in set(incoming) & set(previous) if incoming[k] != previous[k]),
                    previous_date,
                )

            # Same-day re-run replaces its own snapshot; older ones are history.
            await session.execute(
                delete(EtfHolding).where(
                    EtfHolding.etf_instrument_id == funds[symbol], EtfHolding.date == today
                )
            )
            session.add_all(
                EtfHolding(
                    etf_instrument_id=funds[symbol],
                    date=today,
                    source_key=r.key,
                    name=(r.name or None),
                    weight_pct=round(r.weight_pct, 4),
                    instrument_id=resolved.get(r.key),
                )
                for r in rows
            )
            await session.commit()

    logger.info("Done: %d fund(s) failed", failures)
    return 1 if failures else 0


def main() -> int:
    return asyncio.run(refresh())


if __name__ == "__main__":
    raise SystemExit(main())
