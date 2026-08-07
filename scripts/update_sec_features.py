"""Fetch SEC companyfacts and backfill sec_features_daily (CIK instruments).

Filing-driven by default, decided locally: refreshes when a newer EarningsReport
row exists than the cached blob, or the blob is older than the refresh floor —
no submissions request per instrument. Use --force-fetch to re-pull all, and
--rebuild to recompute rows from cached facts after a logic change.

Backfill writes month-end PIT snapshots (filed < as_of) for statement Tier B
columns only — never screener_score. See CLAUDE.md / STRATEGY caveats:
backfilled-era composite is ~68% Tier A; quality gate is ROIC-only; ~23-name
quarterly universe.

    python scripts/update_sec_features.py
    python scripts/update_sec_features.py --force-fetch --backfill-start 2018-01-01
"""

from __future__ import annotations

import argparse
import math
from datetime import date, datetime, timedelta
from typing import Any

import requests
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from backend.utils.piotroski import get_piotroski_f_score
from backend.utils.sec_companyfacts import (
    CompanyFactsUnavailable,
    as_restated_roic_payload,
    build_statement_pair,
    compute_sec_feature_snapshot,
    concept_series,
    pad_cik,
    rate_limited_get,
    share_count_for_fscore,
)
from backend.utils.splits import adjust_share_count
from config import SEC_USER_AGENT, TIMEZONE, logger
from models import (
    EarningsReport,
    Instrument,
    InstrumentYahoo,
    PricesDaily,
    SecCompanyFacts,
    SecFeaturesDaily,
)
from scripts.update_data import get_session

# Refresh floor for names the earnings job never files a report for, so a stale
# blob still turns over eventually without a nightly submissions request.
COMPANYFACTS_REFRESH_FLOOR_DAYS = 90


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _month_ends(start: date, end: date) -> list[date]:
    """Last calendar day of each month in [start, end]."""
    out: list[date] = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        if m == 12:
            last = date(y, 12, 31)
            y, m = y + 1, 1
        else:
            last = date(y, m + 1, 1) - timedelta(days=1)
            m += 1
        if start <= last <= end:
            out.append(last)
    return out


def _adj_close_on(session: Session, symbol: str, as_of: date) -> float | None:
    """Most recent adj close on or before as_of."""
    return session.execute(
        select(PricesDaily.adj_close_price)
        .where(PricesDaily.symbol == symbol, PricesDaily.date <= as_of)
        .order_by(PricesDaily.date.desc())
        .limit(1)
    ).scalar_one_or_none()


def _f_score_split_safe(
    facts: dict[str, Any],
    splits: dict[str, Any] | None,
    *,
    as_of: date,
) -> int | None:
    """Piotroski F-Score with share counts on a common split-adjusted basis."""
    balance_sheet, income_stmt = build_statement_pair(facts, vintage="pit", annual=True)
    # cashflow annual
    ocf = concept_series(facts, "ocf", vintage="pit", annual=True)
    cashflow: dict[str, dict[str, float]] = {}
    for end, point in ocf.items():
        if point.filed < as_of:
            cashflow[end.isoformat()] = {"Operating Cash Flow": point.value}

    # Filter BS/IS to filed < as_of
    op = concept_series(facts, "operating_income", vintage="pit", annual=True)
    assets = concept_series(facts, "assets", vintage="pit", annual=True, instant=True)

    def known(end_iso: str) -> bool:
        end = date.fromisoformat(end_iso)
        filed = []
        if end in op:
            filed.append(op[end].filed)
        if end in assets:
            filed.append(assets[end].filed)
        return bool(filed) and max(filed) < as_of

    balance_sheet = {k: v for k, v in balance_sheet.items() if known(k)}
    income_stmt = {k: v for k, v in income_stmt.items() if known(k)}
    if len(balance_sheet) < 2 or len(income_stmt) < 2:
        return None

    # Overwrite share lines with split-adjusted PIT counts for dilution check.
    for end_iso, row in balance_sheet.items():
        end = date.fromisoformat(end_iso)
        sc = share_count_for_fscore(facts, vintage="pit", period_end=end, as_of=as_of)
        if sc is None:
            continue
        adj = adjust_share_count(sc.value, splits, sc.ref_date)
        if adj is not None:
            row["Ordinary Shares Number"] = adj
            row["Share Issued"] = adj

    result = get_piotroski_f_score(cashflow or None, balance_sheet, income_stmt)
    return int(result["score"]) if result and result["available"] else None


def _should_fetch(session: Session, instrument_id: int, force: bool) -> bool:
    """Whether to re-pull companyfacts, decided locally — no SEC request.

    The 04:00 earnings job already records when a filing landed, so a newer
    EarningsReport row is the refresh trigger.
    """
    if force:
        return True
    row = session.get(SecCompanyFacts, instrument_id)
    if row is None:
        return True

    fetched = row.fetched_at
    if (datetime.now(TIMEZONE).replace(tzinfo=None) - fetched).days >= COMPANYFACTS_REFRESH_FLOOR_DAYS:
        return True

    newest_report = session.execute(
        select(func.max(EarningsReport.created_at)).where(EarningsReport.instrument_id == instrument_id)
    ).scalar()
    return newest_report is not None and newest_report > fetched


def upsert_companyfacts(session: Session, instrument_id: int, cik: str, facts: dict[str, Any]) -> None:
    """Store/replace the companyfacts blob and cached as-restated ROIC series."""
    now = datetime.now(TIMEZONE).replace(tzinfo=None)
    roic_hist = as_restated_roic_payload(facts)
    stmt = pg_insert(SecCompanyFacts).values(
        instrument_id=instrument_id,
        cik=pad_cik(cik),
        facts=facts,
        roic_as_restated=roic_hist,
        fetched_at=now,
        updated_at=now,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["instrument_id"],
        set_={
            "cik": pad_cik(cik),
            "facts": facts,
            "roic_as_restated": roic_hist,
            "fetched_at": now,
            "updated_at": now,
        },
    )
    session.execute(stmt)


def backfill_instrument(
    session: Session,
    instrument_id: int,
    symbol: str,
    facts: dict[str, Any],
    splits: dict[str, Any] | None,
    start: date,
    end: date,
) -> int:
    """Write month-end sec_features_daily rows; returns rows upserted."""
    rows: list[dict[str, Any]] = []
    for as_of in _month_ends(start, end):
        adj = _adj_close_on(session, symbol, as_of)
        snap = compute_sec_feature_snapshot(facts, as_of=as_of, splits=splits, adj_close=adj, vintage="pit")
        if snap is None:
            continue
        f_score = _f_score_split_safe(facts, splits, as_of=as_of)
        period_end = date.fromisoformat(snap["period_end"]) if snap.get("period_end") else None
        rows.append(
            {
                "instrument_id": instrument_id,
                "date": as_of,
                "roic": _num(snap.get("roic")),
                "roic_ttm": _num(snap.get("roic_ttm")),
                "roic_ttm_trend": _num(snap.get("roic_ttm_trend")),
                "gross_margin": _num(snap.get("gross_margin")),
                "operating_margin": _num(snap.get("operating_margin")),
                "profit_margin": _num(snap.get("profit_margin")),
                "revenue_growth": _num(snap.get("revenue_growth")),
                "fcf_yield": _num(snap.get("fcf_yield")),
                "debt_to_equity": _num(snap.get("debt_to_equity")),
                "rule_of_40": _num(snap.get("rule_of_40")),
                "f_score": f_score,
                "gross_margin_trend_4q": _num(snap.get("gross_margin_trend_4q")),
                "operating_margin_trend_4q": _num(snap.get("operating_margin_trend_4q")),
                "revenue_growth_4q_avg": _num(snap.get("revenue_growth_4q_avg")),
                "period_end": period_end,
                "updated_at": datetime.now(TIMEZONE).replace(tzinfo=None),
            }
        )

    # Upsert never removes: a date that stops computing (age bound, restatement)
    # would otherwise keep serving its last value forever.
    stale = [
        SecFeaturesDaily.instrument_id == instrument_id,
        SecFeaturesDaily.date.between(start, end),
    ]
    if rows:
        stale.append(SecFeaturesDaily.date.notin_([r["date"] for r in rows]))
    session.execute(delete(SecFeaturesDaily).where(*stale))

    if not rows:
        return 0

    chunk = 200
    for i in range(0, len(rows), chunk):
        part = rows[i : i + chunk]
        stmt = pg_insert(SecFeaturesDaily).values(part)
        update_cols = {
            c.name: getattr(stmt.excluded, c.name)
            for c in SecFeaturesDaily.__table__.columns
            if c.name not in ("id", "instrument_id", "date")
        }
        stmt = stmt.on_conflict_do_update(constraint="uq_sec_features_instrument_date", set_=update_cols)
        session.execute(stmt)
    return len(rows)


def update_sec_features(
    *,
    force_fetch: bool = False,
    backfill_start: date | None = None,
    only_holdings: bool = False,
    rebuild: bool = False,
) -> None:
    """Refresh companyfacts and backfill sec_features_daily for CIK instruments."""
    end = datetime.now(TIMEZONE).date()
    start = backfill_start or date(2018, 1, 1)

    with get_session() as session:
        q = select(Instrument.id, Instrument.yahoo_symbol, Instrument.cik).where(Instrument.cik.is_not(None))
        instruments = session.execute(q).all()
        if only_holdings:
            from models import HoldingDaily

            latest = session.execute(select(HoldingDaily.date).order_by(HoldingDaily.date.desc()).limit(1)).scalar()
            if latest:
                held = {
                    r
                    for r in session.execute(
                        select(HoldingDaily.instrument_id).where(
                            HoldingDaily.date == latest, HoldingDaily.quantity > 0
                        )
                    ).scalars()
                }
                instruments = [r for r in instruments if r.id in held]

        total_rows = 0
        fetched = 0
        skipped_no_xbrl = 0
        logger.info("SEC companyfacts User-Agent=%r", SEC_USER_AGENT)
        for instrument_id, symbol, cik in instruments:
            if not symbol or not cik:
                continue
            try:
                refreshed = _should_fetch(session, instrument_id, force_fetch)
                if refreshed:
                    facts = rate_limited_get(cik)
                    upsert_companyfacts(session, instrument_id, cik, facts)
                    fetched += 1
                else:
                    row = session.get(SecCompanyFacts, instrument_id)
                    if row is None:
                        continue
                    facts = row.facts
                    # Refresh cached ROIC series even when facts are reused.
                    if not row.roic_as_restated:
                        row.roic_as_restated = as_restated_roic_payload(facts)

                # Unchanged facts produce identical rows; recomputing all ~57k of
                # them nightly is pure CPU. --rebuild forces it after a logic change.
                if not facts or not (refreshed or rebuild):
                    session.commit()
                    continue

                yh = session.get(InstrumentYahoo, instrument_id)
                splits = (yh.splits if yh else None) or {}
                n = backfill_instrument(session, instrument_id, symbol, facts, splits, start, end)
                total_rows += n
                session.commit()
                logger.info("%s: %d sec_features rows", symbol, n)
            except CompanyFactsUnavailable as exc:
                # Valid CIK, no XBRL. Cache an empty blob so it is not retried nightly.
                upsert_companyfacts(session, instrument_id, cik, {})
                session.commit()
                skipped_no_xbrl += 1
                logger.info("%s: %s — cached as no-XBRL", symbol, exc)
                continue
            except requests.HTTPError as exc:
                session.rollback()
                # Keep hammering a banned IP extends the ban — stop the run.
                if exc.response is not None and exc.response.status_code in (403, 429):
                    logger.error(
                        "Aborting SEC update after %s for %s (CIK %s): %s",
                        exc.response.status_code,
                        symbol,
                        cik,
                        exc,
                    )
                    break
                logger.exception("SEC update failed for %s (CIK %s)", symbol, cik)
                continue
            except Exception:
                logger.exception("SEC update failed for %s (CIK %s)", symbol, cik)
                session.rollback()
                continue

        logger.info(
            "SEC features: fetched %d companyfacts (%d no-XBRL), upserted %d feature rows for %d "
            "instruments (backfill %s→%s). Caveats: backfilled composite ~68%% Tier A; ROIC-only "
            "quality gate; ~23-name quarterly universe; thesis sell flags forced False on SEC rows; "
            "fcf_yield uses adj_close × split-adjusted shares.",
            fetched,
            skipped_no_xbrl,
            total_rows,
            len(instruments),
            start,
            end,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-fetch", action="store_true")
    parser.add_argument("--backfill-start", type=date.fromisoformat, default=None)
    parser.add_argument("--only-holdings", action="store_true")
    parser.add_argument(
        "--rebuild", action="store_true", help="Recompute rows from cached facts (after a logic change)"
    )
    args = parser.parse_args()
    update_sec_features(
        force_fetch=args.force_fetch,
        backfill_start=args.backfill_start,
        only_holdings=args.only_holdings,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
