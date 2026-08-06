"""Fetch SEC companyfacts and backfill sec_features_daily (CIK instruments).

Filing-driven by default: refreshes companyfacts when a newer 10-K/10-Q/20-F
appears in submissions since the last fetch. Use --force-fetch to re-pull all.

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

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from backend.utils.piotroski import get_piotroski_f_score
from backend.utils.sec_companyfacts import (
    as_restated_roic_payload,
    build_statement_pair,
    compute_sec_feature_snapshot,
    concept_series,
    fetch_companyfacts,
    pad_cik,
    rate_limited_get,
    share_count_for_fscore,
)
from backend.utils.splits import adjust_share_count
from config import SEC_USER_AGENT, TIMEZONE, logger
from models import Instrument, InstrumentYahoo, PricesDaily, SecCompanyFacts, SecFeaturesDaily
from scripts.update_data import get_session

import requests

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


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


def _should_fetch(session: Session, instrument_id: int, cik: str, force: bool) -> bool:
    if force:
        return True
    row = session.get(SecCompanyFacts, instrument_id)
    if row is None:
        return True
    # Filing-driven: refresh if submissions show a newer 10-K/10-Q/20-F filed after fetched_at.
    url = SUBMISSIONS_URL.format(cik=pad_cik(cik))
    try:
        resp = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("submissions check failed for CIK %s: %s — fetching anyway", cik, exc)
        return True

    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filed = recent.get("filingDate", [])
    fetched = row.fetched_at.date() if isinstance(row.fetched_at, datetime) else row.fetched_at
    for form, fdate in zip(forms, filed):
        if form not in ("10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "20-F/A"):
            continue
        try:
            d = date.fromisoformat(fdate)
        except ValueError:
            continue
        if d >= fetched:
            return True
        break  # recent list is newest-first
    return False


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
        for instrument_id, symbol, cik in instruments:
            if not symbol or not cik:
                continue
            try:
                if _should_fetch(session, instrument_id, cik, force_fetch):
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

                yh = session.get(InstrumentYahoo, instrument_id)
                splits = (yh.splits if yh else None) or {}
                n = backfill_instrument(session, instrument_id, symbol, facts, splits, start, end)
                total_rows += n
                session.commit()
                logger.info("%s: %d sec_features rows", symbol, n)
            except Exception:
                logger.exception("SEC update failed for %s (CIK %s)", symbol, cik)
                session.rollback()
                continue

        logger.info(
            "SEC features: fetched %d companyfacts, upserted %d feature rows for %d instruments "
            "(backfill %s→%s). Caveats: backfilled composite ~68%% Tier A; ROIC-only quality gate; "
            "~23-name quarterly universe; thesis sell flags forced False on SEC rows; "
            "fcf_yield uses adj_close × split-adjusted shares.",
            fetched,
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
    args = parser.parse_args()
    update_sec_features(
        force_fetch=args.force_fetch,
        backfill_start=args.backfill_start,
        only_holdings=args.only_holdings,
    )


if __name__ == "__main__":
    main()
