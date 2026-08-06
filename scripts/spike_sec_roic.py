"""Phase 0 spike: companyfacts → FY ROIC vs Yahoo; restatement retention check.

Needs network + a compliant SEC_USER_AGENT (T212_SEC_USER_AGENT). No T212 API /
Gemini keys. Writes a summary under data/sec_probe/ (gitignored via data/).

    python scripts/spike_sec_roic.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.utils.roic import get_roic, get_roic_history
from backend.utils.sec_companyfacts import (
    fy_roic_series,
    pad_cik,
    rate_limited_get,
    require_sec_user_agent,
    restatement_retention_ok,
)
from config import SEC_USER_AGENT, logger
from models import Instrument, InstrumentYahoo
from scripts.update_data import get_session

OUT_DIR = Path("data/sec_probe")

# ~10 US holdings for divergence categorisation; first three are hand-reconcile targets.
SPIKE_SYMBOLS = [
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "AVGO",
    "ISRG",
    "NFLX",
    "UBER",
    "BKNG",
]


def _yahoo_roic(session: Session, symbol: str) -> dict:
    row = session.execute(
        select(InstrumentYahoo)
        .join(Instrument, Instrument.id == InstrumentYahoo.instrument_id)
        .where(Instrument.yahoo_symbol == symbol)
    ).scalar_one_or_none()
    if row is None:
        return {"error": "no InstrumentYahoo"}
    info = row.info or {}
    bs, inc = row.balance_sheet or {}, row.income_stmt or {}
    return {
        "roic": get_roic(info, bs, inc),
        "roic_history": get_roic_history(bs, inc, periods=3),
        "ann_is_ends": sorted(inc.keys(), reverse=True)[:5],
    }


def main() -> None:
    require_sec_user_agent()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "ua": SEC_USER_AGENT,
        "as_of": date.today().isoformat(),
        "symbols": {},
        "restatement": None,
        "notes": [
            "Gate is hand-reconcile + explained Yahoo gaps, not abs-error vs Yahoo.",
            "Divergence categories: tag_choice | duration | restatement_vintage | fx | financials",
        ],
    }

    with get_session() as session:
        cik_rows = session.execute(
            select(Instrument.yahoo_symbol, Instrument.cik).where(
                Instrument.yahoo_symbol.in_(SPIKE_SYMBOLS), Instrument.cik.is_not(None)
            )
        ).all()
        cik_by_sym = {sym: cik for sym, cik in cik_rows}

    for i, symbol in enumerate(SPIKE_SYMBOLS):
        cik = cik_by_sym.get(symbol)
        entry: dict = {"cik": cik}
        if not cik:
            entry["error"] = "no CIK"
            summary["symbols"][symbol] = entry
            continue
        try:
            facts = rate_limited_get(cik)
            (OUT_DIR / f"cf_{symbol}.json").write_text(json.dumps(facts))
            sec_series = fy_roic_series(facts, vintage="as_restated")
            entry["sec_fy_roic"] = [
                {"end": e.isoformat(), "roic": round(r, 2), "filed": f.isoformat()} for e, r, f in sec_series[-8:]
            ]
            entry["sec_latest"] = entry["sec_fy_roic"][-1] if entry["sec_fy_roic"] else None
            with get_session() as session:
                entry["yahoo"] = _yahoo_roic(session, symbol)
            if entry["sec_latest"] and entry["yahoo"].get("roic") is not None:
                entry["delta_pp"] = round(entry["sec_latest"]["roic"] - entry["yahoo"]["roic"], 2)
            if symbol == "MSFT" or (summary["restatement"] is None and i == 0):
                summary["restatement"] = {"symbol": symbol, **restatement_retention_ok(facts)}
            logger.info(
                "%s: SEC latest=%s Yahoo=%s delta=%s",
                symbol,
                entry.get("sec_latest"),
                entry.get("yahoo", {}).get("roic"),
                entry.get("delta_pp"),
            )
        except Exception as exc:  # noqa: BLE001 — spike must continue across names
            entry["error"] = str(exc)
            logger.exception("%s failed", symbol)
        summary["symbols"][symbol] = entry

    out_path = OUT_DIR / "spike_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Wrote %s (restatement ok=%s)", out_path, (summary.get("restatement") or {}).get("ok"))


if __name__ == "__main__":
    main()
