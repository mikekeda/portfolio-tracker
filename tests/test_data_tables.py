"""Consistency of the hand-maintained lookup tables in data.py.

These are edited by hand from provider pages, so the failure mode is a
transcription slip — a weight typo, a dropped row, a table that no longer sums
to 100 — not a code bug. ETF *constituents* used to live here too; they are now
fetched into the etf_holdings table and validated by
scripts/check_data_integrity.py, which can see whether they resolve and price.

Pure stdlib (data.py has no imports):

    python tests/test_data_tables.py
    pytest tests/test_data_tables.py
"""

import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import ETC_SYMBOLS, ETF_COUNTRY_ALLOCATION, ETF_HOLDING_SOURCES, ETF_SECTOR_ALLOCATION

TOLERANCE = 0.01
DATA_PY = Path(__file__).resolve().parent.parent / "data.py"


def test_allocation_tables_sum_to_100():
    for table in (ETF_COUNTRY_ALLOCATION, ETF_SECTOR_ALLOCATION):
        for sym, alloc in table.items():
            assert abs(sum(alloc.values()) - 100.0) < TOLERANCE, f"{sym}: {sum(alloc.values())}"


def test_no_negative_or_absurd_weights():
    for table in (ETF_COUNTRY_ALLOCATION, ETF_SECTOR_ALLOCATION):
        for sym, alloc in table.items():
            for bucket, pct in alloc.items():
                assert 0.0 <= pct <= 100.0, f"{sym}/{bucket}: {pct}"


def test_sourced_etfs_are_present_in_the_allocation_tables():
    """update_data.py indexes country/sector directly by symbol and would KeyError."""
    for sym in ETF_HOLDING_SOURCES:
        assert sym in ETF_COUNTRY_ALLOCATION, sym
        assert sym in ETF_SECTOR_ALLOCATION, sym


def test_holding_sources_are_well_formed():
    """A typo'd kind or a missing id fails at fetch time, hours after the edit."""
    required = {"dws": "slug", "ishares": "portfolio_id", "ssga": "ticker", "sp500": None}
    for sym, spec in ETF_HOLDING_SOURCES.items():
        if spec is None:
            continue
        assert spec["kind"] in required, f"{sym}: unknown kind {spec['kind']!r}"
        key = required[spec["kind"]]
        assert key is None or spec.get(key), f"{sym}: {spec['kind']} needs {key}"
        assert spec.get("index_name") and spec.get("source_note"), sym


def test_etcs_have_no_holding_source():
    """An ETC holds one commodity; there is nothing to look through."""
    for sym in ETC_SYMBOLS:
        assert sym not in ETF_HOLDING_SOURCES, sym


def test_every_public_table_carries_an_updated_marker():
    """Not a freshness check — it only ensures a new table cannot be added without
    recording when it was last touched, so the dates stay meaningful."""
    text = DATA_PY.read_text()
    declared = set(re.findall(r"^([A-Z_][A-Z0-9_]*)\s*[:=]", text, re.M))
    marked = set(re.findall(r"^([A-Z_][A-Z0-9_]*)[^#\n]*#\s*updated \d{4}-\d{2}-\d{2}", text, re.M))
    assert declared == marked, f"unmarked: {sorted(declared - marked)}"


def test_updated_markers_parse_and_are_not_in_the_future():
    """Catches a typo'd year, which is otherwise invisible."""
    text = DATA_PY.read_text()
    for name, stamp in re.findall(
        r"^([A-Z_][A-Z0-9_]*)[^#\n]*#\s*updated (\d{4}-\d{2}-\d{2})", text, re.M
    ):
        assert date.fromisoformat(stamp) <= date.today(), f"{name}: {stamp}"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
