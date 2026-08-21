"""Consistency of the hand-maintained lookup tables in data.py.

These are edited by hand from provider pages, so the failure mode is a
transcription slip — a weight typo, a dropped row, a fund that no longer sums to
100 — not a code bug. Everything here is a data-shape assertion.

Pure stdlib (data.py has no imports):

    python tests/test_etf_constituents.py
    pytest tests/test_etf_constituents.py
"""

import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import ETC_SYMBOLS, ETF_CONSTITUENTS, ETF_COUNTRY_ALLOCATION, ETF_SECTOR_ALLOCATION

TOLERANCE = 0.01
OTHER = "Other"
ALLOCATION_TABLES = (ETF_CONSTITUENTS, ETF_COUNTRY_ALLOCATION, ETF_SECTOR_ALLOCATION)


def test_every_allocation_table_sums_to_100():
    """The whole point of the "Other" bucket — a fund that does not close to 100
    means a row was dropped or mistyped."""
    for table in ALLOCATION_TABLES:
        for sym, alloc in table.items():
            assert abs(sum(alloc.values()) - 100.0) < TOLERANCE, f"{sym}: {sum(alloc.values())}"


def test_no_negative_or_absurd_weights():
    for table in ALLOCATION_TABLES:
        for sym, alloc in table.items():
            for bucket, pct in alloc.items():
                assert 0.0 <= pct <= 100.0, f"{sym}/{bucket}: {pct}"


def test_constituent_weights_are_descending_before_other():
    """A published top-N is ordered; an out-of-order row is a transcription slip.
    "Other" is the tail and sits last regardless of size."""
    for sym, alloc in ETF_CONSTITUENTS.items():
        weights = [w for k, w in alloc.items() if k != OTHER]
        assert weights == sorted(weights, reverse=True), sym
        assert list(alloc)[-1] == OTHER, f"{sym}: Other must be last"


def test_every_constituent_fund_has_an_other_bucket():
    for sym, alloc in ETF_CONSTITUENTS.items():
        assert OTHER in alloc, sym


def test_no_duplicate_or_blank_tickers():
    """Dual share classes (GOOGL/GOOG) are distinct rows and must stay distinct."""
    for sym, alloc in ETF_CONSTITUENTS.items():
        assert all(k.strip() for k in alloc), sym
        assert len(alloc) == len(set(alloc)), sym


def test_constituent_funds_are_present_in_the_allocation_tables():
    """update_data.py indexes country/sector directly by symbol and would KeyError."""
    for sym in ETF_CONSTITUENTS:
        assert sym in ETF_COUNTRY_ALLOCATION, sym
        assert sym in ETF_SECTOR_ALLOCATION, sym


def test_etcs_have_no_constituents():
    """An ETC holds one commodity; there is nothing to look through."""
    for sym in ETC_SYMBOLS:
        assert sym not in ETF_CONSTITUENTS, sym


def test_every_public_table_carries_an_updated_marker():
    """Not a freshness check — it only ensures a new table cannot be added without
    recording when it was last touched, so the dates stay meaningful."""
    text = (Path(__file__).resolve().parent.parent / "data.py").read_text()
    declared = set(re.findall(r"^([A-Z_][A-Z0-9_]*)\s*[:=]", text, re.M))
    marked = set(re.findall(r"^([A-Z_][A-Z0-9_]*)[^#\n]*#\s*updated \d{4}-\d{2}-\d{2}", text, re.M))
    assert declared == marked, f"unmarked: {sorted(declared - marked)}"


def test_updated_markers_parse_and_are_not_in_the_future():
    """Catches a typo'd year, which is otherwise invisible."""
    text = (Path(__file__).resolve().parent.parent / "data.py").read_text()
    for name, stamp in re.findall(r"^([A-Z_][A-Z0-9_]*)[^#\n]*#\s*updated (\d{4}-\d{2}-\d{2})", text, re.M):
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
