"""UK CPI parsing and rate derivation for the ONS MM23 timeseries.

Pure stdlib so it stays importable (and testable) without the app's dependencies;
the fetching and caching live in `backend/views/projection.py`.
"""

import csv
from typing import Any, Optional

MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
    )
}

# A YoY rate needs 13 monthly points; a 10y CAGR uses whatever is available up to 121.
MIN_CPI_OBSERVATIONS = 13
CPI_LOOKBACK_MONTHS = 120


def parse_ons_csv(text: str) -> Optional[list[dict[str, Any]]]:
    """Parse an ONS timeseries CSV into monthly observations, newest first.

    Monthly rows look like ("2026 MAY", "142.4"); header, annual ("2026") and
    quarterly ("2026 Q1") rows are skipped. Dates are returned as ISO strings
    (first of the month). None when the text holds no monthly rows.
    """
    out = []
    for row in csv.reader(text.splitlines()):
        if len(row) != 2:
            continue
        parts = row[0].split()
        if len(parts) != 2 or parts[1] not in MONTHS:
            continue
        try:
            value = float(row[1])
        except ValueError:
            continue
        out.append({"date": f"{parts[0]}-{MONTHS[parts[1]]:02d}-01", "value": value})
    out.sort(key=lambda o: o["date"], reverse=True)
    return out or None


def cpi_metrics(obs: Optional[list[dict[str, Any]]]) -> dict[str, Any]:
    """Trailing 12m YoY, trailing 10y CAGR, and the latest observation date.

    `obs` is newest-first, as returned by `parse_ons_csv`. Rates are decimals
    (0.024 = 2.4%); every field is None when there is too little history.
    """
    if not obs or len(obs) < MIN_CPI_OBSERVATIONS:
        return {"cpi_12m": None, "cpi_10y": None, "as_of": None}
    latest = obs[0]["value"]
    twelve_back = obs[12]["value"]
    cpi_12m = (latest / twelve_back - 1.0) if twelve_back > 0 else None

    ten_year_idx = min(CPI_LOOKBACK_MONTHS, len(obs) - 1)
    old = obs[ten_year_idx]["value"]
    years = ten_year_idx / 12.0
    cpi_10y = ((latest / old) ** (1.0 / years) - 1.0) if old > 0 and years > 0 else None
    return {"cpi_12m": cpi_12m, "cpi_10y": cpi_10y, "as_of": obs[0]["date"]}
