"""Regression coverage for the redesigned Wisesheets P/E page."""

import json
import unittest
from contextlib import nullcontext
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import scrape_wisesheets_pe as scraper


def page(quarterly, symbol="TEST"):
    data = {"keyMetricsQuarterly": quarterly, "keyMetricsAnnual": [{"date": "2025-09-27", "peRatio": 99}]}
    payload = {"props": {"pageProps": {"symbol": symbol, "data": data}}}
    return f'<script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script>'


class WisesheetsPETests(unittest.TestCase):
    def test_reads_quarterly_ratios_at_actual_fiscal_dates(self):
        html = page([
            {"date": "2026-09-26", "peRatio": 40.41113744},
            {"date": "2026-06-27", "peRatio": 34.94827586},
            {"date": "2026-03-28", "peRatio": -8.3},
            {"date": "2025-12-27", "peRatio": None},
        ])
        self.assertEqual(scraper.parse_pe_data(html, "TEST"), {
            "2026-09-26": 40.41113744,
            "2026-06-27": 34.94827586,
        })

    def test_missing_quarterly_series_is_not_no_coverage(self):
        with self.assertRaisesRegex(ValueError, "missing its embedded data"):
            scraper.parse_pe_data("<html>Service unavailable</html>", "TEST")
        with self.assertRaisesRegex(ValueError, "unexpected structure"):
            scraper.parse_pe_data(page([]).replace("keyMetricsQuarterly", "moved"), "TEST")
        self.assertEqual(scraper.parse_pe_data(page([]), "TEST"), {})

    def test_rejects_page_for_different_ticker(self):
        with self.assertRaisesRegex(ValueError, "different ticker"):
            scraper.parse_pe_data(page([], symbol="OTHER"), "TEST")

    def test_merge_replaces_legacy_quarter_date_without_erasing_other_history(self):
        old_wisesheets = {"stock_price": None, "ttm_eps": None, "pe_ratio": 21.0}
        macrotrends = {"stock_price": 40.0, "ttm_eps": 2.0, "pe_ratio": 20.0}
        existing = {
            "2024-03-31": old_wisesheets,
            "2026-09-25": macrotrends,
            "2026-09-30": old_wisesheets,
            "2026-12-31": old_wisesheets,
        }
        result = scraper.merge_pe_history(existing, {"2026-09-26": 30.0}, date(2026, 11, 11))
        self.assertEqual(result, {
            "2024-03-31": old_wisesheets,
            "2026-09-25": macrotrends,
            "2026-09-26": {"stock_price": None, "ttm_eps": None, "pe_ratio": 30.0},
        })

    def test_total_failure_is_visible_to_celery_and_preserves_history(self):
        row = SimpleNamespace(instrument=SimpleNamespace(yahoo_symbol="TEST"))
        session = Mock()
        session.execute.return_value.scalars.return_value.all.side_effect = [[row], []]
        with (
            patch.object(scraper, "get_session", return_value=nullcontext(session)),
            patch.object(scraper, "fetch_pe_data", side_effect=ValueError("layout changed")),
            patch.object(scraper, "sleep"),
            self.assertLogs(scraper.logger, level="ERROR"),
            self.assertRaisesRegex(RuntimeError, "updated no instruments"),
        ):
            scraper.update_pe_data(limit=1, onboard_limit=0)
        session.commit.assert_not_called()
        session.rollback.assert_called_once()


if __name__ == "__main__":
    unittest.main()
