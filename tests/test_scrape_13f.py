"""Tests for 13F filing-window selection, amendment handling, and multi-CIK merging."""

from scripts.scrape_13f import (
    amendment_disposition,
    find_info_table_xml,
    investor_ciks,
    merge_new_holdings,
    merge_same_quarter,
    normalize_cusip,
    parse_amendment_type,
    parse_nt_successor_ciks,
    primary_doc_name,
    reported_in_thousands,
    select_recent_filings,
)

# Berkshire's Aug 2025 window: the Q1 amendment was filed the same day as the Q2 original,
# so EDGAR lists it first and a one-per-date rule would drop the Q1 original behind it.
BERKSHIRE_RECENT = {
    "form": ["13F-HR/A", "13F-HR", "13F-HR", "13F-HR", "4"],
    "accessionNumber": ["amd-q1", "hr-q2", "hr-q1", "hr-q4", "form4"],
    "primaryDocument": ["primary_doc.xml"] * 5,
    "reportDate": ["2025-03-31", "2025-06-30", "2025-03-31", "2024-12-31", "2025-06-30"],
}


def _holding(cusip, shares, value):
    return {"issuer": "X", "cusip": cusip, "titleOfClass": "COM", "value": value, "shares": shares}


def _filing(cik, report_date, holdings, form="13F-HR", accession="acc"):
    return {
        "investor": "Test",
        "cik": cik,
        "reportDate": report_date,
        "form": form,
        "accessionNumber": accession,
        "holdingsCount": len(holdings),
        "totalValue": sum(h["value"] for h in holdings),
        "holdings": holdings,
    }


def test_select_recent_filings_keeps_original_alongside_amendment():
    picked = select_recent_filings(BERKSHIRE_RECENT, max_filings=2)
    assert [(f["form"], f["reportDate"]) for f in picked] == [
        ("13F-HR/A", "2025-03-31"),
        ("13F-HR", "2025-06-30"),
        ("13F-HR", "2025-03-31"),
    ]


def test_select_recent_filings_window_is_by_report_date_not_position():
    # The Q1 amendment is listed first because EDGAR orders by filing date, but a window of one
    # quarter must still resolve to the newest quarter.
    picked = select_recent_filings(BERKSHIRE_RECENT, max_filings=1)
    assert [(f["form"], f["reportDate"]) for f in picked] == [("13F-HR", "2025-06-30")]


def test_select_recent_filings_ignores_non_13f_forms():
    assert all(f["form"].startswith("13F") for f in select_recent_filings(BERKSHIRE_RECENT, max_filings=4))


def test_select_recent_filings_includes_notices():
    recent = {
        "form": ["13F-NT", "13F-HR"],
        "accessionNumber": ["nt", "hr"],
        "primaryDocument": ["primary_doc.xml", "primary_doc.xml"],
        "reportDate": ["2026-06-30", "2026-03-31"],
    }
    assert [f["form"] for f in select_recent_filings(recent, max_filings=2)] == ["13F-NT", "13F-HR"]


def test_parse_amendment_type():
    xml = (
        "<edgarSubmission><formData><coverPage>"
        "<amendmentType>NEW HOLDINGS</amendmentType>"
        "</coverPage></formData></edgarSubmission>"
    )
    assert parse_amendment_type(xml) == "NEW HOLDINGS"
    assert parse_amendment_type("<edgarSubmission><formData/></edgarSubmission>") == ""


def test_parse_amendment_type_handles_namespaces():
    xml = (
        '<ns:edgarSubmission xmlns:ns="http://www.sec.gov/edgar/thirteenffiler">'
        "<ns:amendmentType>Restatement</ns:amendmentType></ns:edgarSubmission>"
    )
    assert parse_amendment_type(xml) == "RESTATEMENT"


def test_parse_nt_successor_ciks():
    xml = (
        "<edgarSubmission><formData><coverPage><otherManagersInfo>"
        "<otherManager><cik>0002026053</cik><name>PERSHING SQUARE INC.</name></otherManager>"
        "</otherManagersInfo></coverPage></formData></edgarSubmission>"
    )
    assert parse_nt_successor_ciks(xml) == ["0002026053"]


def test_parse_nt_successor_ciks_skips_managers_without_a_cik():
    # Regular 13F-HR/A filings carry otherManager blocks that name a manager but no CIK.
    xml = (
        "<edgarSubmission><otherManagersInfo>"
        "<otherManager><form13FFileNumber>28-554</form13FFileNumber><name>Buffett Warren E</name></otherManager>"
        "</otherManagersInfo></edgarSubmission>"
    )
    assert parse_nt_successor_ciks(xml) == []


def test_primary_doc_name_strips_the_stylesheet_path():
    # EDGAR reports this path, which it renders as HTML; the raw XML is at the bare basename.
    assert primary_doc_name("xslForm13F_X02/primary_doc.xml") == "primary_doc.xml"
    assert primary_doc_name("primary_doc.xml") == "primary_doc.xml"
    assert primary_doc_name("") == "primary_doc.xml"


def test_find_info_table_xml_never_ranks_the_cover_page():
    # Berkshire's 2025-03-31 amendment: the 4-row info table is smaller than primary_doc.xml,
    # so a plain largest-file rule picks the cover page and parses zero holdings.
    index = {"directory": {"item": [{"name": "43981.xml", "size": "2134"}, {"name": "primary_doc.xml", "size": "2854"}]}}
    assert find_info_table_xml(index) == "43981.xml"


def test_find_info_table_xml_prefers_infotable_then_falls_back():
    named = {"directory": {"item": [{"name": "infotable.xml", "size": "10"}, {"name": "big.xml", "size": "999"}]}}
    assert find_info_table_xml(named) == "infotable.xml"
    only_cover = {"directory": {"item": [{"name": "primary_doc.xml", "size": "2854"}]}}
    assert find_info_table_xml(only_cover) == "primary_doc.xml"
    assert find_info_table_xml({"directory": {"item": []}}) is None


def test_amendment_disposition_classifies_the_three_cases():
    assert amendment_disposition("13F-HR", "") == "replace"
    assert amendment_disposition("13F-HR/A", "RESTATEMENT") == "replace"
    assert amendment_disposition("13F-HR/A", "NEW HOLDINGS") == "merge"


def test_amendment_disposition_skips_unclassifiable_amendments():
    # An unreadable primary_doc parses to "". Defaulting that to "replace" is how a four-row
    # NEW HOLDINGS delta would overwrite a 110-row quarter.
    assert amendment_disposition("13F-HR/A", "") == "skip"
    assert amendment_disposition("13F-HR/A", "SOME FUTURE TYPE") == "skip"


def test_reported_in_thousands_ignores_partial_amendments():
    # Pershing's 2024-12-31 amendment is one $46.5M lot. Scaling it as if it were a whole
    # portfolio below the $100M filing floor turned a $12.66bn quarter into $59.1bn.
    assert reported_in_thousands(46_533_105, is_partial=True) is False
    assert reported_in_thousands(46_533_105, is_partial=False) is True


def test_reported_in_thousands_leaves_dollar_filings_alone():
    assert reported_in_thousands(12_614_560_346, is_partial=False) is False
    assert reported_in_thousands(0, is_partial=False) is False


def test_merge_new_holdings_appends_without_deduping_cusips():
    # Berkshire's Q1 2025 amendment reported a further 202-share lot of a CUSIP the original
    # already carried; deduping by CUSIP would silently drop it.
    base = _filing("1067983", "2025-03-31", [_holding("526057302", 152_572, 16_641_028)])
    amendment = _filing(
        "1067983", "2025-03-31", [_holding("526057302", 202, 22_032)], form="13F-HR/A", accession="amd"
    )
    merged = merge_new_holdings(base, amendment)

    assert merged["holdingsCount"] == 2
    assert sum(h["shares"] for h in merged["holdings"]) == 152_774
    assert merged["totalValue"] == 16_663_060
    # Stays identified as the full filing so the repair script does not reselect it forever.
    assert merged["form"] == "13F-HR"
    assert merged["accessionNumber"] == "acc"


def test_merge_same_quarter_unions_ciks_under_the_primary():
    old_cik = _filing("1336528", "2026-03-31", [_holding("44267D107", 18_852_064, 1_193_000_000)])
    new_cik = _filing("2026053", "2026-03-31", [_holding("44267D107", 9_000_000, 569_340_000)])
    merged = merge_same_quarter([old_cik, new_cik], primary_cik="1336528")

    assert len(merged) == 1
    assert merged[0]["cik"] == "1336528"
    assert merged[0]["holdingsCount"] == 2
    assert sum(h["shares"] for h in merged[0]["holdings"]) == 27_852_064
    assert merged[0]["totalValue"] == 1_762_340_000


def test_merge_same_quarter_keeps_distinct_quarters_apart():
    merged = merge_same_quarter(
        [
            _filing("2026053", "2026-06-30", [_holding("A", 1, 10)]),
            _filing("1336528", "2026-03-31", [_holding("B", 2, 20)]),
        ],
        primary_cik="1336528",
    )
    assert {f["reportDate"] for f in merged} == {"2026-06-30", "2026-03-31"}
    assert all(f["cik"] == "1336528" for f in merged)


def test_investor_ciks_defaults_to_the_primary():
    assert investor_ciks({"name": "X", "cik": "123"}) == ["123"]
    assert investor_ciks({"name": "X", "cik": "123", "ciks": ["123", "456"]}) == ["123", "456"]


def test_normalize_cusip():
    assert normalize_cusip(" 48251w104 ") == "48251W104"
