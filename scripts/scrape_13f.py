"""
Scrape SEC Form 13F Data
======================
Fetches 13F-HR filings from SEC EDGAR for a curated list of institutional investors,
parses the information table XML, and saves to database (Form13FManager, Form13FFiling, Form13FHolding).

13F filings are quarterly disclosures by institutional managers with $100M+ AUM.
Signals: new positions and large increases are more reliable than sells.

Note: Value is in dollars (nearest dollar) for filings from Jan 2023 onward.
Pre-2023 filings may report value in thousands.

SEC requires User-Agent: "CompanyName admin@example.com"
"""

from datetime import date, datetime
from time import sleep
from typing import Optional, TypedDict

import requests
from lxml import etree
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from config import SEC_USER_AGENT, logger
from models import Form13FFiling, Form13FHolding, Form13FManager, Instrument
from scripts.update_data import get_session

HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}

# SEC Endpoints
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}"

# Rate limit: 10 requests per second per SEC guidelines
REQUEST_DELAY = 0.12

# Request timeout (seconds) to avoid hanging
REQUEST_TIMEOUT = 30

# Max unmatched CUSIPs to log (sorted by aggregate 13F $); remainder only counted
UNMATCHED_LOG_LIMIT = 20

# Some instruments' ISINs in our DB (typically issuer-country ISINs from Trading212)
# do not embed the SEC 9-char CUSIP used in 13F info tables. For those, override the
# CUSIP used for matching Form13FHolding rows to Instrument rows.
# Omit entries where isin[2:11] already equals the SEC CUSIP (typical US*/CA* lines).
ISIN_TO_CUSIP_OVERRIDES: dict[str, str] = {
    # Already observed mismatches
    "CH0044328745": "H1467J104",  # Chubb Limited
    "LU1778762911": "L8681T102",  # Spotify Technology SA
    # High-value unmatched examples from logs
    "IE00BY7QL619": "G51502105",  # Johnson Controls International plc
    "IE00BDB6Q211": "G96629103",  # Willis Towers Watson plc
    "IE00BLP1HW54": "G0403H108",  # Aon plc
    "IE000IVNQZ81": "G87052109",  # TE Connectivity plc
    # 2026-04-08 run — top unmatched by 13F $ (issuer ISIN vs SEC CUSIP)
    "BE0974293251": "03524A108",  # Anheuser-Busch InBev SA/NV (vs US ADR ISIN)
    "IE00B8KQN827": "G29183103",  # Eaton Corp plc
    "BRPETRACNOR9": "71654V408",  # Petróleo Brasileiro Petrobras (ordinary vs ADR line in 13F)
    "IE0005711209": "G4705A100",  # ICON plc
    "CH0012005267": "66987V109",  # Novartis AG (Swiss line vs NYSE ADR CUSIP)
    "GB0002875804": "110448107",  # British American Tobacco plc (UK vs ADR CUSIP)
    "CH0244767585": "H42097107",  # UBS Group AG
    "GB00BMX86B70": "405552100",  # Haleon plc
    "NL0011585146": "N3167Y103",  # Ferrari N.V. (Euronext vs NYSE CUSIP)
    "BE0003816331": "04016X101",  # argenx SE (Belgium)
    "NL0010832176": "04016X101",  # argenx SE (Euronext NL line)
    "GB00BZ09BD16": "049468101",  # Atlassian Corp Plc
    "IE00B4BNMY34": "G1151C101",  # Accenture plc (Dublin ordinary vs NYSE CUSIP)
    "NL0010545661": "N20944109",  # CNH Industrial N.V. (Euronext line vs NYSE CUSIP)
    "CH1430134226": "H2927K103",  # Amrize Ltd (Swiss ordinary vs NYSE CUSIP)
    # 2026-04-08 — $1B+ 13F names (Irish/CH/MX lines vs SEC CUSIP)
    "IE00BWT6H894": "G3643J108",  # Flutter Entertainment plc
    "CH0114405324": "H2906T109",  # Garmin Ltd (SIX vs NYSE CUSIP)
    "IE00BYTBXV33": "783513203",  # Ryanair Holdings plc (Dublin vs ADR CUSIP)
    "GB00BVZK7T90": "904767803",  # Unilever PLC (UK line vs US ADR CUSIP UL)
    # 2026-04-19 — top unmatched by 13F $ (>$2B) after substring-ISIN UPDATE
    "NL0009434992": "N53745100",  # LyondellBasell Industries N.V. (Euronext NL line)
    "IE00BTN1Y115": "G5960L103",  # Medtronic plc (Irish line)
    "IE0001827041": "G25508105",  # CRH plc (Dublin vs NYSE CUSIP)
    "BMG0112X1056": "0076CA104",  # Aegon Ltd (Bermuda post-2023 redomicile)
    "IE000S9YS762": "G54950103",  # Linde plc (Irish ordinary vs NYSE CUSIP)
    "GB00BMVP7Y09": "G7709Q104",  # Royalty Pharma plc (UK ordinary)
    "DE0005140008": "D18190898",  # Deutsche Bank AG (Xetra line)
    "FI0009000681": "654902204",  # Nokia Oyj (Helsinki line vs US ADR CUSIP)
    "NL0013056914": "N14506104",  # Elastic N.V. (Dutch ISIN vs NYSE CUSIP)
    "GB0009895292": "046353108",  # AstraZeneca PLC (LSE line vs US ADR CUSIP AZN)
    "IL0011762130": "M7S64H106",  # monday.com Ltd (Israeli ISIN vs Nasdaq CUSIP MNDY)
    "NL0000687663": "N00985106",  # AerCap Holdings N.V. (Dutch ISIN vs NYSE CUSIP AER)
}


class FilingMetadata(TypedDict):
    """SEC filing metadata from submissions API."""

    accessionNumber: str
    primaryDocument: str
    form: str
    reportDate: str


class ParsedHolding(TypedDict):
    """Parsed holding from 13F info table XML."""

    issuer: str
    cusip: str
    titleOfClass: str
    value: int
    shares: int


class ScrapedFiling(TypedDict):
    """Scraped filing with holdings for DB save."""

    investor: str
    cik: str
    reportDate: str
    form: str
    accessionNumber: str
    holdingsCount: int
    totalValue: int
    holdings: list[ParsedHolding]


# Curated list of institutional investors to track.
# Selected for buy-and-hold signal quality: concentrated positions, long time horizons,
# fundamental analysis. 21 investors across value, quality growth, activist, and macro.
#
# Removed: Greenlight (underperformed since 2015), Icahn (short-term activist),
#          Soros (macro bets not visible in 13F),
#          Scion (fund deregistered 2025-11-10; Burry liquidated and returned capital),
#          Tiger Global (high turnover; redundant tech-growth signal vs Coatue/Lone Pine).
INVESTORS: list[dict[str, str]] = [
    # Value / long-term
    {"name": "Berkshire Hathaway", "cik": "1067983"},
    {"name": "Baupost Group", "cik": "1061768"},
    {"name": "Pershing Square", "cik": "1336528"},
    {"name": "Pabrai Investment Funds", "cik": "1549575"},  # Filed under "Dalal Street, LLC"; personal CIK 1173334 stopped filing after 2011
    {
        "name": "Yacktman Asset Management",
        "cik": "905567",
    },  # GARP – quality compounders at a discount; patient, low turnover
    {
        "name": "Dodge & Cox",
        "cik": "200217",
    },  # Contrarian deep value; decades-long holds; adds institutional conviction signal
    {
        "name": "Harris Associates",
        "cik": "813917",
    },  # Oakmark funds; concentrated value with margin of safety; mid/large cap coverage
    # International value (UK/EU and global non-US coverage)
    {
        "name": "Causeway Capital Management",
        "cik": "1165797",
    },  # International value, $7B AUM; UK/EU large caps (Deutsche Bank, Barclays, AZN, Smurfit Westrock)
    {
        "name": "First Eagle Investment Management",
        "cik": "1325447",
    },  # Global value/quality (Eveillard/Crystal lineage); $57B AUM; gold as tail-risk hedge
    # Quality compounders (strong alignment with buy-and-hold, 5-10y horizon)
    {"name": "Akre Capital Management", "cik": "1112520"},  # "Triple-double" compounders, 10y+ holds
    {"name": "Polen Capital Management", "cik": "1034524"},  # Concentrated quality growth, low turnover
    {"name": "Himalaya Capital", "cik": "1709323"},  # Li Lu – Buffett-adjacent, very concentrated
    # Long-term growth
    {"name": "Baillie Gifford", "cik": "1088875"},  # 5-10y growth horizon, $120B AUM
    {"name": "Lone Pine Capital", "cik": "1061165"},
    {"name": "Coatue Management", "cik": "1135730"},
    {
        "name": "Artisan Partners",
        "cik": "1466153",
    },  # Multi-strategy; 13F aggregates all funds (Global Value + growth); good international coverage
    # Activist / engagement
    {"name": "Third Point", "cik": "1040273"},
    {"name": "ValueAct Holdings", "cik": "1418814"},
    # Macro / diversified
    {"name": "Duquesne Family Office", "cik": "1536411"},
    {"name": "Appaloosa", "cik": "1656456"},
    {"name": "GQG Partners", "cik": "1697233"},  # Quality growth, $65B AUM, strong track record
]


def _normalize_cik(cik: str) -> str:
    """Ensure CIK is 10-digit zero-padded."""
    return str(cik).zfill(10)


def get_recent_13f_filings(cik: str, max_filings: int = 2) -> list[FilingMetadata]:
    """Fetches the most recent 13F-HR/13F-HR/A filings for the given CIK (latest + previous quarter)."""
    cik = _normalize_cik(cik)
    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()

    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accessions = filings.get("accessionNumber", [])
    primary_docs = filings.get("primaryDocument", [])
    report_dates = filings.get("reportDate", [])

    result: list[FilingMetadata] = []
    seen_dates: set[str] = set()
    for i, form in enumerate(forms):
        if form not in ("13F-HR", "13F-HR/A"):
            continue
        report_date = report_dates[i] if i < len(report_dates) else ""
        if report_date in seen_dates:
            continue
        seen_dates.add(report_date)
        result.append(
            {
                "accessionNumber": accessions[i] if i < len(accessions) else "",
                "primaryDocument": primary_docs[i] if i < len(primary_docs) else "",
                "form": form,
                "reportDate": report_date,
            }
        )
        if len(result) >= max_filings:
            break
    return result


def get_filing_index(cik: str, accession: str) -> Optional[dict]:
    """Fetches the filing index.json to list available documents."""
    cik = _normalize_cik(cik)
    accession_clean = accession.replace("-", "")
    url = SEC_ARCHIVES_URL.format(cik=cik, accession=accession_clean, filename="index.json")
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    if response.status_code != 200:
        return None
    return response.json()


def find_info_table_xml(index_data: dict) -> Optional[str]:
    """
    Determines which XML file contains the information table.
    Prefers: infotable.xml, then largest .xml file, then primary_doc.xml.
    """
    items = index_data.get("directory", {}).get("item", [])
    if isinstance(items, dict):
        items = [items]

    xml_files: list[tuple[str, int]] = []
    for item in items:
        name = item.get("name", "")
        if not name.endswith(".xml"):
            continue
        size_str = item.get("size", "0")
        try:
            size = int(size_str) if size_str else 0
        except ValueError:
            size = 0
        xml_files.append((name, size))

    if not xml_files:
        return None

    # Prefer infotable.xml (common for online filers)
    for name, _ in xml_files:
        if name.lower() == "infotable.xml":
            return name

    # Otherwise use largest XML (info table is typically the biggest)
    xml_files.sort(key=lambda x: x[1], reverse=True)
    return xml_files[0][0]


def download_xml(cik: str, accession: str, filename: str) -> Optional[str]:
    """Downloads XML content from SEC EDGAR."""
    cik = _normalize_cik(cik)
    accession_clean = accession.replace("-", "")
    url = SEC_ARCHIVES_URL.format(cik=cik, accession=accession_clean, filename=filename)
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    if response.status_code != 200:
        return None
    return response.content.decode("utf-8", errors="replace")


def parse_13f_xml(xml_content: str) -> list[ParsedHolding]:
    """
    Parses 13F information table XML and extracts holdings.
    Handles both namespaced and non-namespaced documents.

    Note: Value is in dollars (nearest dollar) for filings from Jan 2023 onward.
    Pre-2023 filings may report value in thousands; caller may need to scale.
    """
    holdings: list[ParsedHolding] = []
    root = etree.fromstring(xml_content.encode("utf-8"))

    # infoTable elements (use double quotes for XPath predicate compatibility)
    info_tables = root.xpath(".//*[local-name()='infoTable']")
    if not info_tables:
        info_tables = root.xpath(".//*[local-name()='informationTable']")
        if info_tables:
            info_tables = info_tables[0].xpath(".//*[local-name()='infoTable']")

    for it in info_tables:
        name_elem = it.xpath(".//*[local-name()='nameOfIssuer']")
        cusip_elem = it.xpath(".//*[local-name()='cusip']")
        value_elem = it.xpath(".//*[local-name()='value']")
        shrs_elem = it.xpath(".//*[local-name()='shrsOrPrnAmt']")
        ssh_prnamt_list = shrs_elem[0].xpath(".//*[local-name()='sshPrnamt']") if shrs_elem else []
        title_elem = it.xpath(".//*[local-name()='titleOfClass']")

        name_elem = name_elem[0] if name_elem else None
        cusip_elem = cusip_elem[0] if cusip_elem else None
        value_elem = value_elem[0] if value_elem else None
        ssh_prnamt = ssh_prnamt_list[0] if ssh_prnamt_list else None
        title_elem = title_elem[0] if title_elem else None

        issuer = (name_elem.text or "").strip() if name_elem is not None else ""
        cusip = (cusip_elem.text or "").strip() if cusip_elem is not None else ""
        title = (title_elem.text or "").strip() if title_elem is not None else ""

        value = 0
        if value_elem is not None and value_elem.text:
            try:
                value = int(value_elem.text.strip().replace(",", ""))
            except ValueError:
                pass

        shares = 0
        if ssh_prnamt is not None and ssh_prnamt.text:
            try:
                shares = int(ssh_prnamt.text.strip().replace(",", ""))
            except ValueError:
                pass

        if issuer or cusip:
            holdings.append(
                {
                    "issuer": issuer,
                    "cusip": cusip,
                    "titleOfClass": title,
                    "value": value,
                    "shares": shares,
                }
            )

    return holdings


def scrape_investor(
    investor: dict[str, str],
    existing_dates: set[str],
    default_quarters: int = 2,
) -> list[ScrapedFiling]:
    """
    Scrapes 13F data for a single investor.

    Fetches `default_quarters` quarters until the DB has enough history for
    trend detection. Once the target is reached, fetches only the most recent
    quarter to pick up new filings and amendments.
    """
    name = investor["name"]
    cik = investor["cik"]

    # Fetch default_quarters until we have enough history; then just fetch 1 (latest)
    fetch_count = default_quarters if len(existing_dates) < default_quarters else 1
    logger.info(
        "Fetching 13F for %s (CIK %s) — %d quarter(s) [%d in DB]",
        name,
        cik,
        fetch_count,
        len(existing_dates),
    )
    sleep(REQUEST_DELAY)

    filings_metadata = get_recent_13f_filings(cik, max_filings=fetch_count)
    if not filings_metadata:
        logger.warning("No 13F-HR filing found for %s", name)
        return []

    results: list[ScrapedFiling] = []
    for metadata in filings_metadata:
        accession = metadata["accessionNumber"]
        report_date = metadata["reportDate"]

        sleep(REQUEST_DELAY)
        index_data = get_filing_index(cik, accession)
        if not index_data:
            logger.warning("Could not fetch index for %s accession %s", name, accession)
            continue

        xml_filename = find_info_table_xml(index_data)
        if not xml_filename:
            logger.warning("No info table XML found for %s", name)
            continue

        sleep(REQUEST_DELAY)
        xml_content = download_xml(cik, accession, xml_filename)
        if not xml_content:
            logger.warning("Could not download XML for %s", name)
            continue

        holdings = parse_13f_xml(xml_content)
        # Scale values for pre-2023 filings: SEC reported in thousands before 2023
        try:
            filing_date = date.fromisoformat(report_date)
        except ValueError:
            filing_date = datetime.now().date()

        total_value = sum(h["value"] for h in holdings)

        # Scale by 1000 if date < 2023 OR if total_value < $100M.
        # $100M is the minimum AUM required to file a 13F. If the raw sum is < 100M,
        # the manager definitely mistakenly filled out the form in thousands.
        if 0 < total_value < 100_000_000:
            holdings = [{**h, "value": h["value"] * 1000} for h in holdings]
            total_value *= 1000
            logger.warning(
                "  Non-compliant 2024+ filing for %s (%s): raw AUM %s < 100M. Scaled ×1000",
                name,
                report_date,
                f"${total_value / 1000:,}",
            )

        results.append(
            {
                "investor": name,
                "cik": cik,
                "reportDate": report_date,
                "form": metadata["form"],
                "accessionNumber": accession,
                "holdingsCount": len(holdings),
                "totalValue": total_value,
                "holdings": holdings,
            }
        )
        logger.info("  %s: report %s, %d holdings, $%s", name, report_date, len(holdings), f"{total_value:,}")

    return results


def _get_existing_report_dates(session: Session, cik: str) -> set[str]:
    """Return ISO report dates already stored in DB for a manager (by CIK)."""
    normalized = _normalize_cik(cik)
    manager = session.execute(select(Form13FManager).where(Form13FManager.cik == normalized)).scalar_one_or_none()
    if not manager:
        return set()
    rows = session.execute(select(Form13FFiling.report_date).where(Form13FFiling.manager_id == manager.id)).all()
    return {row.report_date.isoformat() for row in rows}


def _build_cusip_to_instrument_map(session: Session) -> dict[str, int]:
    """Build CUSIP -> instrument_id from Instrument.isin.
    All ISINs follow the same 12-char structure: 2-char country code + 9-char CUSIP + 1 check digit.
    This works for US ISINs (US + CUSIP) and non-US cross-listings (e.g. CA11271J1075 -> 11271J107
    for Brookfield Corp BN). Keys are uppercased for case-insensitive lookup."""
    result = session.execute(select(Instrument.id, Instrument.isin).where(Instrument.isin.is_not(None)))
    mapping: dict[str, int] = {}
    for row in result.all():
        isin = (row.isin or "").strip().upper()
        if len(isin) < 11:
            continue
        cusip = ISIN_TO_CUSIP_OVERRIDES.get(isin, isin[2:11])
        mapping[cusip.upper()] = row.id
    return mapping


def _save_to_db(session: Session, results: list[ScrapedFiling]) -> None:
    """Saves scraped 13F data to database."""
    cusip_to_instrument = _build_cusip_to_instrument_map(session)
    matched_count = 0
    # Per CUSIP: [issuer from largest line, sum of 13F value this run, max single-line value]
    unmatched_agg: dict[str, list] = {}

    for r in results:
        cik = _normalize_cik(r["cik"])
        try:
            report_date = date.fromisoformat(r["reportDate"])
        except ValueError:
            logger.warning("Invalid report date %s for %s", r["reportDate"], r["investor"])
            continue

        # Get or create manager
        manager = session.execute(select(Form13FManager).where(Form13FManager.cik == cik)).scalar_one_or_none()
        if not manager:
            manager = Form13FManager(name=r["investor"], cik=cik)
            session.add(manager)
            session.flush()

        # Get or create filing (replace holdings if exists)
        filing = (
            session.execute(
                select(Form13FFiling).where(
                    Form13FFiling.manager_id == manager.id, Form13FFiling.report_date == report_date
                )
            )
        ).scalar_one_or_none()

        if filing:
            # Delete existing holdings (we're replacing)
            session.execute(delete(Form13FHolding).where(Form13FHolding.filing_id == filing.id))
            filing.accession_number = r["accessionNumber"]
            filing.form = r["form"]
            filing.total_value = r["totalValue"]
        else:
            filing = Form13FFiling(
                manager_id=manager.id,
                report_date=report_date,
                form=r["form"],
                accession_number=r["accessionNumber"],
                total_value=r["totalValue"],
            )
            session.add(filing)
            session.flush()

        # Insert holdings
        for h in r["holdings"]:
            cusip = h["cusip"].strip()
            if not cusip:
                continue
            issuer = h["issuer"].strip() or "Unknown"
            instrument_id = cusip_to_instrument.get(cusip.upper())
            if instrument_id:
                matched_count += 1
            else:
                key = cusip.upper()
                row = unmatched_agg.get(key)
                v = h["value"]
                if row is None:
                    unmatched_agg[key] = [issuer, v, v]
                else:
                    row[1] += v
                    if v > row[2]:
                        row[2] = v
                        row[0] = issuer
            holding = Form13FHolding(
                filing_id=filing.id,
                instrument_id=instrument_id,
                issuer=issuer,
                cusip=cusip,
                title_of_class=h["titleOfClass"].strip() or "COM",
                value=h["value"],
                shares=h["shares"],
            )
            session.add(holding)

    logger.info("Matched %d holdings to instruments via CUSIP", matched_count)
    if unmatched_agg:
        n_unmatched = len(unmatched_agg)
        keys_by_value = sorted(unmatched_agg, key=lambda k: unmatched_agg[k][1], reverse=True)
        logger.warning(
            "%d unique securities not matched to instruments (no instrument with matching US ISIN); "
            "top %d by aggregate 13F $ this run:",
            n_unmatched,
            min(UNMATCHED_LOG_LIMIT, n_unmatched),
        )
        for cusip_key in keys_by_value[:UNMATCHED_LOG_LIMIT]:
            issuer, total_val, _ = unmatched_agg[cusip_key]
            logger.warning("  - %s (CUSIP: %s) aggregate_13f_value=$%s", issuer, cusip_key, f"{total_val:,}")
        if n_unmatched > UNMATCHED_LOG_LIMIT:
            logger.warning("  (%d more unmatched not listed)", n_unmatched - UNMATCHED_LOG_LIMIT)


def main(limit: Optional[int] = None, default_quarters: int = 4) -> None:
    """
    Scrapes 13F data for all configured investors and saves to database.

    Targets `default_quarters` of history per investor (default 4 = 1 year).
    Once the DB has enough history, fetches only the latest quarter to pick up
    new filings and amendments efficiently.
    """
    investors = INVESTORS[:limit] if limit else INVESTORS
    results: list[ScrapedFiling] = []

    with get_session() as session:
        for inv in investors:
            existing_dates = _get_existing_report_dates(session, inv["cik"])
            try:
                data_list = scrape_investor(inv, existing_dates=existing_dates, default_quarters=default_quarters)
                results.extend(data_list)
            except Exception as e:
                logger.exception("Error scraping %s: %s", inv["name"], e)

        if results:
            _save_to_db(session, results)
            logger.info("Saved %d filings to database", len(results))
        else:
            logger.warning("No new 13F data scraped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape SEC 13F filings for institutional investors")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of investors to scrape (default: all)",
    )
    parser.add_argument(
        "--quarters",
        type=int,
        default=4,
        help="Target quarters of history per investor (default: 4 for trend detection)",
    )
    args = parser.parse_args()
    main(limit=args.limit, default_quarters=args.quarters)
