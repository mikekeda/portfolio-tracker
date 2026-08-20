"""
Scrape SEC Form 13F Data
======================
Fetches 13F-HR filings from SEC EDGAR for a curated list of institutional investors,
parses the information table XML, and saves to database (Form13FManager, Form13FFiling, Form13FHolding).

13F filings are quarterly disclosures by institutional managers with $100M+ AUM.
Signals: new positions and large increases are more reliable than sells.

One quarter can span several filings. A 13F-HR/A of type NEW HOLDINGS is a delta added to the
original, not a replacement; a 13F-NT means the manager now reports under another CIK, which
has to be added to its INVESTORS entry. Filings from a manager's CIKs merge into one per quarter.

Note: Value is in dollars (nearest dollar) for filings from Jan 2023 onward.
Pre-2023 filings may report value in thousands.

SEC requires User-Agent: "CompanyName admin@example.com"
"""

from datetime import date
from time import sleep
from typing import Optional, TypedDict

import requests
from lxml import etree
from sqlalchemy import delete, func, select
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
    "GB0031348658": "06738E204",  # Barclays
    "GB0007980591": "055622104",  # BP
    "GB0002374006": "25243Q205",  # Diageo
    "GB00BM8PJY71": "639057207",  # NatWest
    "GB0008706128": "539439109",  # Lloyds
    "IL0010811243": "M3760D101",  # Elbit Systems
    "AU0000185993": "Q4982L109",  # IREN
    "IL0011974909": "M7518J104",  # Oddity
    "MU0295S00016": "V5633W109",  # MakeMyTrip
    # 2026-07-07 — confirmed against prod DB (existing NULL rows fixed via one-off UPDATE)
    "IE00028FXN24": "G8267P108",  # Smurfit WestRock plc
    "GB00BFMBMT84": "G8060N102",  # Sensata Technologies (UK plc, NYSE ST)
    "GB00BRXH2664": "G0378L100",  # AngloGold Ashanti plc
    "IL0011301780": "M98068105",  # Wix.com Ltd
    "IL0011684185": "M6191J100",  # JFrog Ltd
    "IE00BDVJJQ56": "G6700G107",  # nVent Electric plc
    "LU0974299876": "L44385109",  # Globant S.A.
    "IL0010824113": "M22465104",  # Check Point Software
    # 2026-08-20 — instruments with no 13F match at all, found by cross-checking every monitored
    # instrument against unmatched CUSIPs. CINS lines (non-US domicile) and US ADR lines.
    "NL0009805522": "N97284108",  # Nebius Group N.V.
    "NL0009538784": "N6596X109",  # NXP Semiconductors N.V.
    "IE00BKVD2N49": "G7997R103",  # Seagate Technology Holdings plc
    "JE00BTDN8H13": "G3265R107",  # Aptiv plc (Jersey)
    "JE00BJ1F3079": "G0250X107",  # Amcor plc (Jersey)
    "IE00BFRT3W74": "G0176J109",  # Allegion plc
    "IE00BFY8C754": "G8473T100",  # Steris plc
    "GB00BMHVL512": "G5279N105",  # Klarna Group plc
    "IL0011858912": "M7S64L123",  # Pagaya Technologies Ltd
    "IL0011741688": "M5216V106",  # Global-E Online Ltd
    "IL0011582033": "M4R82T106",  # Fiverr International Ltd
    "IL0010952641": "M20791105",  # Camtek Ltd
    "DE0007164600": "803054204",  # SAP SE (Xetra line vs US ADR CUSIP)
    "GB00BP6MXD84": "780259305",  # Shell plc
    "GB00B2B0DG97": "759530108",  # RELX plc
    "GB0007099541": "74435K204",  # Prudential plc
    "GB00BDR05C01": "636274409",  # National Grid plc
    "GB0009223206": "83175M205",  # Smith & Nephew plc
    "GB0005405286": "404280406",  # HSBC Holdings plc
    "GB0007188757": "767204100",  # Rio Tinto plc
    "FR0000120271": "F92124100",  # TotalEnergies SE
    "US4228061093": "422806208",  # Heico Corp — managers file only the Class A line
}

# Some issuers file under multiple CUSIPs (e.g. ADR + ordinary-share lines), but the
# dict above allows only one CUSIP per ISIN: AstraZeneca GB0009895292 already maps to
# the ADR CUSIP 046353108, so its ordinary-share line (CUSIP G0593M107, ~$1.46B across
# Q1'26 filings) would otherwise stay unmatched. Aliases here map additional CUSIPs to
# the ISIN whose instrument they should resolve to.
EXTRA_CUSIP_TO_ISIN_ALIASES: dict[str, str] = {
    "G0593M107": "GB0009895292",  # AstraZeneca PLC ordinary shares -> same instrument as ADR line
    # 2026-08-20 — second lines for instruments that already match on another CUSIP.
    "904767704": "GB00BVZK7T90",  # Unilever PLC — older ADR line alongside 904767803
    "G2004J103": "PA1436583006",  # Carnival plc — dual-listed twin of Carnival Corp (143658300)
    "K08588103": "US04351P1012",  # Ascendis Pharma A/S Danish ordinary -> same instrument as ADR
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

# "cik" is the primary (stored on Form13FManager); optional "ciks" lists every CIK whose filings
# belong to this manager, merged per quarter, for managers that re-filed under a new entity.
INVESTORS: list[dict[str, str | list[str]]] = [
    # Value / long-term
    {"name": "Berkshire Hathaway", "cik": "1067983"},
    {"name": "Baupost Group", "cik": "1061768"},
    {"name": "Pershing Square", "cik": "1336528", "ciks": ["1336528", "2026053"]},
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


def normalize_cusip(cusip: str) -> str:
    """Canonical CUSIP form for storage and matching."""
    return cusip.strip().upper()


def investor_ciks(investor: dict[str, str | list[str]]) -> list[str]:
    """Every CIK to scrape for an investor, primary first."""
    ciks = investor.get("ciks")
    if isinstance(ciks, list) and ciks:
        return [str(c) for c in ciks]
    return [str(investor["cik"])]


def select_recent_filings(recent: dict, max_filings: int) -> list[FilingMetadata]:
    """
    Pick every 13F in a submissions `filings.recent` block covering its newest report dates.

    Keeps all filings for each of the `max_filings` newest report dates rather than one per
    date, so a quarter's original and its amendments both surface for the caller to combine.
    13F-NT rows are kept so a manager that stopped filing under this CIK is detectable rather
    than looking like a late filer.
    """
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    report_dates = recent.get("reportDate", [])

    candidates = [
        (report_dates[i] if i < len(report_dates) else "", i)
        for i, form in enumerate(forms)
        if form in ("13F-HR", "13F-HR/A", "13F-NT")
    ]
    # Chosen by report date, not position: EDGAR orders by filing date, so an amendment for an
    # older quarter can precede a newer quarter's original.
    wanted = set(sorted({report_date for report_date, _ in candidates}, reverse=True)[:max_filings])

    return [
        {
            "accessionNumber": accessions[i] if i < len(accessions) else "",
            "primaryDocument": primary_docs[i] if i < len(primary_docs) else "",
            "form": forms[i],
            "reportDate": report_date,
        }
        for report_date, i in candidates
        if report_date in wanted
    ]


def get_recent_13f_filings(cik: str, max_filings: int = 2) -> list[FilingMetadata]:
    """Fetch 13F metadata for the `max_filings` most recent report dates of a CIK."""
    cik = _normalize_cik(cik)
    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return select_recent_filings(response.json().get("filings", {}).get("recent", {}), max_filings)


def get_filing_index(cik: str, accession: str) -> Optional[dict]:
    """Fetches the filing index.json to list available documents."""
    cik = _normalize_cik(cik)
    accession_clean = accession.replace("-", "")
    url = SEC_ARCHIVES_URL.format(cik=cik, accession=accession_clean, filename="index.json")
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    if response.status_code != 200:
        return None
    return response.json()


def primary_doc_name(primary_document: str) -> str:
    """
    Raw cover-page filename from a submissions `primaryDocument` value.

    EDGAR reports 13F cover pages as `xslForm13F_X02/primary_doc.xml`; that path renders the
    document as HTML through a stylesheet, while the parseable XML sits at the bare basename.
    """
    return primary_document.rsplit("/", 1)[-1] or "primary_doc.xml"


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

    # Otherwise use largest XML (info table is typically the biggest), but never rank the cover
    # page: on a small amendment primary_doc.xml outweighs a four-row info table.
    ranked = sorted((f for f in xml_files if primary_doc_name(f[0]).lower() != "primary_doc.xml"), key=lambda x: -x[1])
    if ranked:
        return ranked[0][0]
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


def _parse_xml(xml_content: str) -> Optional[etree._Element]:
    """
    Parse a filing document, or None when it is not well-formed XML.

    EDGAR renders some document paths as HTML, so callers must degrade rather than raise: an
    unreadable cover page has to reach amendment_disposition's "skip" branch, not abort the
    whole investor.
    """
    try:
        return etree.fromstring(xml_content.encode("utf-8"))
    except etree.XMLSyntaxError as e:
        logger.warning("Not well-formed XML (%s) — treating the document as unreadable", e)
        return None


def parse_13f_xml(xml_content: str) -> list[ParsedHolding]:
    """
    Parses 13F information table XML and extracts holdings.
    Handles both namespaced and non-namespaced documents.

    Note: Value is in dollars (nearest dollar) for filings from Jan 2023 onward.
    Pre-2023 filings may report value in thousands; caller may need to scale.
    """
    holdings: list[ParsedHolding] = []
    root = _parse_xml(xml_content)
    if root is None:
        return holdings

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


def _first_text(xml_content: str, tag: str) -> str:
    """Text of the first element with this local name, or empty."""
    root = _parse_xml(xml_content)
    if root is None:
        return ""
    found = root.xpath(f".//*[local-name()='{tag}']")
    return (found[0].text or "").strip() if found else ""


def parse_amendment_type(xml_content: str) -> str:
    """
    Amendment type from a 13F primary_doc.xml: "RESTATEMENT", "NEW HOLDINGS", or "".

    RESTATEMENT replaces the whole info table; NEW HOLDINGS carries only rows not previously
    reported (typically confidential-treatment positions being released), so it must be added
    to the original filing rather than replacing it.
    """
    return _first_text(xml_content, "amendmentType").upper()


def amendment_disposition(form: str, amendment_type: str) -> str:
    """
    How one filing combines with a quarter's existing rows: "replace", "merge", or "skip".

    An original states the full table. An amendment replaces it only when SEC-classified as a
    RESTATEMENT; NEW HOLDINGS is a delta to merge. Anything else — usually an unreadable
    primary_doc, which parses to "" — is skipped: guessing "replace" would let a four-row
    delta stand in for the whole quarter.
    """
    if form != "13F-HR/A":
        return "replace"
    return {"RESTATEMENT": "replace", "NEW HOLDINGS": "merge"}.get(amendment_type, "skip")


def reported_in_thousands(total_value: int, is_partial: bool) -> bool:
    """
    Whether a filing's values look like the pre-2023 thousands convention rather than dollars.

    $100M is the minimum AUM that obliges a manager to file at all, so a whole portfolio below
    it was reported in thousands. A NEW HOLDINGS amendment is a delta, not a portfolio, and can
    legitimately be one small lot — scaling that inflates the merged quarter a thousandfold.
    """
    return not is_partial and 0 < total_value < 100_000_000


def parse_nt_successor_ciks(xml_content: str) -> list[str]:
    """CIKs listed as other managers on a 13F-NT — where this manager's holdings now report."""
    root = _parse_xml(xml_content)
    if root is None:
        return []
    return [
        (el.text or "").strip()
        for el in root.xpath(".//*[local-name()='otherManager']/*[local-name()='cik']")
        if (el.text or "").strip()
    ]


def merge_new_holdings(base: ScrapedFiling, amendment: ScrapedFiling) -> ScrapedFiling:
    """
    Add a NEW HOLDINGS amendment's rows to the original filing.

    Rows are appended, never deduplicated by CUSIP: an amendment can report a further lot of a
    CUSIP the original already carries, and duplicate CUSIP rows per filing are expected anyway
    (share classes, otherManager splits). Keeps the original's form and accession so the row
    still identifies as the full filing.
    """
    holdings = base["holdings"] + amendment["holdings"]
    return {
        **base,
        "holdingsCount": len(holdings),
        "totalValue": base["totalValue"] + amendment["totalValue"],
        "holdings": holdings,
    }


def merge_same_quarter(filings: list[ScrapedFiling], primary_cik: str) -> list[ScrapedFiling]:
    """
    Combine one manager's filings from several CIKs into one filing per report date.

    Sets every result's cik to the primary so _save_to_db resolves a single Form13FManager,
    keeping one filing per manager per quarter as uq_form13f_manager_report_date requires.
    """
    merged: dict[str, ScrapedFiling] = {}
    for filing in filings:
        report_date = filing["reportDate"]
        existing = merged.get(report_date)
        if existing is None:
            merged[report_date] = {**filing, "cik": primary_cik}
            continue
        holdings = existing["holdings"] + filing["holdings"]
        merged[report_date] = {
            **existing,
            "holdingsCount": len(holdings),
            "totalValue": existing["totalValue"] + filing["totalValue"],
            "holdings": holdings,
        }
    return list(merged.values())


def _fetch_filing(name: str, cik: str, metadata: FilingMetadata, is_partial: bool = False) -> Optional[ScrapedFiling]:
    """Download and parse one filing's information table.

    Set `is_partial` for an amendment that reports only added rows, so its total is not treated
    as a whole portfolio.
    """
    accession = metadata["accessionNumber"]
    report_date = metadata["reportDate"]

    sleep(REQUEST_DELAY)
    index_data = get_filing_index(cik, accession)
    if not index_data:
        logger.warning("Could not fetch index for %s accession %s", name, accession)
        return None

    xml_filename = find_info_table_xml(index_data)
    if not xml_filename:
        logger.warning("No info table XML found for %s accession %s", name, accession)
        return None

    sleep(REQUEST_DELAY)
    xml_content = download_xml(cik, accession, xml_filename)
    if not xml_content:
        logger.warning("Could not download XML for %s accession %s", name, accession)
        return None

    holdings = parse_13f_xml(xml_content)
    total_value = sum(h["value"] for h in holdings)

    if reported_in_thousands(total_value, is_partial):
        holdings = [{**h, "value": h["value"] * 1000} for h in holdings]
        total_value *= 1000
        logger.warning(
            "  Non-compliant filing for %s (%s): raw AUM %s < 100M. Scaled ×1000",
            name,
            report_date,
            f"${total_value / 1000:,}",
        )

    return {
        "investor": name,
        "cik": cik,
        "reportDate": report_date,
        "form": metadata["form"],
        "accessionNumber": accession,
        "holdingsCount": len(holdings),
        "totalValue": total_value,
        "holdings": holdings,
    }


def _fetch_amendment_type(cik: str, metadata: FilingMetadata) -> str:
    """Amendment type for a 13F-HR/A, or empty when the primary document is unreadable."""
    sleep(REQUEST_DELAY)
    xml_content = download_xml(cik, metadata["accessionNumber"], primary_doc_name(metadata["primaryDocument"]))
    return parse_amendment_type(xml_content) if xml_content else ""


def _report_notice(name: str, cik: str, metadata: FilingMetadata, known_ciks: list[str]) -> None:
    """Log a 13F-NT — the manager's holdings now report under a different CIK."""
    sleep(REQUEST_DELAY)
    xml_content = download_xml(cik, metadata["accessionNumber"], primary_doc_name(metadata["primaryDocument"]))
    successors = parse_nt_successor_ciks(xml_content) if xml_content else []
    tracked = {_normalize_cik(c) for c in known_ciks}
    missing = [s for s in successors if _normalize_cik(s) not in tracked]

    if successors and not missing:
        logger.info(
            "  %s (CIK %s) filed a 13F-NT for %s; successor CIK(s) %s already tracked",
            name,
            cik,
            metadata["reportDate"],
            ", ".join(successors),
        )
        return

    logger.error(
        "%s (CIK %s) filed a 13F-NT for %s — holdings now report under CIK(s) %s. "
        "Add them to this investor's 'ciks' in INVESTORS, or the manager silently stops updating.",
        name,
        cik,
        metadata["reportDate"],
        ", ".join(missing) or "unknown (see the filing's otherManagersInfo)",
    )


def _scrape_cik(name: str, cik: str, fetch_count: int, known_ciks: list[str]) -> list[ScrapedFiling]:
    """Scrape one CIK, resolving each report date's original and amendments into one filing."""
    sleep(REQUEST_DELAY)
    filings_metadata = get_recent_13f_filings(cik, max_filings=fetch_count)
    if not filings_metadata:
        logger.warning("No 13F filing found for %s (CIK %s)", name, cik)
        return []

    by_date: dict[str, list[FilingMetadata]] = {}
    for metadata in filings_metadata:
        by_date.setdefault(metadata["reportDate"], []).append(metadata)

    results: list[ScrapedFiling] = []
    for report_date, metas in by_date.items():
        originals = [m for m in metas if m["form"] == "13F-HR"]
        # EDGAR lists newest first; apply amendments oldest-first so the newest wins a restatement.
        amendments = list(reversed([m for m in metas if m["form"] == "13F-HR/A"]))
        if not originals and not amendments:
            _report_notice(name, cik, metas[0], known_ciks)
            continue

        filing: Optional[ScrapedFiling] = None
        for metadata in originals[:1] + amendments:
            amendment_type = _fetch_amendment_type(cik, metadata) if metadata["form"] == "13F-HR/A" else ""
            disposition = amendment_disposition(metadata["form"], amendment_type)
            if disposition == "skip":
                logger.error(
                    "%s %s: cannot classify amendment %s (amendmentType %r) — keeping the original, "
                    "since a NEW HOLDINGS delta would otherwise replace the full filing",
                    name,
                    report_date,
                    metadata["accessionNumber"],
                    amendment_type,
                )
                continue

            parsed = _fetch_filing(name, cik, metadata, is_partial=disposition == "merge")
            if parsed is None:
                continue

            if disposition == "replace":
                filing = parsed
            elif filing is None:
                logger.error(
                    "%s %s: NEW HOLDINGS amendment %s has no readable original — skipping, "
                    "storing the delta alone would look like the whole quarter",
                    name,
                    report_date,
                    metadata["accessionNumber"],
                )
            else:
                filing = merge_new_holdings(filing, parsed)
                logger.info("  %s: %s amendment added %d holding(s)", name, report_date, parsed["holdingsCount"])

        if filing is None:
            continue
        results.append(filing)
        logger.info(
            "  %s: report %s, %d holdings, $%s",
            name,
            report_date,
            filing["holdingsCount"],
            f"{filing['totalValue']:,}",
        )

    return results


def scrape_investor(
    investor: dict[str, str | list[str]],
    existing_dates: set[str],
    default_quarters: int = 2,
) -> list[ScrapedFiling]:
    """
    Scrapes 13F data for a single investor across all of its CIKs.

    Fetches `default_quarters` quarters until the DB has enough history for
    trend detection. Once the target is reached, fetches only the most recent
    quarter to pick up new filings and amendments.
    """
    name = str(investor["name"])
    primary_cik = str(investor["cik"])
    ciks = investor_ciks(investor)

    # existing_dates counts the manager's stored quarters, which says nothing about how much of
    # a secondary CIK we hold — fetch the full window whenever more than one CIK is in play.
    fetch_count = default_quarters if len(existing_dates) < default_quarters or len(ciks) > 1 else 1
    logger.info(
        "Fetching 13F for %s (CIK %s) — %d quarter(s) [%d in DB]",
        name,
        ", ".join(ciks),
        fetch_count,
        len(existing_dates),
    )

    scraped: list[ScrapedFiling] = []
    for cik in ciks:
        scraped.extend(_scrape_cik(name, cik, fetch_count, ciks))
    return merge_same_quarter(scraped, primary_cik)


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
    for Brookfield Corp BN). Keys are uppercased for case-insensitive lookup.
    EXTRA_CUSIP_TO_ISIN_ALIASES adds further CUSIPs for issuers filed under more than one
    (e.g. ADR + ordinary-share lines)."""
    result = session.execute(select(Instrument.id, Instrument.isin).where(Instrument.isin.is_not(None)))
    mapping: dict[str, int] = {}
    isin_to_id: dict[str, int] = {}
    for row in result.all():
        isin = (row.isin or "").strip().upper()
        if len(isin) < 11:
            continue
        isin_to_id[isin] = row.id
        cusip = ISIN_TO_CUSIP_OVERRIDES.get(isin, isin[2:11])
        mapping[cusip.upper()] = row.id
    for alias_cusip, alias_isin in EXTRA_CUSIP_TO_ISIN_ALIASES.items():
        instrument_id = isin_to_id.get(alias_isin.upper())
        if instrument_id is not None:
            mapping[alias_cusip.upper()] = instrument_id
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
            # An empty parse must never wipe a good quarter — that is how Berkshire's Q1 2025
            # book (110 holdings, $258.7bn) was replaced by a zero-row row.
            if not r["holdings"]:
                existing = session.execute(
                    select(func.count()).select_from(Form13FHolding).where(Form13FHolding.filing_id == filing.id)
                ).scalar_one()
                if existing:
                    logger.error(
                        "Refusing to overwrite %s %s (%d holdings) with a filing that parsed to none — "
                        "accession %s",
                        r["investor"],
                        r["reportDate"],
                        existing,
                        r["accessionNumber"],
                    )
                    continue

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
            cusip = normalize_cusip(h["cusip"])
            if not cusip:
                continue
            issuer = h["issuer"].strip() or "Unknown"
            instrument_id = cusip_to_instrument.get(cusip)
            if instrument_id:
                matched_count += 1
            else:
                key = cusip
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
            existing_dates = _get_existing_report_dates(session, str(inv["cik"]))
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
