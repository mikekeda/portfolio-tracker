# ─── Exchange‑suffix + alias tables ───────────────────────────────────────────
STOCKS_SUFFIX: dict[str, str] = {
    "US": "",      # NYSE / Nasdaq
    "l": ".L",    # London
    "p": ".PA",   # Paris
    "d": ".DE",   # Xetra
    "CA": ".TO",  # Toronto
    "ST": ".ST",  # Stockholm
    "BR": ".BR",  # Brussels
    "AS": ".AS",  # Amsterdam
    "MI": ".MI",  # Milan
    "SW": ".SW",  # Zurich
    "HK": ".HK",  # Hong‑Kong
    "VX": ".SW",  # Virt‑X (shares Zurich feed)
}

STOCKS_ALIASES: dict[str, str] = {
    "FB": "META",
    "UTX": "RTX",
    "PCLN": "BKNG",
    "RBS": "NWG",
    "VACQ": "RKLB",
    "NPA": "ASTS",
    "IPOE": "SOFI",
    "GNPK": "RDW",
    "IPAX": "LUNR",
    "AHAC": "HUMA",
    "LSE": "LSEG",
    "IPOB": "OPEN",
    "CCIR": "KYIV",
}


STOCKS_DELISTED: set[str] = {"HYUDl_EQ"}


ETF_COUNTRY_ALLOCATION = {
    "XUTC.L": {                 # Xtrackers MSCI USA IT
        "United States": 97.88,
        "Ireland": 1.40,
        "Netherlands": 0.34,
        "Australia": 0.20,
        "Singapore": 0.18,
    },
    "XNAS.L": {                 # Xtrackers Nasdaq-100
        "United States": 95.00,
        "United Kingdom": 1.92,
        "Canada": 0.99,
        "Netherlands": 0.74,
        "Uruguay": 0.74,
        "Other": 0.61,
    },
    "R2SC.L": {                 # SPDR Russell 2000 US Small Cap
        "United States": 94.75,
        "Bermuda": 0.84,
        "Cayman Islands": 0.65,
        "Canada": 0.57,
        "United Kingdom": 0.55,
        "Other": 2.64,
    },
    "UKDV.L": {                 # SPDR UK Dividend Aristocrats
        "United Kingdom": 100.00,
    },
    "VUAG.L": {                 # Vanguard S&P 500 (USD Acc)
        "United States": 97.48,
        "Ireland": 1.40,
        "United Kingdom": 0.55,
        "Switzerland": 0.26,
        "Netherlands": 0.11,
        "Other": 0.2,
    },
    "R1GR.L": {                 # iShares Russell 1000 Growth
        "United States": 100.00,
    },
    "CIBR.L": {                 # First Trust NASDAQ Cybersecurity
        "United States": 91.70,
        "Israel": 3.58,
        "Canada": 2.34,
        "Japan": 2.15,
        "United Kingdom": 0.22,
        "Other": 0.01,
    },
    "BOTZ.L": {                 # Global X Robotics & AI UCITS
        "United States": 49.7,
        "Japan": 28.8,
        "Switzerland": 9.5,
        "China": 2.7,
        "Finland": 2.4,
        "South Korea": 2.0,
        "Hong Kong": 1.5,
        "Canada": 1.3,
        "United Kingdom": 1.2,
        "Norway": 0.9,
    },
    "CBRX.L": {
        "United States": 74.78,
        "Canada": 6.77,
        "Israel": 6.72,
        "South Korea": 3.18,
    },
}


ETF_SECTOR_ALLOCATION = {
    "XUTC.L": {
        "Technology": 100.0,
    },
    "XNAS.L": {
        "Technology": 51.16,
        "Communication Services": 16.05,
        "Consumer Cyclical": 13.21,
        "Consumer Defensive": 6.13,
        "Healthcare": 5.62,
        "Industrials": 3.75,
        "Utilities": 1.47,
        "Basic Materials": 1.46,
        "Energy": 0.50,
        "Financial Services": 0.44,
        "Real Estate": 0.21,
    },
    "R2SC.L": {
        "Financial Services": 18.06,
        "Healthcare": 16.96,
        "Industrials": 16.02,
        "Technology": 14.66,
        "Consumer Cyclical": 8.96,
        "Real Estate": 7.25,
        "Consumer Defensive": 4.41,
        "Energy": 4.41,
        "Basic Materials": 3.98,
        "Utilities": 3.17,
        "Communication Services": 2.10,
        "Other": 0.02,
    },
    "UKDV.L": {
        "Financial Services": 28.31,
        "Industrials": 16.15,
        "Consumer Defensive": 11.35,
        "Consumer Cyclical": 11.14,
        "Utilities": 10.85,
        "Healthcare": 7.45,
        "Real Estate": 4.86,
        "Communication Services": 4.47,
        "Technology": 3.75,
        "Basic Materials": 1.67,
    },
    "VUAG.L": {
        "Technology": 31.68,
        "Financial Services": 14.04,
        "Healthcare": 10.85,
        "Consumer Cyclical": 10.38,
        "Communication Services": 9.46,
        "Industrials": 7.66,
        "Consumer Defensive": 6.15,
        "Energy": 3.18,
        "Utilities": 2.56,
        "Real Estate": 2.25,
        "Basic Materials": 1.77,
        "Other": 0.02,
    },
    "R1GR.L": {
        "Technology": 48.75,
        "Consumer Cyclical": 14.28,
        "Communication Services": 13.48,
        "Healthcare": 7.59,
        "Financial Services": 6.85,
        "Consumer Defensive": 3.68,
        "Industrials": 3.51,
        "Basic Materials": 0.59,
        "Real Estate": 0.56,
        "Energy": 0.46,
        "Utilities": 0.24,
        "Other": 0.01,
    },
    "CIBR.L": {
        "Technology": 95.42,
        "Industrials": 4.58,
    },
    "BOTZ.L": {
        "Technology": 41.83,
        "Industrials": 41.81,
        "Healthcare": 12.76,
        "Financial Services": 2.10,
        "Consumer Cyclical": 1.07,
        "Energy": 0.43,
    },
    "CBRX.L": {
        "Technology": 93.49,
        "Communication Services": 4.97,
    },
}

