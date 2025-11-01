#!/usr/bin/env python3
"""
Backfill CurrencyRateDaily with historical currency rates.

This script fetches historical exchange rates for USD/GBP, EUR/GBP, and CAD/GBP
from April 18, 2024 to October 11, 2025 using Yahoo Finance, and stores them
in the CurrencyRateDaily table.
"""

from datetime import date, datetime

import pandas as pd
import yfinance as yf  # type: ignore[import-untyped]

from models import CurrencyRateDaily
from scripts.update_data import get_session


def fetch_currency_data(symbol: str, start_date: str, end_date: str, invert: bool = False) -> pd.Series:
    """
    Fetch currency data from Yahoo Finance.

    Args:
        symbol: Yahoo Finance symbol (e.g., "GBPUSD=X")
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        invert: Whether to invert the rate (1/rate)

    Returns:
        pandas Series with dates as index and rates as values
    """
    try:
        print(f"  📈 Fetching {symbol} from {start_date} to {end_date}...")

        # Fetch data from Yahoo Finance
        data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)

        if data.empty:
            print(f"  ⚠️  No data found for {symbol}")
            return pd.Series(dtype=float)

        # Debug: print DataFrame structure
        print(f"  🔍 Debug: DataFrame columns: {list(data.columns)}")
        print(f"  🔍 Debug: DataFrame shape: {data.shape}")
        print("  🔍 Debug: First few rows:")
        print(data.head(3))

        # Get the Close column - yfinance always returns DataFrame with Close column
        if "Close" in data.columns:
            rates = data["Close"]
        else:
            # If no Close column, get the first column
            rates = data.iloc[:, 0]

        # Invert if needed (e.g., GBP/USD -> USD/GBP)
        if invert:
            rates = 1 / rates

        # Remove NaN values
        rates = rates.dropna()

        print(f"  ✅ Fetched {len(rates)} rates for {symbol}")
        return rates

    except Exception as e:
        print(f"  ❌ Error fetching {symbol}: {e}")
        return pd.Series(dtype=float)


def store_currency_rate(session, rate_date: date, from_currency: str, to_currency: str, rate: float) -> bool:
    """
    Store a currency rate in the database.
    """
    try:
        # Create new rate entry
        currency_rate = CurrencyRateDaily(
            date=rate_date,
            from_currency=from_currency,
            to_currency=to_currency,
            rate=rate,
        )

        session.add(currency_rate)
        return True

    except Exception as e:
        # If it's a duplicate key error, that's fine - just skip
        if "duplicate key" in str(e).lower() or "unique constraint" in str(e).lower():
            return False
        print(f"    ❌ Error storing rate for {from_currency}/{to_currency} on {rate_date}: {e}")
        return False


def backfill_currency_pair(
    session,
    from_currency: str,
    to_currency: str,
    yahoo_symbol: str,
    start_date: str,
    end_date: str,
    invert: bool = False,
) -> dict[str, int]:
    """
    Backfill currency rates for a specific pair.

    Args:
        session: Database session
        from_currency: Source currency
        to_currency: Target currency
        yahoo_symbol: Yahoo Finance symbol
        start_date: Start date string
        end_date: End date string
        invert: Whether to invert the rates

    Returns:
        Dictionary with statistics
    """
    print(f"\n💱 Processing {from_currency}/{to_currency}...")

    # Fetch data from Yahoo Finance
    rates = fetch_currency_data(yahoo_symbol, start_date, end_date, invert)

    if rates.empty:
        return {"fetched": 0, "stored": 0, "skipped": 0}

    # Debug: print what we actually got
    print(f"  🔍 Debug: rates type: {type(rates)}")
    print(f"  🔍 Debug: rates length: {len(rates)}")

    # Store rates in database
    stored_count = 0
    skipped_count = 0
    processed_count = 0

    # If rates is a DataFrame, get the first column as a Series
    if isinstance(rates, pd.DataFrame):
        if len(rates.columns) > 0:
            rates_series = rates.iloc[:, 0]  # Get first column as Series
        else:
            print("  ⚠️  No columns in DataFrame")
            return {"fetched": 0, "stored": 0, "skipped": 0}
    else:
        rates_series = rates

    print(f"  🔍 Debug: rates_series type: {type(rates_series)}")
    print("  🔍 Debug: first 3 items from series:")
    for i, (date_key, value) in enumerate(rates_series.items()):
        if i >= 3:
            break
        print(f"    {i + 1}. Date: {date_key} (type: {type(date_key)}), Value: {value} (type: {type(value)})")

    for rate_date, rate_value in rates_series.items():
        processed_count += 1

        # Skip if rate_date is not a date (e.g., if it's the symbol name)
        if not isinstance(rate_date, (pd.Timestamp, datetime)):
            print("    ⏭️  Skipping non-date item: {rate_date}")
            continue

        # Convert to scalar if it's a Series
        if hasattr(rate_value, "iloc"):
            rate_value = rate_value.iloc[0] if len(rate_value) > 0 else None

        if pd.isna(rate_value) or rate_value <= 0:
            print(f"    ⏭️  Skipping invalid rate: {rate_value}")
            continue

        # Convert rate_date to date object
        if hasattr(rate_date, "date"):
            rate_date_python = rate_date.date()
        else:
            # If it's already a string or other format, parse it
            rate_date_python = pd.to_datetime(rate_date).date()

        # Debug: show what we're trying to store
        if processed_count <= 5:  # Show first 5 attempts
            print(f"    🔍 Attempting to store: {rate_date_python} {from_currency}/{to_currency} = {rate_value}")

        # Store in database
        if store_currency_rate(session, rate_date_python, from_currency, to_currency, float(rate_value)):
            stored_count += 1
            if stored_count <= 3:  # Show first 3 successful stores
                print(f"    ✅ Stored: {rate_date_python} {from_currency}/{to_currency} = {rate_value}")
        else:
            skipped_count += 1
            if skipped_count <= 3:  # Show first 3 skips
                print(f"    ⏭️  Skipped (exists): {rate_date_python} {from_currency}/{to_currency}")

    print(f"  📊 Processed {processed_count} items total")

    print(f"  📊 Results: {len(rates)} fetched, {stored_count} stored, {skipped_count} skipped")

    return {"fetched": len(rates), "stored": stored_count, "skipped": skipped_count}


def main():
    """Main function to backfill currency rates."""
    print("=" * 80)
    print("CURRENCY RATE BACKFILL SCRIPT")
    print("=" * 80)
    print()

    # Define date range
    start_date_str = "2024-04-18"
    end_date_str = "2025-08-29"
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    print(f"📅 Date range: {start_date} to {end_date}")
    print(f"📊 Total days: {(end_date - start_date).days + 1}")
    print()

    # Currency pairs to backfill
    currency_pairs: list[dict[str, str | bool]] = [
        {
            "from_currency": "USD",
            "to_currency": "GBP",
            "yahoo_symbol": "GBPUSD=X",
            "invert": True,  # Convert GBP/USD to USD/GBP
            "description": "US Dollar to British Pound",
        },
        {
            "from_currency": "EUR",
            "to_currency": "GBP",
            "yahoo_symbol": "EURGBP=X",
            "invert": False,  # Direct EUR/GBP
            "description": "Euro to British Pound",
        },
        {
            "from_currency": "CAD",
            "to_currency": "GBP",
            "yahoo_symbol": "GBPCAD=X",
            "invert": True,  # Convert GBP/CAD to CAD/GBP
            "description": "Canadian Dollar to British Pound",
        },
    ]

    with get_session() as session:
        # Process each currency pair
        total_stats = {"fetched": 0, "stored": 0, "skipped": 0}

        for pair_config in currency_pairs:
            from_curr = pair_config["from_currency"]
            to_curr = pair_config["to_currency"]
            pair_key = f"{from_curr}/{to_curr}"

            print(f"\n🔄 Processing {pair_config['description']} ({pair_key})")

            # Get statistics for this pair
            stats = backfill_currency_pair(
                session=session,
                from_currency=from_curr,
                to_currency=to_curr,
                yahoo_symbol=pair_config["yahoo_symbol"],
                start_date=start_date_str,
                end_date=end_date_str,
                invert=pair_config["invert"],
            )

            # Add to totals
            for key in total_stats:
                total_stats[key] += stats[key]

            # Commit after each currency pair
            session.commit()
            print(f"  💾 Committed {stats['stored']} new rates to database")

        # Final summary
        print("\n" + "=" * 80)
        print("BACKFILL SUMMARY")
        print("=" * 80)
        print(f"📈 Total rates fetched: {total_stats['fetched']}")
        print(f"💾 Total rates stored: {total_stats['stored']}")
        print(f"⏭️  Total rates skipped (already exist): {total_stats['skipped']}")

        if total_stats["stored"] > 0:
            print(f"\n✅ Successfully backfilled {total_stats['stored']} currency rates!")
            print("🎯 PortfolioDaily backfill can now proceed with accurate currency data.")
        else:
            print("\n⚠️  No new rates were stored. All data may already exist.")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
