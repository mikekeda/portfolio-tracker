"""
Import Trading212 transaction history from CSV exports.

Trading212 has a 1-year limit for CSV exports, so this script processes multiple
CSV files from the 'csv' directory and imports them into the TransactionHistory table.

Features:
- Processes all CSV files in alphabetical (chronological) order
- Handles orders, dividends, interest, and deposits
- Skips stock splits (they're quantity adjustments, not transactions)
- Handles duplicate detection (safe to re-run)
- Shows comprehensive analysis before importing
- Extracts all fees (currency conversion, stamp duty, withholding tax, etc.)

Usage:
    1. Export CSV files from Trading212 (Activity → Export)
    2. Place them in the 'csv' directory
    3. Run: python update_history_from_csv.py
"""

import csv
import os
from collections import Counter
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from sqlalchemy import select
from update_data import get_session
from models import TransactionHistory, Instrument


def safe_float(value: str) -> float:
    """Parse string to float, return 0.0 for invalid values."""
    try:
        return float(value) if value and value.strip() else 0.0
    except (ValueError, AttributeError):
        return 0.0


def parse_csv_row(row: Dict[str, str], row_num: int, instruments_lookup: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Parse a single CSV row into transaction data format."""
    # Skip empty rows
    if not row.get("Action") or not row.get("Time"):
        return None

    # Get original action name from CSV
    action = row["Action"].strip()

    # Skip non-transaction actions (stock splits don't involve cash flow)
    if "STOCK_SPLIT" in action.upper():
        return None

    # Parse quantity based on action type
    if "BUY" in action.upper():
        filled_quantity = float(row["No. of shares"])
    elif "SELL" in action.upper():
        filled_quantity = -float(row["No. of shares"])
    elif action.startswith("Dividend"):
        filled_quantity = float(row["No. of shares"])  # Quantity of shares that earned dividend
    elif action in ["Interest on cash", "Deposit", "Withdrawal"]:
        filled_quantity = 0.0  # No shares for cash movements
    else:
        print(f"  ⚠️  Unknown action type in row {row_num}: {action}")
        return None

    # Parse timestamp
    try:
        time_str = row["Time"].strip()
        datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")  # Validate format
    except ValueError as e:
        print(f"  ⚠️  Error parsing date in row {row_num}: {e}")
        return None

    # Parse pricing and fees
    fill_price = safe_float(row["Price / share"]) if row.get("Price / share") else None
    filled_value = safe_float(row["Total"])
    exchange_rate = safe_float(row.get("Exchange rate", ""))
    fill_result = safe_float(row.get("Result", ""))

    # Parse fees from CSV columns
    fees = []
    fee_configs = [
        ("Currency conversion fee", "CURRENCY_CONVERSION_FEE"),
        ("Stamp duty reserve tax", "STAMP_DUTY_RESERVE_TAX"),
        ("French transaction tax", "FRENCH_TRANSACTION_TAX"),
        ("Withholding tax", "WITHHOLDING_TAX"),  # Added for dividends
    ]

    for csv_column, fee_name in fee_configs:
        fee_amount = safe_float(row.get(csv_column, ""))
        if fee_amount:
            fees.append({"name": fee_name, "quantity": -fee_amount, "timeCharged": time_str})

    # Get currency and name from CSV, then normalize ticker using name-based matching
    currency = row.get("Currency (Price / share)", "USD").strip()
    csv_ticker = row.get("Ticker", "")  # May be empty for deposits/interest
    csv_name = row.get("Name", "").strip()

    # Normalize ticker using name-based matching (only for transactions with tickers)
    normalized_ticker: Optional[str] = None
    was_normalized = False

    if csv_ticker and action not in ["Deposit", "Interest on cash", "Withdrawal"]:
        clean_csv_name = csv_name.strip().strip("\"'")

        # Try exact match first
        if clean_csv_name in instruments_lookup:
            normalized_ticker = instruments_lookup[clean_csv_name]
            was_normalized = True
            print(f"  ✓ Normalized: {csv_ticker} → {normalized_ticker} (matched: '{clean_csv_name}')")
        else:
            # Try case-insensitive or partial match
            for db_name, yahoo_symbol in instruments_lookup.items():
                if db_name.lower() == clean_csv_name.lower() or (
                    len(clean_csv_name) > 3
                    and (clean_csv_name.lower() in db_name.lower() or db_name.lower() in clean_csv_name.lower())
                ):
                    normalized_ticker = yahoo_symbol
                    was_normalized = True
                    print(f"  ✓ Normalized: {csv_ticker} → {yahoo_symbol} (matched: '{clean_csv_name}' ≈ '{db_name}')")
                    break
            else:
                print(f"  ⚠️  Missing: {csv_ticker} ('{clean_csv_name}')")
                normalized_ticker = csv_ticker
                was_normalized = False
    else:
        normalized_ticker = csv_ticker if csv_ticker else None

    return {
        "csv_id": row["ID"] if row["ID"].strip() else None,  # Handle empty IDs
        "action": action,
        "ticker": normalized_ticker,
        "isin": row["ISIN"],
        "originalTicker": csv_ticker,  # Keep original for logging
        "originalName": csv_name,  # Keep original name for logging
        "was_normalized": was_normalized,  # Track if normalization was successful
        "notes": row.get("Notes", "").strip() or None,  # Add Notes field
        "currency": currency,
        "quantity": filled_quantity,
        "price": fill_price,
        "total": filled_value,
        "exchangeRate": exchange_rate if exchange_rate else None,
        "result": fill_result if "SELL" in action.upper() and fill_result else None,
        "fees": fees if fees else None,
        "dateCreated": time_str,
    }


def parse_csv_file(csv_file_path: str, instruments_lookup: Dict[str, str]) -> List[Dict[str, Any]]:
    """Parse Trading212 CSV export and return list of transaction dictionaries."""
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

    orders = []

    with open(csv_file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row_num, row in enumerate(reader, start=2):  # Start at 2 (1=header)
            try:
                order_data = parse_csv_row(row, row_num, instruments_lookup)
                if order_data:
                    orders.append(order_data)
            except Exception as e:
                print(f"  ⚠️  Error parsing row {row_num}: {e}")
                continue

    return orders


def store_transaction(session, transaction_data: Dict[str, Any]) -> bool:
    """Store a single transaction in the database."""
    csv_id = transaction_data["csv_id"]

    # Check if transaction already exists (duplicate detection by CSV ID)
    if csv_id:
        existing = session.execute(
            select(TransactionHistory).filter(TransactionHistory.csv_id == csv_id)
        ).scalar_one_or_none()
        if existing:
            return False  # Skip duplicate

    # Create and store new transaction
    transaction = TransactionHistory(
        csv_id=csv_id,
        timestamp=datetime.strptime(transaction_data["dateCreated"], "%Y-%m-%d %H:%M:%S"),
        ticker=transaction_data["ticker"],
        isin=transaction_data["isin"],
        action=transaction_data["action"],
        quantity=transaction_data["quantity"],
        price=transaction_data["price"],
        total=transaction_data["total"],
        notes=transaction_data.get("notes"),
        exchange_rate=transaction_data.get("exchangeRate"),
        result=transaction_data.get("result"),
        fees=transaction_data.get("fees"),
    )
    session.add(transaction)
    return True


def analyze_and_import_csv_files(
    csv_files: List[str], instruments_lookup: Dict[str, str]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Analyze all CSV files and return transactions and statistics."""
    all_transactions = []
    missing_instruments = {}
    stats: Dict[str, int] = Counter()

    for csv_file in csv_files:
        print(f"\n📄 File: {os.path.basename(csv_file)}")
        print("-" * 80)

        transactions = parse_csv_file(csv_file, instruments_lookup)
        all_transactions.extend(transactions)

        if transactions:
            # Calculate statistics for this file
            buys = [o for o in transactions if "BUY" in o["action"].upper()]
            sells = [o for o in transactions if "SELL" in o["action"].upper()]
            dividends = [o for o in transactions if o["action"].startswith("Dividend")]
            interest = [o for o in transactions if o["action"] == "Interest on cash"]
            deposits = [o for o in transactions if o["action"] == "Deposit"]

            file_stats = {
                "transactions": len(transactions),
                "buys": len(buys),
                "sells": len(sells),
                "dividends": len(dividends),
                "interest": len(interest),
                "deposits": len(deposits),
                "invested": sum(o["total"] for o in buys),
                "proceeds": sum(o["total"] for o in sells),
                "dividend_income": sum(o["total"] for o in dividends),
                "interest_income": sum(o["total"] for o in interest),
                "deposit_amount": sum(o["total"] for o in deposits),
                "fees": sum(abs(fee["quantity"]) for o in transactions for fee in (o.get("fees") or [])),
            }

            # Update combined stats
            for key, value in file_stats.items():
                stats[key] += value

            print(
                f"  Orders: {file_stats['buys'] + file_stats['sells']} ({file_stats['buys']} buys, {file_stats['sells']} sells)"
            )
            if file_stats["dividends"]:
                print(f"  Dividends: {file_stats['dividends']} (£{file_stats['dividend_income']:,.2f})")
            if file_stats["interest"]:
                print(f"  Interest: {file_stats['interest']} (£{file_stats['interest_income']:,.2f})")
            if file_stats["deposits"]:
                print(f"  Deposits: {file_stats['deposits']} (£{file_stats['deposit_amount']:,.2f})")
            print(f"  Invested: £{file_stats['invested']:,.2f}")
            print(f"  Proceeds: £{file_stats['proceeds']:,.2f}")

            # Collect missing instruments
            for transaction in transactions:
                if (
                    transaction.get("originalTicker")
                    and transaction.get("action") not in ["Deposit", "Interest on cash", "Withdrawal"]
                    and not transaction.get("was_normalized", False)
                ):
                    csv_ticker = transaction["originalTicker"]
                    csv_name = transaction.get("originalName", "")
                    currency = transaction["currency"]
                    key = f"{csv_ticker} ({csv_name})" if csv_name else csv_ticker

                    if key not in missing_instruments:
                        missing_instruments[key] = {
                            "ticker": csv_ticker,
                            "name": csv_name,
                            "currency": currency,
                            "count": 0,
                        }
                    missing_instruments[key]["count"] += 1

    combined_stats = {
        "total_transactions": stats["transactions"],
        "buys": stats["buys"],
        "sells": stats["sells"],
        "dividends": stats["dividends"],
        "interest": stats["interest"],
        "deposits": stats["deposits"],
        "invested": stats["invested"],
        "proceeds": stats["proceeds"],
        "dividend_income": stats["dividend_income"],
        "interest_income": stats["interest_income"],
        "deposit_amount": stats["deposit_amount"],
        "fees": stats["fees"],
    }

    return all_transactions, combined_stats, missing_instruments


def print_summary(stats: Dict[str, Any], missing_instruments: Dict[str, Dict[str, Any]]) -> None:
    """Print combined statistics and missing instruments summary."""
    print("\n" + "=" * 80)
    print("📊 COMBINED STATISTICS (ALL FILES)")
    print("=" * 80)
    print(f"  Total Transactions: {stats['total_transactions']}")
    print(f"  Buy Orders: {stats['buys']}")
    print(f"  Sell Orders: {stats['sells']}")
    print(f"  Dividends: {stats['dividends']} (£{stats['dividend_income']:,.2f})")
    print(f"  Interest: {stats['interest']} (£{stats['interest_income']:,.2f})")
    print(f"  Deposits: {stats['deposits']} (£{stats['deposit_amount']:,.2f})")
    print(f"  Total Invested: £{stats['invested']:,.2f}")
    print(f"  Total Proceeds: £{stats['proceeds']:,.2f}")
    print(f"  Total Fees: £{stats['fees']:,.2f}")
    print(f"  Net Investment: £{stats['invested'] - stats['proceeds']:,.2f}")

    # Missing instruments summary
    if not missing_instruments:
        print("\n✅ All tickers were successfully normalized using name-based matching!")
        return

    print("\n" + "=" * 80)
    print("📋 MISSING INSTRUMENTS SUMMARY")
    print("=" * 80)
    print("The following tickers from CSV could not be matched to instruments by name:")
    print()

    for key, details in sorted(missing_instruments.items()):
        ticker = details["ticker"]
        name = details["name"]
        currency = details["currency"]
        count = details["count"]

        if name:
            print(f'  • {ticker} - "{name}" ({currency}) - {count} transactions')
        else:
            print(f"  • {ticker} ({currency}) - {count} transactions")

    print()
    print("💡 To fix this, you can:")
    print("  1. Check if these instrument names exist in your database")
    print("  2. Verify the name format matches exactly (case-sensitive)")
    print("  3. Add missing instruments to your database")
    print()


def main():
    """Main function to import CSV data from csv directory."""
    print("🔄 Trading212 CSV Import Tool")
    print("=" * 80)

    # Load instruments for ticker normalization
    with get_session() as session:
        try:
            instruments = session.execute(select(Instrument)).scalars().all()
            instruments_lookup = {instrument.name: instrument.yahoo_symbol for instrument in instruments}
            print(f"📊 Loaded {len(instruments_lookup)} instruments for name-based lookup")
        except Exception as e:
            print(f"❌ Failed to load instruments: {e}")
            print("Please ensure the database is accessible and models are up to date.")
            return

    # Get all CSV files from csv directory
    csv_dir = "csv"
    if not os.path.exists(csv_dir):
        print(f"❌ CSV directory not found: {csv_dir}")
        return

    csv_files = sorted([os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith(".csv")])

    if not csv_files:
        print("\n❌ No CSV files found in 'csv' directory")
        print("Please ensure CSV files are in the 'csv' directory.")
        return

    print(f"\n📁 Found {len(csv_files)} CSV file(s):")
    for i, csv_file in enumerate(csv_files, 1):
        print(f"  {i}. {os.path.basename(csv_file)}")

    # Analyze all files first (dry run)
    print("\n" + "=" * 80)
    print("🔍 ANALYZING ALL FILES (DRY RUN)")
    print("=" * 80)

    all_transactions, combined_stats, missing_instruments = analyze_and_import_csv_files(csv_files, instruments_lookup)

    # Show combined statistics and missing instruments
    print_summary(combined_stats, missing_instruments)

    # Ask for confirmation
    print("\n" + "=" * 80)
    response = (
        input(f"\n❓ Import {combined_stats['total_transactions']} transactions from {len(csv_files)} file(s)? (y/N): ")
        .strip()
        .lower()
    )
    if response not in ["y", "yes"]:
        print("❌ Import cancelled")
        return

    # Perform actual import for all files
    print("\n" + "=" * 80)
    print("💾 STARTING IMPORT")
    print("=" * 80)

    total_imported = 0
    total_skipped = 0

    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n📄 [{i}/{len(csv_files)}] Processing: {os.path.basename(csv_file)}")
        print("-" * 80)

        transactions = parse_csv_file(csv_file, instruments_lookup)
        if not transactions:
            print("  ⚠️  No transactions found, skipping")
            continue

        try:
            with get_session() as session:
                imported_count = 0
                skipped_count = 0

                for transaction_data in transactions:
                    try:
                        result = store_transaction(session, transaction_data)
                        if result:
                            imported_count += 1
                        else:
                            skipped_count += 1
                    except Exception as e:
                        print(f"  ⚠️  Error importing transaction {transaction_data.get('csv_id', 'unknown')}: {e}")
                        skipped_count += 1

                session.commit()

                print(f"  ✅ Imported: {imported_count} transactions")
                if skipped_count > 0:
                    print(f"  ⏭️  Skipped: {skipped_count} transactions (already exist)")

                total_imported += imported_count
                total_skipped += skipped_count
        except Exception as e:
            print(f"  ❌ Failed to import transactions from {os.path.basename(csv_file)}: {e}")
            continue

    # Final summary
    print("\n" + "=" * 80)
    print("🎉 IMPORT COMPLETED!")
    print("=" * 80)
    print(f"  Files processed: {len(csv_files)}")
    print(f"  Transactions imported: {total_imported}")
    print(f"  Transactions skipped: {total_skipped}")
    print(f"  Total transactions: {total_imported + total_skipped}")
    print("\n✨ Your transaction history is now complete!")


if __name__ == "__main__":
    main()
