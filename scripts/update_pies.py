from datetime import datetime
from time import sleep

from config import TRADING212_API_KEY
from models import Pie, PieInstrument
from scripts.update_data import get_session, request_json


def update_pies():
    """Fetch and store pie data from Trading 212 API."""

    # Database setup

    with get_session() as session:
        try:
            # Fetch all pies from API
            url = "https://live.trading212.com/api/v0/equity/pies"
            pies = request_json(url, {"Authorization": TRADING212_API_KEY})

            print(f"Found {len(pies)} pies to update")

            for pie_data in pies:
                pie_id = pie_data["id"]
                print(f"Processing pie {pie_id}...")

                # Fetch detailed pie data
                sleep(10)  # Rate limiting
                detail_url = f"https://live.trading212.com/api/v0/equity/pies/{pie_id}"
                detailed_data = request_json(detail_url, {"Authorization": TRADING212_API_KEY})

                # Store/update pie data
                pie = store_pie_data(session, pie_data, detailed_data)
                print(f"✓ Updated pie {pie_id}: {pie.name}")

            session.commit()
            print("✓ All pies updated successfully!")

        except Exception as e:
            session.rollback()
            print(f"❌ Error updating pies: {e}")
            raise


def store_pie_data(session, pie_data: dict, detailed_data: dict) -> Pie:
    """Store or update pie data in the database."""

    pie_id = pie_data["id"]

    # Check if pie already exists
    existing_pie = session.query(Pie).filter(Pie.id == pie_id).first()

    if existing_pie:
        pie = existing_pie
        print(f"  Updating existing pie {pie_id}")
    else:
        pie = Pie(id=pie_id)
        session.add(pie)
        print(f"  Creating new pie {pie_id}")

    # Update pie basic data (from first API call)
    pie.cash = pie_data.get("cash", 0.0)
    pie.progress = pie_data.get("progress")
    pie.status = pie_data.get("status")
    pie.dividend_details = pie_data.get("dividendDetails")
    pie.result = pie_data.get("result")

    # Update pie detailed data (from second API call)
    if "settings" in detailed_data:
        settings = detailed_data["settings"]
        pie.name = settings.get("name")
        pie.dividend_cash_action = settings.get("dividendCashAction")
        pie.goal = settings.get("goal")

        # Parse creation_date (timestamp)
        if "creationDate" in settings:
            try:
                pie.creation_date = datetime.fromtimestamp(settings["creationDate"])
            except (ValueError, TypeError):
                pie.creation_date = None

        # Parse end_date (ISO string)
        if "endDate" in settings and settings["endDate"]:
            try:
                pie.end_date = datetime.fromisoformat(settings["endDate"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pie.end_date = None

        # Store raw settings for debugging
        pie.settings = settings

    # Update pie instruments
    if "instruments" in detailed_data:
        store_pie_instruments(session, pie, detailed_data["instruments"])

    return pie


def store_pie_instruments(session, pie: Pie, instruments_data: list):
    """Store pie instruments data."""

    # Clear existing instruments for this pie (to handle removals)
    session.query(PieInstrument).filter(PieInstrument.pie_id == pie.id).delete()

    for instrument_data in instruments_data:
        instrument = PieInstrument(
            pie_id=pie.id,
            t212_code=instrument_data["ticker"],
            expected_share=instrument_data.get("expectedShare", 0.0),
            current_share=instrument_data.get("currentShare", 0.0),
            owned_quantity=instrument_data.get("ownedQuantity", 0.0),
            result=instrument_data.get("result"),
            issues=instrument_data.get("issues", []),
        )
        session.add(instrument)

    print(f"  Updated {len(instruments_data)} instruments")


if __name__ == "__main__":
    update_pies()
