"""
Database services for Trading212 Portfolio Manager
=================================================
Provides high-level database operations using SQLAlchemy models.
"""

import logging
import time
from datetime import datetime, date, timedelta, timezone
from typing import Iterable, Sequence, List, Dict, Optional, Tuple, Set
from contextlib import contextmanager

import pandas as pd
import yfinance as yf
from sqlalchemy import func

from models import (
    DailyPrice, Instrument, Holding, CurrencyRate, PortfolioSnapshot,
    get_db_manager
)

logger = logging.getLogger(__name__)


class DatabaseService:
    """High-level database operations for the portfolio manager."""

    def __init__(self):
        self.db_manager = get_db_manager()
        self.batch_size_yf = 25
        self.request_sleep = 1.0

    @contextmanager
    def get_session(self):
        """Context manager for database sessions."""
        session = self.db_manager.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # Daily Price Operations
    def get_price_history(
        self,
        tickers: Iterable[str],
        start: datetime,
        end: datetime,
        price_field: str
    ) -> pd.DataFrame:
        """
        Get price history for given tickers and date range.
        Downloads missing data from Yahoo Finance if needed.
        """
        tickers = list({t.upper() for t in tickers})
        if not tickers:
            return pd.DataFrame()

        # Get cached data
        have = self._query_cached_prices(tickers, start, end, price_field)
        need, start_needed = self._discover_price_gaps(tickers, have, start, end)

        # Download missing data
        if need:
            df_yf = self._download_prices(need, start_needed, end.date())
            if not df_yf.empty:
                self._bulk_upsert_prices(df_yf)
                have = have.combine_first(df_yf[price_field].loc[start:end])

        return have

    def _query_cached_prices(
        self,
        tickers: Sequence[str],
        start: datetime,
        end: datetime,
        price_field: str
    ) -> pd.DataFrame:
        """Query cached price data from database."""
        with self.get_session() as session:
            query = session.query(
                DailyPrice.symbol,
                DailyPrice.date,
                getattr(DailyPrice, price_field.lower().replace(" ", "_") + "_price").label('px')
            ).filter(
                DailyPrice.symbol.in_(tickers),
                DailyPrice.date.between(start.date(), end.date())
            )

            rows = query.all()

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame([
                {
                    'symbol': row.symbol,
                    'date': row.date,
                    'px': getattr(row, 'px')
                }
                for row in rows
            ])

            if df.empty:
                return pd.DataFrame()

            return df.pivot_table(index='date', columns='symbol', values='px')

    def _discover_price_gaps(
        self,
        tickers: Sequence[str],
        have: pd.DataFrame,
        start: datetime,
        end: datetime
    ) -> Tuple[List[str], date]:
        """Discover which tickers need price data downloads."""
        all_days = pd.date_range(start, end, freq='B')
        need = []
        needed_date = all_days.max().date()

        for sym in tickers:
            if sym not in have.columns:
                need.append(sym)
                needed_date = start.date()
                continue

            if have[sym].empty:
                need.append(sym)
                needed_date = start.date()
            elif have[sym].index.max() < all_days.max().date():
                need.append(sym)
                needed_date = min(needed_date, have[sym].index.max() + timedelta(days=1))

        return need, needed_date

    def _download_prices(
        self,
        tickers: Sequence[str],
        start: date,
        end: date
    ) -> pd.DataFrame:
        """Download price data from Yahoo Finance."""
        frames = []

        if start < end:
            for i in range(0, len(tickers), self.batch_size_yf):
                sub = tickers[i:i + self.batch_size_yf]
                logger.info("Downloading prices for %s", sub)

                try:
                    df = yf.download(
                        tickers=sub,
                        start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"),
                        interval="1d",
                        group_by="column",
                        auto_adjust=False,
                        progress=False,
                        threads=False,
                    )
                    if not df.empty:
                        frames.append(df)
                except Exception as e:
                    logger.error("Failed to download prices for %s: %s", sub, e)

                time.sleep(self.request_sleep)

        return pd.concat(frames, axis=1) if frames else pd.DataFrame()

    def _bulk_upsert_prices(self, df: pd.DataFrame) -> None:
        """Bulk upsert price data to database."""
        if df.empty:
            return

        # Prepare data for database
        records = []
        df.columns.names = ["Field", "Ticker"]
        df.index.name = "Date"

        wide = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        df_long = wide.stack(level="Ticker", future_stack=True).reset_index()
        df_long = df_long.rename(columns={"Adj Close": "Adj_Close"})
        for _, row in df_long.iterrows():
            if pd.isna(row["Close"]):
                continue

            records.append({
                'symbol': row['Ticker'],
                'date': row['Date'].date(),
                'open_price': float(row['Open']),
                'high_price': float(row['High']),
                'low_price': float(row['Low']),
                'close_price': float(row['Close']),
                'adj_close_price': float(row['Adj_Close']),
                'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0,
            })

        # Bulk upsert
        with self.get_session() as session:
            for record in records:
                existing = session.query(DailyPrice).filter(
                    DailyPrice.symbol == record['symbol'],
                    DailyPrice.date == record['date']
                ).first()

                if existing:
                    # Update existing record
                    for key, value in record.items():
                        if key not in ['symbol', 'date']:
                            setattr(existing, key, value)
                else:
                    # Insert new record
                    session.add(DailyPrice(**record))

    # Instrument Operations
    def get_instruments_by_codes(self, t212_codes: Set[str]) -> Dict[str, dict]:
        """Get instruments by their Trading212 codes."""
        with self.get_session() as session:
            instruments = session.query(Instrument).filter(
                Instrument.t212_code.in_(t212_codes),
                Instrument.created_at > datetime.now(timezone.utc) - timedelta(days=30),
            ).all()

            # Return dictionary data instead of ORM objects to avoid session issues
            return {
                instrument.t212_code: {
                    't212_code': instrument.t212_code,
                    'name': instrument.name,
                    'currency': instrument.currency,
                    'sector': instrument.sector,
                    'country': instrument.country,
                    'yahoo_symbol': instrument.yahoo_symbol,
                    'yahoo_data': instrument.yahoo_data,
                    'updated_at': instrument.updated_at
                }
                for instrument in instruments
            }

    def save_instruments(self, instruments_data: List[Dict]) -> None:
        """Save instruments in bulk to database."""
        with self.get_session() as session:
            for instrument_data in instruments_data:
                existing = session.query(Instrument).filter(
                    Instrument.t212_code == instrument_data['t212_code']
                ).first()

                if not existing:
                    # Create new instrument only
                    instrument = Instrument(
                        t212_code=instrument_data['t212_code'],
                        name=instrument_data['name'],
                        currency=instrument_data['currency'],
                        yahoo_symbol=instrument_data.get('yahoo_symbol')
                    )
                    session.add(instrument)

    def get_or_create_instrument(
        self,
        t212_code: str,
        name: str,
        currency: str,
        yahoo_symbol: Optional[str] = None
    ) -> Instrument:
        """Get existing instrument or create new one."""
        with self.get_session() as session:
            instrument = session.query(Instrument).filter(
                Instrument.t212_code == t212_code
            ).first()

            if not instrument:
                # Create new instrument
                instrument = Instrument(
                    t212_code=t212_code,
                    name=name,
                    currency=currency,
                    yahoo_symbol=yahoo_symbol
                )
                session.add(instrument)
                session.flush()  # Get the ID

            return instrument

    def update_instrument_yahoo_data(
        self,
        t212_code: str,
        yahoo_data: dict
    ) -> None:
        """Update instrument with Yahoo Finance data."""
        with self.get_session() as session:
            instrument = session.query(Instrument).filter(
                Instrument.t212_code == t212_code
            ).first()

            if instrument:
                instrument.yahoo_data = yahoo_data
                # updated_at will be automatically set by SQLAlchemy

    def get_instruments_with_fresh_yahoo_data(self, t212_codes: Set[str], max_age_days: int = 1) -> Dict[str, dict]:
        """Get instruments that have fresh Yahoo Finance data (less than max_age_seconds old)."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=max_age_days)

        with self.get_session() as session:
            instruments = session.query(Instrument).filter(
                Instrument.t212_code.in_(t212_codes),
                Instrument.updated_at >= cutoff_time,
                Instrument.yahoo_data.isnot(None)  # Only instruments with Yahoo data
            ).all()

            return {
                instrument.t212_code: {
                    't212_code': instrument.t212_code,
                    'name': instrument.name,
                    'currency': instrument.currency,
                    'sector': instrument.sector,
                    'country': instrument.country,
                    'yahoo_symbol': instrument.yahoo_symbol,
                    'yahoo_data': instrument.yahoo_data,
                    'updated_at': instrument.updated_at
                }
                for instrument in instruments
            }

    def get_yahoo_profile_from_cache(self, yahoo_symbol: str, max_age_seconds: int = 86400) -> Optional[dict]:
        """Get Yahoo Finance profile from database cache if it's fresh enough."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)

        with self.get_session() as session:
            instrument = session.query(Instrument).filter(
                Instrument.yahoo_symbol == yahoo_symbol,
                Instrument.updated_at >= cutoff_time,
                Instrument.yahoo_data.isnot(None)
            ).first()

            return instrument.yahoo_data if instrument else None

    def get_instruments_by_yahoo_symbols(self, yahoo_symbols: List[str]) -> List[str]:
        """Get T212 codes for given Yahoo symbols."""
        with self.get_session() as session:
            instruments = session.query(Instrument).filter(
                Instrument.yahoo_symbol.in_(yahoo_symbols)
            ).all()

            return [instrument.t212_code for instrument in instruments]

    # Holding Operations
    def save_holdings(self, holdings_data: List[Dict]) -> None:
        """Save portfolio holdings to database."""
        with self.get_session() as session:
            # Get current date (without time) for snapshot
            current_date = datetime.now(timezone.utc).date()

            for holding_data in holdings_data:
                # Get or create instrument within the same session
                instrument = session.query(Instrument).filter(
                    Instrument.t212_code == holding_data['t212_code']
                ).first()

                if not instrument:
                    # Create new instrument
                    instrument = Instrument(
                        t212_code=holding_data['t212_code'],
                        name=holding_data['name'],
                        currency=holding_data['currency'],
                        yahoo_symbol=holding_data.get('yahoo_symbol')
                    )
                    session.add(instrument)
                    session.flush()  # Get the ID

                # Check if holding already exists for this instrument and date
                existing_holding = session.query(Holding).filter(
                    Holding.instrument_id == instrument.id,
                    Holding.date == current_date
                ).first()

                if existing_holding:
                    # Update existing holding
                    existing_holding.quantity = holding_data['quantity']
                    existing_holding.avg_price = holding_data['avg_price']
                    existing_holding.current_price = holding_data['current_price']
                    existing_holding.ppl = holding_data['ppl']
                    existing_holding.fx_ppl = holding_data.get('fx_ppl', 0.0)
                    existing_holding.market_cap = holding_data.get('market_cap')
                    existing_holding.pe_ratio = holding_data.get('pe_ratio')
                    existing_holding.beta = holding_data.get('beta')
                    existing_holding.institutional = holding_data.get('institutional')
                else:
                    # Create new holding record
                    holding = Holding(
                        instrument_id=instrument.id,
                        quantity=holding_data['quantity'],
                        avg_price=holding_data['avg_price'],
                        current_price=holding_data['current_price'],
                        ppl=holding_data['ppl'],
                        fx_ppl=holding_data.get('fx_ppl', 0.0),
                        market_cap=holding_data.get('market_cap'),
                        pe_ratio=holding_data.get('pe_ratio'),
                        institutional=holding_data.get('institutional'),
                        beta=holding_data.get('beta'),
                        date=current_date
                    )
                    session.add(holding)

    def get_latest_holdings(self) -> List[Holding]:
        """Get the most recent holdings snapshot."""
        with self.get_session() as session:
            # Get the latest snapshot date
            latest_date = session.query(func.max(Holding.date)).scalar()

            if not latest_date:
                return []

            # Join with instrument to get all the data we need
            return session.query(Holding).join(Instrument).filter(
                Holding.date == latest_date
            ).all()

    # Currency Rate Operations
    def get_currency_rate(
        self,
        from_currency: str,
        to_currency: str,
        rate_date: date
    ) -> Optional[float]:
        """Get cached currency rate."""
        with self.get_session() as session:
            rate = session.query(CurrencyRate).filter(
                CurrencyRate.from_currency == from_currency,
                CurrencyRate.to_currency == to_currency,
                CurrencyRate.date == rate_date
            ).first()

            return rate.rate if rate else None

    def save_currency_rate(
        self,
        from_currency: str,
        to_currency: str,
        rate: float,
        rate_date: date
    ) -> None:
        """Save currency rate to cache."""
        with self.get_session() as session:
            existing = session.query(CurrencyRate).filter(
                CurrencyRate.from_currency == from_currency,
                CurrencyRate.to_currency == to_currency,
                CurrencyRate.date == rate_date
            ).first()

            if existing:
                existing.rate = rate
            else:
                currency_rate = CurrencyRate(
                    from_currency=from_currency,
                    to_currency=to_currency,
                    rate=rate,
                    date=rate_date
                )
                session.add(currency_rate)

    # Portfolio Snapshot Operations
    def save_portfolio_snapshot(self, snapshot_data: Dict) -> None:
        """Save portfolio snapshot."""
        with self.get_session() as session:
            # Check if snapshot already exists for this date
            existing_snapshot = session.query(PortfolioSnapshot).filter(
                PortfolioSnapshot.date == snapshot_data['snapshot_date']
            ).first()

            if existing_snapshot:
                # Update existing snapshot
                existing_snapshot.total_value_gbp = snapshot_data['total_value']
                existing_snapshot.total_profit_gbp = snapshot_data['total_profit']
                existing_snapshot.total_return_pct = snapshot_data['total_return_pct']
                existing_snapshot.country_allocation = snapshot_data.get('country_allocation')
                existing_snapshot.sector_allocation = snapshot_data.get('sector_allocation')
                existing_snapshot.etf_equity_split = snapshot_data.get('etf_equity_split')
            else:
                # Create new snapshot
                snapshot = PortfolioSnapshot(
                    date=snapshot_data['snapshot_date'],
                    total_value_gbp=snapshot_data['total_value'],
                    total_profit_gbp=snapshot_data['total_profit'],
                    total_return_pct=snapshot_data['total_return_pct'],
                    country_allocation=snapshot_data.get('country_allocation'),
                    sector_allocation=snapshot_data.get('sector_allocation'),
                    etf_equity_split=snapshot_data.get('etf_equity_split')
                )
                session.add(snapshot)

    def get_portfolio_history(
        self,
        days: int = 30
    ) -> List[PortfolioSnapshot]:
        """Get portfolio snapshots for the last N days."""
        with self.get_session() as session:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            return session.query(PortfolioSnapshot).filter(
                PortfolioSnapshot.date >= cutoff_date
            ).order_by(PortfolioSnapshot.date.desc()).all()


# Global database service instance
_db_service: Optional[DatabaseService] = None


def get_db_service() -> DatabaseService:
    """Get or create the global database service instance."""
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
    return _db_service
