"""
Database services for Trading212 Portfolio Manager
=================================================
Provides high-level database operations using SQLAlchemy models.
"""

import logging
import time
from datetime import datetime, date, timedelta
from typing import Iterable, Sequence, List, Dict, Optional, Tuple
from contextlib import contextmanager

import pandas as pd
import yfinance as yf
from sqlalchemy import func

from models import (
    DailyPrice, Instrument, Holding, CurrencyRate, PortfolioSnapshot,
    get_db_manager, DatabaseManager
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
    
    def _to_int_day(self, dt: datetime | date) -> int:
        """Convert datetime/date to days since epoch."""
        d = dt.date() if isinstance(dt, datetime) else dt
        epoch = date(1970, 1, 1)
        return (d - epoch).days
    
    def _to_datetime(self, int_day: int) -> datetime:
        """Convert days since epoch to datetime."""
        epoch = date(1970, 1, 1)
        return datetime.combine(epoch + timedelta(days=int_day), datetime.min.time())
    
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
        
        start_int, end_int = self._to_int_day(start), self._to_int_day(end)
        
        # Get cached data
        have = self._query_cached_prices(tickers, start_int, end_int, price_field)
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
        start_int: int,
        end_int: int,
        price_field: str
    ) -> pd.DataFrame:
        """Query cached price data from database."""
        with self.get_session() as session:
            query = session.query(
                DailyPrice.symbol,
                DailyPrice.date,
                getattr(DailyPrice, price_field).label('px')
            ).filter(
                DailyPrice.symbol.in_(tickers),
                DailyPrice.day.between(start_int, end_int)
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
        df_long = wide.stack(level="Ticker").reset_index()
        df_long = df_long.rename(columns={"Adj Close": "Adj_Close"})
        df_long["day"] = pd.to_datetime(df_long["Date"]).dt.date.map(self._to_int_day)
        
        for _, row in df_long.iterrows():
            if pd.isna(row["Close"]):
                continue
                
            records.append({
                'symbol': row['Ticker'],
                'date': row['Date'].date(),
                'day': row['day'],
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
                    DailyPrice.day == record['day']
                ).first()
                
                if existing:
                    # Update existing record
                    for key, value in record.items():
                        if key not in ['symbol', 'day']:
                            setattr(existing, key, value)
                else:
                    # Insert new record
                    session.add(DailyPrice(**record))
    
    # Instrument Operations
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
            
            if instrument:
                # Update if needed
                if instrument.name != name or instrument.currency != currency:
                    instrument.name = name
                    instrument.currency = currency
                    instrument.updated_at = datetime.utcnow()
                if yahoo_symbol and instrument.yahoo_symbol != yahoo_symbol:
                    instrument.yahoo_symbol = yahoo_symbol
                return instrument
            else:
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
    
    def update_instrument_metadata(
        self,
        t212_code: str,
        sector: Optional[str] = None,
        country: Optional[str] = None
    ) -> None:
        """Update instrument metadata."""
        with self.get_session() as session:
            instrument = session.query(Instrument).filter(
                Instrument.t212_code == t212_code
            ).first()
            
            if instrument:
                if sector is not None:
                    instrument.sector = sector
                if country is not None:
                    instrument.country = country
                
                instrument.updated_at = datetime.utcnow()
    
    # Holding Operations
    def save_holdings(self, holdings_data: List[Dict]) -> None:
        """Save portfolio holdings to database."""
        with self.get_session() as session:
            # Clear existing holdings for this snapshot
            snapshot_date = datetime.utcnow()
            
            for holding_data in holdings_data:
                # Get or create instrument
                instrument = self.get_or_create_instrument(
                    t212_code=holding_data['t212_code'],
                    name=holding_data['name'],
                    currency=holding_data['currency'],
                    yahoo_symbol=holding_data.get('yahoo_symbol')
                )
                
                # Create holding record
                holding = Holding(
                    instrument_id=instrument.id,
                    quantity=holding_data['quantity'],
                    avg_price=holding_data['avg_price'],
                    current_price=holding_data['current_price'],
                    ppl=holding_data['ppl'],
                    fx_ppl=holding_data.get('fx_ppl', 0.0),
                    market_cap=holding_data.get('market_cap'),
                    pe_ratio=holding_data.get('pe_ratio'),
                    beta=holding_data.get('beta'),
                    snapshot_date=snapshot_date
                )
                session.add(holding)
    
    def get_latest_holdings(self) -> List[Holding]:
        """Get the most recent holdings snapshot."""
        with self.get_session() as session:
            # Get the latest snapshot date
            latest_date = session.query(func.max(Holding.snapshot_date)).scalar()
            
            if not latest_date:
                return []
            
            return session.query(Holding).filter(
                Holding.snapshot_date == latest_date
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
                CurrencyRate.rate_date == rate_date
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
                CurrencyRate.rate_date == rate_date
            ).first()
            
            if existing:
                existing.rate = rate
            else:
                currency_rate = CurrencyRate(
                    from_currency=from_currency,
                    to_currency=to_currency,
                    rate=rate,
                    rate_date=rate_date
                )
                session.add(currency_rate)
    
    # Portfolio Snapshot Operations
    def save_portfolio_snapshot(
        self,
        total_value_gbp: float,
        total_profit_gbp: float,
        total_return_pct: float,
        country_allocation: Optional[Dict] = None,
        sector_allocation: Optional[Dict] = None,
        etf_equity_split: Optional[Dict] = None
    ) -> None:
        """Save portfolio snapshot."""
        with self.get_session() as session:
            snapshot = PortfolioSnapshot(
                snapshot_date=datetime.utcnow(),
                total_value_gbp=total_value_gbp,
                total_profit_gbp=total_profit_gbp,
                total_return_pct=total_return_pct,
                country_allocation=country_allocation,
                sector_allocation=sector_allocation,
                etf_equity_split=etf_equity_split
            )
            session.add(snapshot)
    
    def get_portfolio_history(
        self,
        days: int = 30
    ) -> List[PortfolioSnapshot]:
        """Get portfolio snapshots for the last N days."""
        with self.get_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            return session.query(PortfolioSnapshot).filter(
                PortfolioSnapshot.snapshot_date >= cutoff_date
            ).order_by(PortfolioSnapshot.snapshot_date.desc()).all()


# Global database service instance
_db_service: Optional[DatabaseService] = None


def get_db_service() -> DatabaseService:
    """Get or create the global database service instance."""
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
    return _db_service
