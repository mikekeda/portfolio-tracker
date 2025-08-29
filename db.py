import os
import redis

import logging, sqlite3, time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from forex_python.converter import CurrencyRates
import pandas as pd
import yfinance as yf

TTL = 60 * 60 * 12          # 12 h

rd = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/1"))


def currency_table() -> dict[str, float]:
    """GBP conversion rates. Fallback to hard‑coded if API offline."""
    c = CurrencyRates()
    table = {"GBX": 0.01, "GBP": 1.0, "USD": 0.74, "EUR": 0.87, "CAD": 0.54}

    for cur in ("USD", "EUR", "CAD"):
        if not (rate := rd.get(cur)):
            try:
                rate = c.get_rate(cur, "GBP")
                rd.setex(cur, TTL, rate)
            except Exception as exc:
                rate = table[cur]
                logging.error("Currency API failed → using fallback :: %s", exc)
                break


        table[cur] = float(rate)

    return table


class DailyPriceManager:
    """Stock daily price job manager."""

    # ── configuration you might tune ──────────────────────────────────────
    DB_PATH        = Path("portfolio.db")          # one DB for everything
    BATCH_SIZE_YF  = 25
    REQUEST_SLEEP  = 1.0          # polite gap between Yahoo calls (s)
    PRICE_FIELDS   = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
    _EPOCH         = date(1970, 1, 1)

    # ── connection lifecycle ─────────────────────────────────────────────
    def __init__(self):
        self.conn = sqlite3.connect(self.DB_PATH, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self._init_pragmas()
        self._create_table()

    # ── public API ───────────────────────────────────────────────────────
    def history(
        self,
        tickers: Iterable[str],
        start: datetime,
        end: datetime,
        price_field: str = "Adj Close",
    ) -> pd.DataFrame:
        """
        Return a DataFrame whose *index* is business-day DatetimeIndex and whose
        *columns* are the tickers requested—exactly the shape your old
        `yahoo_history` delivered.

        *   Reads any rows already cached in `daily_price`.
        *   Downloads *only* the missing (ticker, day) gaps from Yahoo Finance.
        *   Upserts those rows (`INSERT OR IGNORE`) and stitches everything
            together before returning.
        """
        tickers = list({t.upper() for t in tickers})
        if not tickers:
            return pd.DataFrame()

        start_int, end_int = self._to_int_day(start), self._to_int_day(end)

        have = self._query_cache(tickers, start_int, end_int, price_field)
        need, start_needed = self._discover_gaps(tickers, have, start, end)

        if need:
            df_yf = self._download_prices(need, start_needed, end.date())
            if not df_yf.empty:
                self._bulk_upsert(df_yf)
                have = have.combine_first(df_yf[price_field].loc[start:end])

        full_index = pd.date_range(start, end, freq="B")
        # return have.reindex(full_index, columns=tickers)
        return have

    # ── connection helper ────────────────────────────────────────────────
    def _init_pragmas(self):
        self.cursor.executescript(
            """
            PRAGMA foreign_keys = ON;
            PRAGMA strict       = ON;
            PRAGMA journal_mode = WAL;
            """
        )

    # ── DDL ----------------------------------------------------------------
    def _create_table(self):
        """Create `daily_price` if it does not exist."""
        self.cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS daily_price (
                symbol     TEXT    NOT NULL,
                date       TEXT    NOT NULL,            -- store ISO yyyy-mm-dd
                day        INTEGER NOT NULL,           -- days since 1970-01-01
                open       REAL    NOT NULL,
                high       REAL    NOT NULL,
                low        REAL    NOT NULL,
                close      REAL    NOT NULL,
                adj_close  REAL    NOT NULL,
                volume     INTEGER NOT NULL,
                PRIMARY KEY (symbol, day)
            ) WITHOUT ROWID;

            -- You probably only need one of these.  Un-comment when real
            -- query plans show a full scan.
            --CREATE INDEX IF NOT EXISTS dp_idx_day    ON daily_price (day);
            --CREATE INDEX IF NOT EXISTS dp_idx_symbol ON daily_price (symbol);
            """
        )

    # ── core helpers -------------------------------------------------------
    @classmethod
    def _to_int_day(cls, dt: datetime | date) -> int:
        d = dt.date() if isinstance(dt, datetime) else dt
        return (d - cls._EPOCH).days

    @classmethod
    def _to_datetime(cls, int_day: int) -> datetime:
        return datetime.utcfromtimestamp(int_day * 86_400)

    # ---- cache read -------------------------------------------------------
    def _query_cache(
        self,
        tickers: Sequence[str],
        start_int: int,
        end_int: int,
        price_field: str,
    ) -> pd.DataFrame:
        ph = ",".join("?" * len(tickers))
        rows = self.cursor.execute(
            f"""
            SELECT symbol,
                   day,
                   {price_field.lower().replace(' ', '_')} AS px
            FROM   daily_price
            WHERE  symbol IN ({ph})
              AND  day BETWEEN ? AND ?
            """,
            (*tickers, start_int, end_int),
        ).fetchall()

        if not rows:
            return pd.DataFrame()

        df = (
            pd.DataFrame(rows, columns=["symbol", "day", "px"])
            .assign(date=lambda df: pd.to_datetime(df["day"], unit="D"))
            .pivot_table(index="date", columns="symbol", values="px")
        )
        return df

    # ---- determine what’s missing ---------------------------------------
    def _discover_gaps(
        self,
        tickers: Sequence[str],
        have: pd.DataFrame,
        start: datetime,
        end: datetime,
    ) -> Tuple[list[str], date]:
        """Return tickers that lack complete coverage in [start, end]."""
        all_days = pd.date_range(start, end, freq="B")
        need = []
        needed_date = all_days.max().date()
        for sym in tickers:
            if sym not in have:
                need.append(sym)
                needed_date = start.date()
                continue
            # missing if any NaN or no column at all
            if have[sym].empty:
                need.append(sym)
                needed_date = start.date()
            elif have[sym].index.max().date() < all_days.max().date():
                need.append(sym)
                needed_date = min(needed_date, have[sym].index.max().date() + timedelta(days=1))

        return need, needed_date

    # ---- download & upsert ----------------------------------------------
    def _download_prices(
        self, tickers: Sequence[str], start: date, end: date
    ) -> pd.DataFrame:
        frames = []
        for i in range(0, len(tickers), self.BATCH_SIZE_YF):
            sub = tickers[i : i + self.BATCH_SIZE_YF]
            logging.info("Yahoo download %s …", sub)
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
                logging.error(e)
            time.sleep(self.REQUEST_SLEEP)

        return pd.concat(frames, axis=1) if frames else pd.DataFrame()

    def _bulk_upsert(self, df: pd.DataFrame) -> None:
        # Ensure column index names for predictable stack/reset_index names
        df.columns.names = ["Field", "Ticker"]
        df.index.name = "Date"
        wide = df[["Open","High","Low","Close","Adj Close","Volume"]]
        df_long = wide.stack(level="Ticker").reset_index()  # columns: Date, Ticker, Field columns...
        df_long = df_long.rename(columns={"Adj Close": "Adj_Close"})
        df_long["day"] = pd.to_datetime(df_long["Date"]).dt.date.map(self._to_int_day)

        records = [
            (
                row.Ticker,
                row.Date.date().isoformat(),
                row.day,
                float(row.Open),
                float(row.High),
                float(row.Low),
                float(row.Close),
                float(row.Adj_Close),
                int(row.Volume) if not pd.isna(row.Volume) else 0,
            )
            for row in df_long.itertuples(index=False)
            if not pd.isna(row.Close)
        ]
        with self.conn:
            self.cursor.executemany(
                """INSERT OR IGNORE INTO daily_price
                   (symbol, date, day, open, high, low, close, adj_close, volume)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                records,
            )
