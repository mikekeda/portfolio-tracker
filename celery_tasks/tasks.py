import asyncio
from backend.app import dispose_engine
from celery_tasks.celery_app import app
from scripts.update_returns import update_returns
from scripts.fix_split_prices import fix_split_prices
from scripts.get_earnings_reports import get_earnings_reports
from scripts.get_uk_earnings_reports import get_uk_earnings_reports
from scripts.scrape_13f import main as scrape_13f_main
from scripts.scrape_wisesheets_pe import update_pe_data
from scripts.run_position_review import run_position_reviews
from scripts.run_trade_agent import run_trade_agent
from scripts.update_data import update_data
from scripts.update_features import update_features
from scripts.update_market_metrics import update_market_metrics
from scripts.update_pies import update_pies
from scripts.sync_transactions import sync_transactions
from scripts.update_sec_features import update_sec_features


def _run_async_db_job(coro) -> None:
    """Run an async DB job in a fresh event loop without leaking pooled connections.

    The prefork worker reuses one process (and backend.app's lru_cached engine)
    across tasks, but each asyncio.run() creates a new loop. Abandon any pool
    left by a previous loop before starting, and dispose our own pool before
    the loop closes, so no asyncpg connection is ever touched from a foreign loop.
    """
    async def _job():
        await dispose_engine(close=False)
        try:
            await coro
        finally:
            await dispose_engine()

    asyncio.run(_job())


@app.task
def calculate_portfolio_returns_task():
    update_returns(rebuild=False)


@app.task
def update_data_task():
    update_data()


@app.task
def fix_split_prices_task():
    """Repair split-adjustment discontinuities in PricesDaily and HoldingDaily.

    Pre-split rows keep the old price scale after a split. Detects affected
    instruments via InstrumentYahoo.splits, re-downloads price history and
    rescales holdings snapshots. No-op when consistent — safe to re-run.
    """
    fix_split_prices()


@app.task
def update_market_metrics_task():
    asyncio.run(update_market_metrics())


@app.task(soft_time_limit=5400, time_limit=5700)
def update_pe_data_task():
    """Update PE ratio historical data from Macrotrends for instruments with oldest PE data.

    Overrides the global time limit: ~19 s/ticker over 140 tickers is a ~45 min
    run, and the retry-on-error path can push it further.
    """
    update_pe_data(limit=100)


@app.task
def update_pies_task():
    """Update Trading212 pies data from API."""
    update_pies()


@app.task
def fetch_earnings_reports_task(limit: int = 100):
    """Fetch and process earnings reports from SEC EDGAR API."""
    get_earnings_reports(limit=limit)


@app.task
def fetch_uk_earnings_reports_task(limit: int = 100):
    """Fetch and process UK RNS earnings reports from Investegate.

    Targets ``.L`` instruments without a populated CIK (i.e. not already
    routed via SEC). See scripts/get_uk_earnings_reports.py for the full
    selection model and per-instrument decision flow.
    """
    get_uk_earnings_reports(limit=limit)


@app.task
def scrape_13f_task():
    """Scrape SEC 13F institutional holdings for all configured investors.

    Runs weekly — 13F filings are quarterly (deadlines: May 15, Aug 14, Nov 14, Feb 14).
    The script fetches only missing quarters, so this is a no-op when already up to date.
    """
    scrape_13f_main()


@app.task
def run_position_reviews_task():
    """Generate LLM position reviews for held instruments with a thesis.

    Hash-gated: a review regenerates only when its slow-moving inputs changed
    (thesis, rule/band state, fundamentals, latest earnings, macro regime) —
    quiet days cost zero LLM calls. See scripts/run_position_review.py.
    """
    _run_async_db_job(run_position_reviews())


@app.task
def run_trade_agent_task():
    """Generate today's trade suggestions (suggest-only) into trade_suggestions.

    Runs the rules strategy through the deterministic constraint layer; results
    surface on the /agent page where the user accepts or dismisses them.
    """
    _run_async_db_job(run_trade_agent())


@app.task
def update_features_task():
    """Persist today's point-in-time feature snapshot for every monitored instrument.

    Captures the derived features (screener score, ROIC, DCF, thesis-rule state)
    that otherwise only exist in the overwritten InstrumentYahoo cache, building
    the historical series the trade-suggestion agent trains and backtests on.
    """
    _run_async_db_job(update_features())


@app.task
def update_sec_features_task():
    """Refresh SEC companyfacts and month-end sec_features_daily for CIK names.

    Filing-driven fetch; statement Tier B only. Does not write screener_score.
    """
    update_sec_features(only_holdings=False)


@app.task
def sync_transactions_task():
    """Sync transaction and dividend history from Trading212 API.

    Fetches paginated history from:
      /api/v0/equity/history/orders
      /api/v0/equity/history/transactions
      /api/v0/equity/history/dividends
    and upserts rows by reference ID / synthetic fill key.
    Safe to re-run — existing records are skipped via csv_id dedup.
    """
    sync_transactions()
