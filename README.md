# Trading212 Portfolio Manager

A comprehensive portfolio management application that integrates with Trading212 API and Yahoo Finance to provide real-time portfolio tracking, analysis, and screening capabilities.

## Features

- **Portfolio Tracking**: Real-time portfolio value, profit/loss, and performance metrics
- **Technical Analysis**: RSI, SMA, Bollinger Bands, and other technical indicators
- **Stock Screening**: Advanced screening capabilities with customizable criteria
- **Chart Visualization**: Interactive charts for price and metric analysis
- **Top Movers**: Track top gainers and losers across different time periods
- **Allocation Analysis**: Sector and country allocation breakdowns
- **Async Backend**: High-performance async FastAPI backend
- **Modern Frontend**: React-based responsive web interface

## Architecture

- **Backend**: FastAPI with async SQLAlchemy and PostgreSQL
- **Frontend**: React with modern UI components and charts
- **Data Sources**: Trading212 API, Yahoo Finance API
- **Database**: PostgreSQL with optimized queries and caching
- **Background Tasks**: Celery with Redis broker for scheduled data updates

## Background Tasks (Celery)

The application uses Celery for periodic background data updates. Tasks are automatically scheduled via Celery Beat.

### Scheduled Tasks

- **`update_data_task`**: Updates portfolio holdings, prices, currency rates, and technical indicators
  - Runs every 4 hours
  - Fetches fresh data from Trading212 API and Yahoo Finance

- **`calculate_portfolio_returns_task`**: Calculates Money-Weighted Rate of Return (MWRR) and Time-Weighted Rate of Return (TWRR)
  - Runs every 8 hours
  - Reconstructs historical portfolio values from transaction history

### Running Celery

```bash
# Start Celery worker (task executor)
celery -A celery_tasks.celery_app worker --loglevel=info

# Start Celery beat (task scheduler)
celery -A celery_tasks.celery_app beat --loglevel=info
```

### Running Scripts Manually

All scripts are located in the `scripts/` directory. When running manually, you must set the `PYTHONPATH` environment variable to the project root:

```bash
# From the project root directory
PYTHONPATH=/home/voron/sites/portfolio_tracker python scripts/backfill_portfolio_daily.py
PYTHONPATH=/home/voron/sites/portfolio_tracker python scripts/update_data.py
```

**Available Scripts:**

- **`backfill_portfolio_daily.py`**: Backfill and calculate MWRR/TWRR metrics for historical dates
- **`update_data.py`**: Update all database tables with fresh API data
- **`backfill_currency_rates.py`**: Backfill historical currency exchange rates
- **`update_history_from_csv.py`**: Import Trading212 CSV transaction exports
- **`update_pies.py`**: Update Trading212 Pies data
- **`scrape_macrotrends_pe.py`**: Scrape PE ratios from Macrotrends
- **`scrape_wisesheets_pe.py`**: Scrape PE ratios from Wisesheets

## Code Quality

This project maintains high code quality standards with comprehensive typing, documentation, and automated quality checks.

### Quality Tools

```bash
# Code style and linting
pycodestyle --max-line-length 120 --exclude frontend/ .
flake8 --exclude frontend/ --max-line-length=120 .
pylint --max-line-length=120 backend/

# Code formatting
ruff format --line-length 120

# Type checking
mypy .
mypy . --check-untyped-def

# Import organization
isort .
```

### Quality Standards

- **100% Type Coverage**: All functions have comprehensive type annotations
- **Complete Documentation**: All functions have detailed docstrings
- **Async Best Practices**: Proper async/await patterns throughout
- **Error Handling**: Comprehensive error handling and logging
- **Code Organization**: Clean separation of concerns and modular design
- **Performance**: Optimized database queries and efficient data processing


### Documentation Standards

- Module-level docstrings explaining purpose and functionality
- Function docstrings with Args, Returns, and Raises sections
- Inline comments for complex business logic
- Type hints for all parameters and return values
