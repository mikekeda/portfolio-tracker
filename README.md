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
