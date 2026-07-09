"""
Route registration: imports all routers and includes them into the FastAPI app.
`app` is re-exported so that `run.py` can do `from backend.views import app`.
"""

from backend.app import app, get_db_session  # noqa: F401  (get_db_session re-exported for convenience)
from backend.views import chart, earnings, form13f, instrument, market, portfolio, projection, risk, transactions

app.include_router(portfolio.router)
app.include_router(risk.router)
app.include_router(instrument.router)
app.include_router(chart.router)
app.include_router(market.router)
app.include_router(form13f.router)
app.include_router(earnings.router)
app.include_router(transactions.router)
app.include_router(projection.router)

__all__ = ["app", "get_db_session"]
