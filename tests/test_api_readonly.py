"""The API is authenticated and read-only: nothing the UI sends can change state."""
import asyncio
import re
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import backend.app as app_module
from backend.views import app, earnings

TOKEN = "test-token-0123456789abcdef"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "API_TOKEN", TOKEN)
    return TestClient(app)


def test_every_route_is_get_only():
    methods = {method for route in app.routes if isinstance(route, APIRoute) for method in route.methods}
    assert methods <= {"GET", "HEAD"}


def test_views_only_use_the_read_only_session():
    views = list((Path(__file__).parents[1] / "backend" / "views").glob("*.py"))
    assert views
    for path in views:
        assert not re.search(r"\bget_session\b", path.read_text()), path


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_write_methods_are_rejected_even_with_a_valid_token(client, method):
    response = client.request(method, "/api/portfolio/summary", headers=AUTH)
    assert response.status_code == 405
    assert response.json() == {"detail": "The API is read-only"}


@pytest.mark.parametrize("header", [
    None, "Bearer", "Bearer a b", f"Basic {TOKEN}", "Bearer wrong-token", f"Bearer {TOKEN}x",
])
def test_bad_or_malformed_credentials_get_401_not_500(client, header):
    headers = {"Authorization": header} if header is not None else {}
    response = client.get("/api/portfolio/summary", headers=headers)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_unset_token_fails_closed(monkeypatch):
    monkeypatch.setattr(app_module, "API_TOKEN", "")
    response = TestClient(app).get("/api/portfolio/summary", headers={"Authorization": "Bearer "})
    assert response.status_code == 401


def test_valid_token_reaches_the_report_and_traversal_is_refused(client, monkeypatch, tmp_path):
    filings = tmp_path / "filings"
    (filings / "NVDA").mkdir(parents=True)
    (filings / "NVDA" / "2026-01-25_10-K.html").write_text("<p>10-K</p>")
    (tmp_path / "secret_x.html").write_text("outside the filings folder")
    monkeypatch.setattr(earnings, "FILINGS_DIR", filings)
    response = client.get("/api/earnings-report/NVDA/2026-01-25", headers=AUTH)
    assert response.status_code == 200 and response.text == "<p>10-K</p>"

    for symbol, report_date in [("..", "secret"), ("NVDA", "*"), ("..", "*"), ("NVDA", "2026-01-2?")]:
        with pytest.raises(HTTPException) as refused:
            asyncio.run(earnings.get_earnings_report_html(symbol, report_date))
        assert refused.value.status_code == 404


def test_db_session_is_read_only_and_never_commits(monkeypatch):
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.execute = AsyncMock()
    session.rollback = AsyncMock()
    session.commit = AsyncMock()
    monkeypatch.setattr(app_module, "_get_session_factory", lambda: lambda: session)

    async def use():
        dependency = app_module.get_db_session()
        assert await dependency.__anext__() is session
        await dependency.aclose()

    asyncio.run(use())
    assert str(session.execute.await_args.args[0]) == "SET TRANSACTION READ ONLY"
    session.rollback.assert_awaited_once()
    session.commit.assert_not_called()


def test_postgres_rejects_writes_through_the_api_session(monkeypatch):
    from sqlalchemy.exc import DBAPIError

    from models import CurrencyRateDaily
    from postgres_helpers import postgres_database

    async def check():
        async with postgres_database() as factory:
            monkeypatch.setattr(app_module, "_get_session_factory", lambda: factory)
            dependency = app_module.get_db_session()
            session = await dependency.__anext__()
            session.add(CurrencyRateDaily(from_currency="USD", to_currency="GBP", rate=0.8, date=date(2026, 11, 25)))
            with pytest.raises(DBAPIError, match="read-only transaction"):
                await session.flush()
            await dependency.aclose()

    asyncio.run(check())
