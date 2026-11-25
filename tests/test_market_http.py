"""Upstream fetches made during API requests are bounded, and latest-row lookups stay one row."""
import asyncio
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from aiohttp import web

from backend.views import portfolio
from utils import market_data

ROOT = Path(__file__).parents[1]


def test_api_code_only_uses_the_bounded_session():
    sources = list((ROOT / "backend").rglob("*.py"))
    assert sources
    for path in sources:
        assert not re.search(r"aiohttp\.ClientSession\(", path.read_text()), path


def test_slow_upstream_times_out_and_http_errors_raise(monkeypatch):
    monkeypatch.setattr(market_data, "HTTP_TIMEOUT", aiohttp.ClientTimeout(total=0.2))

    async def slow(request):
        await asyncio.sleep(1)
        return web.Response(text="late")

    async def broken(request):
        return web.Response(status=500)

    async def check():
        app = web.Application()
        app.router.add_get("/slow", slow)
        app.router.add_get("/broken", broken)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        try:
            async with market_data.market_http_session() as session:
                with pytest.raises(asyncio.TimeoutError):
                    await session.get(f"http://127.0.0.1:{port}/slow")
                with pytest.raises(aiohttp.ClientResponseError):
                    await session.get(f"http://127.0.0.1:{port}/broken")
        finally:
            await runner.cleanup()

    asyncio.run(check())


@pytest.mark.parametrize("endpoint", [portfolio.get_portfolio_summary, portfolio.get_portfolio_allocations])
def test_latest_snapshot_lookup_reads_one_row(endpoint):
    result = MagicMock()
    result.scalars.return_value.first.return_value = None
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)

    assert asyncio.run(endpoint(session=session)) == {"error": "No portfolio data available"}
    sql = str(session.execute.await_args.args[0].compile(compile_kwargs={"literal_binds": True}))
    assert sql.rstrip().endswith("LIMIT 1")
