"""Opt-in integration tests against an isolated local PostgreSQL instance.

Set T212_TEST_POSTGRES_URL to
postgresql+asyncpg://build_trading212_admin@localhost:5432/build_trading212 (the
same database name on a dev machine and in the server's Jenkins job).
Each case creates and removes its own schema. No application DB configuration is used.
"""

import importlib
import os
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from models import Base, TradeAgentRun

# The only role/database these tests may create schemas in; it owns nothing else.
TEST_ROLE, TEST_DATABASE = "build_trading212_admin", "build_trading212"


def _validate_test_database_url(url: str) -> None:
    parsed = make_url(url)
    allowed = (
        parsed.drivername == "postgresql+asyncpg"
        and (parsed.username, parsed.database) == (TEST_ROLE, TEST_DATABASE)
        and parsed.host in {"localhost", "127.0.0.1", "::1"}
        and parsed.port in {None, 5432}
        and not parsed.query
    )
    if not allowed:
        raise ValueError("Integration tests require an explicitly allowed local test database")


@asynccontextmanager
async def postgres_database():
    url = os.getenv("T212_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("set T212_TEST_POSTGRES_URL for local PostgreSQL integration tests")
    _validate_test_database_url(url)
    schema = "test_" + uuid4().hex
    admin = create_async_engine(url)
    engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
    try:
        async with admin.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        async with engine.begin() as conn:
            await conn.run_sync(
                lambda c: Base.metadata.create_all(
                    c, tables=[t for t in Base.metadata.sorted_tables if t is not TradeAgentRun.__table__]
                )
            )

            # Load by path: alembic/versions is not a Python package.
            def migrate_from_file(connection):
                from pathlib import Path

                spec = importlib.util.spec_from_file_location(
                    "agent_run_migration",
                    Path(__file__).parents[1] / "alembic/versions/d91f7b203ac6_add_trade_agent_runs.py",
                )
                migration = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(migration)
                with Operations.context(MigrationContext.configure(connection)):
                    migration.upgrade()

            await conn.run_sync(migrate_from_file)
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()
        async with admin.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await admin.dispose()
