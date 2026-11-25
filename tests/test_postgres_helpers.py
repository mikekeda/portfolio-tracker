"""Keep integration-test schema creation confined to the approved local database."""

import pytest

from postgres_helpers import _validate_test_database_url


@pytest.mark.parametrize("url", [
    "postgresql+asyncpg://build_trading212_admin@localhost:5432/build_trading212",
    "postgresql+asyncpg://build_trading212_admin@127.0.0.1/build_trading212",
    "postgresql+asyncpg://build_trading212_admin@[::1]:5432/build_trading212",
    "postgresql+asyncpg://build_trading212_admin:secret@localhost:5432/build_trading212",
])
def test_accepts_the_approved_local_database(url):
    _validate_test_database_url(url)


@pytest.mark.parametrize("url", [
    "postgresql+asyncpg://build_trading212_admin@localhost/trading212",
    "postgresql+asyncpg://postgres@localhost/build_trading212",
    "postgresql+asyncpg://trading212_admin@localhost/build_trading212",
    "postgresql+asyncpg://test_trading212_admin@localhost/test_trading212",
    "postgresql+asyncpg://build_trading212_admin@production.example/build_trading212",
    "postgresql+asyncpg://build_trading212_admin@localhost/build_trading212?host=production.example",
    "postgresql+asyncpg://build_trading212_admin@localhost/build_trading212?port=5433",
    "postgresql+asyncpg://build_trading212_admin@localhost:5433/build_trading212",
    "postgresql+asyncpg://review_test@/postgres?host=/tmp&port=55439",
    "postgresql+psycopg2://build_trading212_admin@localhost/build_trading212",
])
def test_rejects_other_databases_and_connection_overrides(url):
    with pytest.raises(ValueError, match="allowed local test database"):
        _validate_test_database_url(url)
