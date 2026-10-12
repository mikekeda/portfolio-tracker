"""Keep integration-test schema creation confined to approved local databases."""

import pytest

from postgres_helpers import _validate_test_database_url


@pytest.mark.parametrize("url", [
    "postgresql+asyncpg://test_trading212_admin@localhost:5432/test_trading212",
    "postgresql+asyncpg://test_trading212_admin@127.0.0.1/test_trading212",
    "postgresql+asyncpg://test_trading212_admin@[::1]:5432/test_trading212",
    "postgresql+asyncpg://review_test@/postgres?host=/tmp&port=55439",
])
def test_accepts_approved_local_databases(url):
    _validate_test_database_url(url)


@pytest.mark.parametrize("url", [
    "postgresql+asyncpg://test_trading212_admin@localhost/trading212",
    "postgresql+asyncpg://postgres@localhost/test_trading212",
    "postgresql+asyncpg://test_trading212_admin@production.example/test_trading212",
    "postgresql+asyncpg://test_trading212_admin@localhost/test_trading212?host=production.example",
    "postgresql+asyncpg://test_trading212_admin@localhost/test_trading212?port=5433",
    "postgresql+asyncpg://test_trading212_admin@localhost:5433/test_trading212",
    "postgresql+asyncpg://review_test@/trading212?host=/tmp&port=55439",
    "postgresql+asyncpg://review_test@/postgres?host=/tmp&port=5432",
    "postgresql+psycopg2://test_trading212_admin@localhost/test_trading212",
])
def test_rejects_other_databases_and_connection_overrides(url):
    with pytest.raises(ValueError, match="allowed local test database"):
        _validate_test_database_url(url)
