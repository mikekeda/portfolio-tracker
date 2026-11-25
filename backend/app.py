import hmac
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import API_TOKEN, DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER, DOMAIN, logger

# The API is read-only by design: nothing the UI sends may change state.
READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

app = FastAPI(title="Trading212 Portfolio API", description="API for accessing portfolio data", version="1.0.0")


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        raise exc  # keeps original status/detail

    logger.exception("Unhandled server error", extra={"path": request.url.path})
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.middleware("http")
async def check_api_token(request: Request, call_next):
    """Reject every non-read method, then require the shared bearer token.

    Fails closed when API_TOKEN is unset, and answers a malformed header with 401.
    """
    if request.method not in READ_METHODS:
        return JSONResponse(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
            content={"detail": "The API is read-only"},
            headers={"Allow": ", ".join(sorted(READ_METHODS))},
        )

    parts = request.headers.get("Authorization", "").split()
    token = parts[1] if len(parts) == 2 and parts[0].lower() == "bearer" else ""
    if not (API_TOKEN and token and hmac.compare_digest(token.encode(), API_TOKEN.encode())):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Invalid or missing bearer token"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    return await call_next(request)


# Add CORS middleware - allow localhost, local network IPs, and configured DOMAIN
# The regex matches: localhost, 127.0.0.1, and private network IP ranges (192.168.x.x, 10.x.x.x, 172.16-31.x.x)
# DOMAIN is also included in allow_origins for production deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=[DOMAIN] if DOMAIN else [],
    allow_origin_regex=r"http(s)?://(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.\d{1,3}\.\d{1,3})(:\d+)?",
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _get_engine():
    db_url = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_async_engine(db_url, echo=False, pool_pre_ping=True)


@lru_cache(maxsize=1)
def _get_session_factory() -> async_sessionmaker:
    """Create and cache the async database session factory."""
    return async_sessionmaker(bind=_get_engine(), autoflush=False, autocommit=False, expire_on_commit=False)


async def dispose_engine(close: bool = True) -> None:
    """Release the cached engine's pooled connections.

    Async Celery tasks run each job in a fresh asyncio.run() loop, but the
    lru_cached engine outlives the loop — a pooled connection reused by the
    next task then awaits futures bound to a closed loop (RuntimeError:
    "attached to a different loop"). Tasks must dispose inside their own loop
    when done; close=False only abandons the pool without touching connections
    (safe when the pool may hold another loop's connections).
    """
    await _get_engine().dispose(close=close)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for database sessions."""
    AsyncSessionLocal = _get_session_factory()
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: a read-only session, always rolled back.

    Postgres rejects any INSERT/UPDATE/DELETE in it, so no endpoint can change
    state even by mistake. Writers (scripts, Celery) use get_session() instead.
    """
    async with _get_session_factory()() as session:
        await session.execute(text("SET TRANSACTION READ ONLY"))
        try:
            yield session
        finally:
            await session.rollback()
