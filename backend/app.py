import os

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession

API_TOKEN = os.getenv("API_TOKEN")
app = FastAPI(title="Trading212 Portfolio API", description="API for accessing portfolio data", version="1.0.0")


@app.middleware("http")
async def check_api_token(request: Request, call_next):
    auth_header = request.headers.get("Authorization")

    if not auth_header:
        # If no auth header, return 401
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Authorization header missing"},
        )

    # TODO: Expect "Bearer YOUR_TOKEN", could crash on malformed token
    scheme, token = auth_header.split()
    if scheme.lower() != "bearer" or token != API_TOKEN:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Invalid authorization header"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    # If the token is valid, proceed with the request
    response = await call_next(request)

    return response


# Add CORS middleware
app.add_middleware(CORSMiddleware, allow_origins=[os.getenv("DOMAIN") or "http://localhost:3000"])


@lru_cache(maxsize=1)
def _get_session_factory() -> async_sessionmaker:
    """Create and cache the async database session factory."""
    db_url = "postgresql+asyncpg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}".format(
        db_name=os.getenv("DB_NAME", "trading212_portfolio"),
        db_password=os.getenv("DB_PASSWORD"),
        db_user=os.getenv("DB_USER", "postgres"),
        db_host=os.getenv("DB_HOST", "localhost"),
        db_port=os.getenv("DB_PORT", "5432"),
    )
    engine = create_async_engine(db_url, echo=False, pool_pre_ping=True)

    return async_sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


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
    """FastAPI dependency for database sessions."""
    async with get_session() as session:
        yield session
