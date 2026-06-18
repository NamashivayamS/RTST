# database/connection.py
"""
PostgreSQL connection pool for RealTimeSpeechTranslator.

Why a pool instead of get_connection() / conn.close()?
────────────────────────────────────────────────────────
The old pattern opened a fresh TCP connection to Postgres on every DB call
and tore it down immediately after. Under concurrent WebSocket sessions
(multiple speakers, each firing utterances every few seconds) this means:

  • New TCP handshake + TLS negotiation on every INSERT   (~5–15 ms penalty)
  • PostgreSQL spawns a new backend process per connection (memory pressure)
  • Postgres default max_connections=100 is exhausted quickly under load

ThreadedConnectionPool pre-opens `minconn` connections at startup and keeps
them alive, lending them out and reclaiming them — exactly like a database
ORM would. The pool is thread-safe, which matters because our DB writes run
inside loop.run_in_executor() on background threads.

Usage
─────
  from database.connection import get_connection, release_connection

  conn = get_connection()
  try:
      cur = conn.cursor()
      ...
      conn.commit()
  finally:
      release_connection(conn)   # returns conn to the pool, does NOT close it

Pool lifecycle
──────────────
  Call init_pool() once at FastAPI startup.
  Call close_pool() once at FastAPI shutdown.
  Never call conn.close() directly — use release_connection() instead.
"""

import logging
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from config import (
    POSTGRES_HOST,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
)

logger = logging.getLogger("ispeak.db")

# The pool is module-level so it is shared across all threads and all
# requests for the lifetime of the process.
_pool: pool.ThreadedConnectionPool | None = None


def init_pool(minconn: int = 2, maxconn: int = 20) -> None:
    """
    Create the connection pool.
    Call this ONCE inside FastAPI's startup_event before any DB query runs.

    minconn: connections kept open even when idle. 2 is enough for most demos.
    maxconn: hard ceiling. Under heavy load, get_connection() will block
             (up to the timeout set in psycopg2) rather than exceeding this.
             Set to (expected_concurrent_users * 2) as a starting heuristic.
    """
    global _pool

    if _pool is not None:
        logger.warning("[DB Pool] init_pool() called more than once — ignoring.")
        return

    logger.info(
        f"[DB Pool] Initialising ThreadedConnectionPool "
        f"(min={minconn}, max={maxconn}) → "
        f"{POSTGRES_USER}@{POSTGRES_HOST}/{POSTGRES_DB}"
    )

    _pool = pool.ThreadedConnectionPool(
        minconn=minconn,
        maxconn=maxconn,
        host=POSTGRES_HOST,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        # RealDictCursor makes row['column_name'] work everywhere in queries.py
        cursor_factory=RealDictCursor,
    )

    logger.info("[DB Pool] Pool ready.")


def close_pool() -> None:
    """
    Drain and close all connections in the pool.
    Call this inside FastAPI's shutdown_event.
    """
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
        logger.info("[DB Pool] All connections closed.")


def get_connection():
    """
    Borrow a connection from the pool.
    Always pair with release_connection() in a finally block.
    Raises RuntimeError if init_pool() was never called.
    """
    if _pool is None:
        raise RuntimeError(
            "[DB Pool] Pool is not initialised. "
            "Call database.connection.init_pool() in startup_event first."
        )
    return _pool.getconn()


def release_connection(conn) -> None:
    """
    Return a borrowed connection to the pool.
    If conn is None (e.g. get_connection() raised), this is a safe no-op.
    """
    if _pool is not None and conn is not None:
        _pool.putconn(conn)