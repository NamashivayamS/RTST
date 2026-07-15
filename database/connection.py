# database/connection.py
"""
MS SQL Server connection handling for RealTimeSpeechTranslator (via pyodbc).

Why not a manual pool like the old ThreadedConnectionPool?
────────────────────────────────────────────────────────────
pyodbc enables ODBC-level connection pooling by default (pyodbc.pooling = True,
set once at import time, before the first connection is opened). The ODBC driver
manager keeps physical connections alive and hands them back out on pyodbc.connect()
calls with a matching connection string — so calling connect() per borrow here is
NOT equivalent to psycopg2's old per-call handshake cost.

get_connection()/release_connection(conn) are kept as the public interface so
queries.py needed no changes to its calling convention, only its SQL syntax.

Usage (unchanged from the Postgres version)
─────
  from database.connection import get_connection, release_connection

  conn = get_connection()
  try:
      cur = conn.cursor()
      ...
      conn.commit()
  finally:
      release_connection(conn)
"""

import logging
import time
import pyodbc

from config import (
    MSSQL_HOST,
    MSSQL_PORT,
    MSSQL_DB,
    MSSQL_USER,
    MSSQL_PASSWORD,
    MSSQL_DRIVER,
    MSSQL_ENCRYPT,
)

logger = logging.getLogger("ispeak.db")

# Must be set before the first pyodbc.connect() call.
pyodbc.pooling = True

_CONN_STR = (
    f"DRIVER={MSSQL_DRIVER};"
    f"SERVER={MSSQL_HOST},{MSSQL_PORT};"
    f"DATABASE={MSSQL_DB};"
    f"UID={MSSQL_USER};"
    f"PWD={MSSQL_PASSWORD};"
    f"Encrypt={MSSQL_ENCRYPT};"
    f"TrustServerCertificate=yes;"   # internal/self-signed server cert — set to 'no'
                                      # and supply a real cert if this ever crosses
                                      # an untrusted network
)


_initialised = False


def init_pool(minconn: int = 2, maxconn: int = 20, retries: int = 10, retry_delay: float = 3.0) -> None:
    """
    Verifies SQL Server is reachable before the app starts accepting connections.
    minconn/maxconn are accepted for call-site compatibility with main.py's
    startup_event but aren't used directly — ODBC's own pool sizing is controlled
    by the driver manager, not per-process here.
    """
    global _initialised
    if _initialised:
        logger.warning("[DB Pool] init_pool() called more than once — ignoring.")
        return

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            conn = pyodbc.connect(_CONN_STR, timeout=5)
            conn.close()
            _initialised = True
            logger.info(
                "[DB Pool] Connected to MSSQL at %s:%s/%s (attempt %d).",
                MSSQL_HOST, MSSQL_PORT, MSSQL_DB, attempt,
            )
            return
        except pyodbc.Error as e:
            last_err = e
            logger.warning(
                "[DB Pool] MSSQL not ready yet (attempt %d/%d): %s",
                attempt, retries, e,
            )
            time.sleep(retry_delay)

    raise RuntimeError(f"[DB Pool] Could not connect to MSSQL after {retries} attempts: {last_err}")


def close_pool() -> None:
    """No persistent pool object to drain — ODBC manages its own connections.
    Kept for call-site compatibility with main.py's shutdown_event."""
    global _initialised
    _initialised = False
    logger.info("[DB Pool] Shutdown acknowledged (ODBC pool managed by driver manager).")


def get_connection():
    """Borrow a (pooled) connection. Always pair with release_connection()."""
    if not _initialised:
        raise RuntimeError(
            "[DB Pool] Not initialised. Call database.connection.init_pool() in startup_event first."
        )
    return pyodbc.connect(_CONN_STR, timeout=10)


def release_connection(conn) -> None:
    """Return a connection. With ODBC pooling enabled, .close() returns the
    physical connection to the pool rather than tearing down the TCP session."""
    if conn is not None:
        conn.close()
