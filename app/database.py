# backend/app/database.py
import os
import sqlite3
import logging
from fastapi import HTTPException
from app.core.config import settings

def create_connection(db_path: str) -> sqlite3.Connection | None:
    """Creates a new database connection to the SQLite database."""
    try:
        # This is the key change to allow multi-threaded access.
        # FastAPI runs endpoints in a thread pool, and our background tasks
        # might run in different threads.
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logging.error(f"Error connecting to database at {db_path}: {e}", exc_info=True)
        return None

def get_db_path() -> str:
    """Gets the full path to the SQLite database file from settings."""
    return os.path.join(settings.NEXUS_DATA_PATH, "nexus.sqlite")

def get_db_connection():
    """
    Provides a database connection. To be used with FastAPI's dependency injection.
    This function is a generator that yields a connection and ensures it's closed.
    """
    db_path = get_db_path()
    conn = create_connection(db_path)
    if conn is None:
        raise HTTPException(status_code=500, detail=f"Database connection failed at path: {db_path}")
    
    try:
        yield conn
    finally:
        if conn:
            conn.close()

def get_db_connection_for_bg() -> sqlite3.Connection | None:
    """
    Provides a direct database connection for background tasks.
    The caller is responsible for closing the connection.
    """
    db_path = get_db_path()
    return create_connection(db_path)