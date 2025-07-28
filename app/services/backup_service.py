# backend/app/services/backup_service.py
import sqlite3
import os
from ..core.config import settings
from .vector_service import vector_service
import logging
from typing import Optional, Any, Dict, List

# Build the absolute path to the database from settings
DB_PATH = os.path.join(settings.NEXUS_DATA_PATH, "nexus.sqlite")

def get_user_tables(conn: sqlite3.Connection) -> List[str]:
    """Gets a list of all user-created tables in the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    return [row[0] for row in cursor.fetchall()]

def dict_factory(cursor: sqlite3.Cursor, row: tuple) -> Dict[str, Any]:
    """A factory to return query results as dictionaries."""
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

async def create_backup() -> Dict[str, Any]:
    """Creates a full backup of SQLite and Vector DB by dynamically discovering tables."""
    logging.info("Starting backup creation process...")
    
    sqlite_data: Dict[str, List[Dict[str, Any]]] = {}
    try:
        logging.info(f"Connecting to SQLite database at: {DB_PATH}")
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"Database file not found at {DB_PATH}. Ensure the main application has run once to initialize it.")
            
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        
        tables_to_backup = get_user_tables(conn)
        logging.info(f"Found tables to back up: {tables_to_backup}")

        for table in tables_to_backup:
            cursor.execute(f"SELECT * FROM {table}")
            sqlite_data[table] = cursor.fetchall()
        conn.close()
        logging.info("SQLite data backup completed.")
    except Exception as e:
        logging.error(f"Error during SQLite backup: {e}", exc_info=True)
        raise

    try:
        logging.info("Backing up vector data...")
        vector_data = vector_service.get_all("knowledge_base")
        logging.info("Vector data backup completed.")
    except Exception as e:
        logging.error(f"Error during vector data backup: {e}", exc_info=True)
        raise

    return {
        "sqlite_data": sqlite_data,
        "vector_data": vector_data.model_dump()
    }

async def restore_from_backup(backup_data: Dict[str, Any]):
    """Clears all data and restores from a backup object."""
    logging.info("Starting restore process from backup data...")
    sqlite_data = backup_data.get("sqlite_data", {})
    vector_data = backup_data.get("vector_data", {})

    try:
        logging.info(f"Connecting to SQLite database for restore: {DB_PATH}")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA foreign_keys = OFF;")
        
        tables_to_clear = get_user_tables(conn)
        logging.info(f"Clearing existing SQLite data from tables: {tables_to_clear}")
        for table in tables_to_clear:
            cursor.execute(f"DELETE FROM {table}")
        
        logging.info(f"Restoring SQLite data for tables: {list(sqlite_data.keys())}")
        for table, rows in sqlite_data.items():
            if not rows:
                continue
            
            columns = ', '.join(rows[0].keys())
            placeholders = ', '.join(['?'] * len(rows[0]))
            sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            
            data_to_insert = [tuple(row.values()) for row in rows]
            cursor.executemany(sql, data_to_insert)

        cursor.execute("PRAGMA foreign_keys = ON;")
        conn.commit()
        conn.close()
        logging.info("SQLite restore completed.")
    except Exception as e:
        logging.error(f"Error during SQLite restore: {e}", exc_info=True)
        conn.rollback()
        conn.close()
        raise

    try:
        logging.info("Clearing and restoring vector data...")
        vector_service.clear_collection("knowledge_base")
        if vector_data and vector_data.get("ids"):
            vector_service._client.get_collection("knowledge_base").add(
                ids=vector_data["ids"],
                embeddings=vector_data["embeddings"],
                documents=vector_data["documents"],
                metadatas=vector_data["metadatas"]
            )
        logging.info("Vector data restore completed.")
    except Exception as e:
        logging.error(f"Error during vector data restore: {e}", exc_info=True)
        raise