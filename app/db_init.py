# backend/app/db_init.py
import logging
import os
import sys
import sqlite3

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.database import create_connection, get_db_path

def init_db():
    """
    Initializes the database for the FastAPI application.
    This function is self-contained and ensures the DB path is resolved
    and all necessary tables are created.
    """
    logging.info("Attempting to initialize FastAPI database...")
    db_path = get_db_path()
    logging.info(f"Database path resolved to: {db_path}")

    conn = None
    try:
        conn = create_connection(db_path)
        if conn:
            logging.info("Database connection successful. Verifying tables...")
            cursor = conn.cursor()
            
            # --- Agent Tables ---
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_tasks (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                user_goal TEXT NOT NULL,
                status TEXT NOT NULL,
                mode TEXT NOT NULL DEFAULT 'plan',
                created_at INTEGER NOT NULL,
                updated_at INTEGER,
                final_report TEXT,
                plan TEXT,
                research_content TEXT
            )
            """)
            logging.info("Table 'agent_tasks' verified.")
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_task_steps (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                thought TEXT,
                action TEXT NOT NULL,
                action_input TEXT NOT NULL,
                observation TEXT,
                status TEXT NOT NULL,
                history TEXT,
                result TEXT,
                FOREIGN KEY (task_id) REFERENCES agent_tasks (id) ON DELETE CASCADE
            )
            """)
            logging.info("Table 'agent_task_steps' verified.")

            # --- Knowledge Graph Tables ---
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                file_path TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """)
            logging.info("Table 'notes' verified.")

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS note_links (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id),
                FOREIGN KEY (source_id) REFERENCES notes (id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES notes (id) ON DELETE CASCADE
            )
            """)
            logging.info("Table 'note_links' verified.")

            # --- Integration Tables ---
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS integration_tasks (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                integration_name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER,
                final_report TEXT
            )
            """)
            logging.info("Table 'integration_tasks' verified.")

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS integration_task_steps (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                description TEXT NOT NULL,
                details TEXT,
                status TEXT NOT NULL,
                FOREIGN KEY (task_id) REFERENCES integration_tasks (id) ON DELETE CASCADE
            )
            """)
            logging.info("Table 'integration_task_steps' verified.")

            # --- Statistics Table ---
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_call_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                service_name TEXT NOT NULL,
                model_identifier TEXT,
                timestamp INTEGER NOT NULL
            )
            """)
            logging.info("Table 'api_call_logs' verified.")
            
            # Add research_content column if it doesn't exist (for migration)
            try:
                cursor.execute("ALTER TABLE agent_tasks ADD COLUMN research_content TEXT;")
                logging.info("Added 'research_content' column to 'agent_tasks' table.")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    pass # Column already exists, which is fine
                else:
                    raise

            # Add api_config column if it doesn't exist (for migration)
            try:
                cursor.execute("ALTER TABLE agent_tasks ADD COLUMN api_config TEXT;")
                logging.info("Added 'api_config' column to 'agent_tasks' table.")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    pass # Column already exists, which is fine
                else:
                    raise
            
            conn.commit()
            print("Database initialization complete.")
        else:
            logging.error(f"Failed to create a database connection to {db_path}.")
    except Exception as e:
        logging.error(f"Database initialization failed: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed after initialization.")


if __name__=="__main__":
    init_db()