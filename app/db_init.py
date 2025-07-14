# backend/app/db_init.py
import logging
import os
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
            
            # --- Create/Update agent_tasks table ---
            cursor.execute("PRAGMA table_info(agent_tasks)")
            columns = [col[1] for col in cursor.fetchall()]

            if not columns:
                # Table does not exist, create it
                cursor.execute("""
                CREATE TABLE agent_tasks (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    user_goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER,
                    final_report TEXT
                )
                """)
                logging.info("Table 'agent_tasks' created.")
            else:
                # Table exists, add columns if they don't exist
                if 'updated_at' not in columns:
                    cursor.execute("ALTER TABLE agent_tasks ADD COLUMN updated_at INTEGER")
                    logging.info("Column 'updated_at' added to 'agent_tasks'.")
                if 'final_report' not in columns:
                    cursor.execute("ALTER TABLE agent_tasks ADD COLUMN final_report TEXT")
                    logging.info("Column 'final_report' added to 'agent_tasks'.")

            logging.info("Table 'agent_tasks' verified.")
            
            # --- Create agent_task_steps table ---
            # This table structure is flexible enough for the new agent logic,
            # so no changes are needed. It will store the sequence of thoughts,
            # actions, and observations as they are dynamically generated.
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
                FOREIGN KEY (task_id) REFERENCES agent_tasks (id) ON DELETE CASCADE
            )
            """)
            logging.info("Table 'agent_task_steps' verified.")
            
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