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
    # Step 1: Get the database path from the centralized function
    db_path = get_db_path()
    logging.info(f"Database path resolved to: {db_path}")

    # Step 2: Create connection and initialize tables
    conn = None
    try:
        conn = create_connection(db_path)
        if conn:
            logging.info("Database connection successful. Verifying tables...")
            cursor = conn.cursor()
            
            # Create agent_tasks table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_tasks (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                user_goal TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """)
            logging.info("Table 'agent_tasks' verified.")
            
            # Create agent_task_steps table
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