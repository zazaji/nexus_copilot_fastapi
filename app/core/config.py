# backend/app/core/config.py
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import logging

load_dotenv()

# --- Path Configuration ---
# Get the directory of the current file (config.py)
# backend/app/core/config.py -> backend/app/core
CORE_DIR = Path(__file__).parent.resolve()
# Go up to the app directory: backend/app
APP_DIR = CORE_DIR.parent
# Go up to the backend directory: backend/
BACKEND_DIR = APP_DIR.parent

class Settings(BaseSettings):
    # Set NEXUS_DATA_PATH to be the 'backend' directory itself.
    # This makes the path predictable and independent of the launch environment.
    NEXUS_DATA_PATH: str = str(BACKEND_DIR)
    
    # TAVILY_API_KEY is no longer loaded from environment variables here.
    # It will be managed via the application's UI settings and passed in API calls.

    @property
    def CHROMA_PERSIST_PATH(self) -> str:
        # Vector data will be stored in backend/chroma_data
        return os.path.join(self.NEXUS_DATA_PATH, "chroma_data")

    class Config:
        case_sensitive = True

settings = Settings()

# Log the resolved path for clarity during startup
logging.info(f"NEXUS_DATA_PATH resolved to: {settings.NEXUS_DATA_PATH}")