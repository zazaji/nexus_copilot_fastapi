# backend/app/main.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .api.v1.api import api_router
import logging
import os
from .db_init import init_db
from .core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Nexus Backend",
    version="2.0.0",
    description="Backend services for the Nexus application, including vector DB and LLM proxy."
)

# CORS (Cross-Origin Resource Sharing) Middleware Configuration
origins = [
    "http://localhost:1631",  # The default Vite dev server port for the frontend
    "tauri://localhost",      # The origin for Tauri's custom protocol
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the 'files' directory to serve agent logs and reports
files_dir = os.path.join(settings.NEXUS_DATA_PATH, "files")
os.makedirs(files_dir, exist_ok=True)
app.mount("/files", StaticFiles(directory=files_dir), name="files")

# Mount the 'tasks' directory to serve generated files for preview
tasks_dir = os.path.join(settings.NEXUS_DATA_PATH, "tasks")
os.makedirs(tasks_dir, exist_ok=True)
app.mount("/tasks", StaticFiles(directory=tasks_dir), name="tasks")


app.include_router(api_router, prefix="/api/v1")

def create_example_scripts():
    """Creates example scripts on first startup if they don't exist."""
    scripts_dir = os.path.join(settings.NEXUS_DATA_PATH, "scripts", "system")
    os.makedirs(scripts_dir, exist_ok=True)

    memory_script_path = os.path.join(scripts_dir, "get_memory.py")
    if not os.path.exists(memory_script_path):
        memory_script_content = """
# ### NEXUS-TOOL ###
# {
#   "name": "get_memory_usage",
#   "description": "Get the current system RAM and swap memory usage.",
#   "input_schema": {
#     "type": "object",
#     "properties": {}
#   }
# }
# ### NEXUS-TOOL-END ###
import psutil
import json

def get_memory_usage():
    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()

    def bytes_to_gb(b):
        return round(b / (1024**3), 2)

    usage = {
        "ram_total_gb": bytes_to_gb(ram.total),
        "ram_used_gb": bytes_to_gb(ram.used),
        "ram_percent": ram.percent,
        "swap_total_gb": bytes_to_gb(swap.total),
        "swap_used_gb": bytes_to_gb(swap.used),
        "swap_percent": swap.percent
    }
    print(json.dumps(usage))

if __name__ == "__main__":
    get_memory_usage()
"""
        with open(memory_script_path, "w", encoding="utf-8") as f:
            f.write(memory_script_content)
        logging.info(f"Created example script: {memory_script_path}")


@app.on_event("startup")
def startup_event():
    """
    Actions to perform on application startup.
    """
    logging.info("FastAPI application starting up...")
    init_db()
    create_example_scripts()
    logging.info("Startup event complete.")

@app.on_event("shutdown")
def shutdown_event():
    """Actions to perform on application shutdown."""
    logging.info("FastAPI application shutting down.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Nexus Backend"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8008, reload=True)