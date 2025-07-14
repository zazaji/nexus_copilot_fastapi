# backend/app/main.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.v1.api import api_router
import logging
from .db_init import init_db

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

app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
def startup_event():
    """
    Actions to perform on application startup.
    """
    logging.info("FastAPI application starting up...")
    init_db()
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