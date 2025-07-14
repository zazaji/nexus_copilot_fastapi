# backend/main.py
import uvicorn
from fastapi import FastAPI
from app.api.v1.api import api_router
from app.core.config import settings
import os
import typer

cli = typer.Typer()

app = FastAPI(
    title="Nexus Copilot Brain Service",
    version="2.0.0",
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Nexus Copilot Brain Service is running."}

@cli.command()
def main(
    host: str = "127.0.0.1",
    port: int = 8008,
    app_data_dir: str = typer.Option("", help="Path to the application data directory."),
):
    """
    Run the Nexus Copilot Brain Service.
    """
    # This is the most reliable way to set the data path for the entire application
    if app_data_dir:
        settings.NEXUS_DATA_PATH = app_data_dir
        print(f"Brain Service using data directory: {settings.NEXUS_DATA_PATH}")
    else:
        print("Warning: Running Brain Service without a specified app-data-dir.")

    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    cli()