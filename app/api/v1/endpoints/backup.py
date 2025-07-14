# backend/app/api/v1/endpoints/backup.py
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from app.services import backup_service
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/export")
async def export_data():
    logger.info("Received request for /export")
    try:
        backup_data = await backup_service.create_backup()
        logger.info("Export data created successfully.")
        return JSONResponse(content=backup_data)
    except Exception as e:
        logger.error(f"Export failed with exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.post("/import")
async def import_data(request: Request):
    logger.info("Received request for /import")
    try:
        backup_data = await request.json()
        await backup_service.restore_from_backup(backup_data)
        logger.info("Import completed successfully.")
        return {"status": "ok", "message": "Import successful. Please restart the main application."}
    except json.JSONDecodeError:
        logger.error("Import failed due to invalid JSON.")
        raise HTTPException(status_code=400, detail="Invalid JSON data provided.")
    except Exception as e:
        logger.error(f"Import failed with exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")