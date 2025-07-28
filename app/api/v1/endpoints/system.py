# backend/app/api/v1/endpoints/system.py
import logging
from fastapi import APIRouter
from pydantic import BaseModel
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

class DataPathResponse(BaseModel):
    data_path: str

@router.get("/data-path", response_model=DataPathResponse)
async def get_data_path():
    """
    Returns the absolute path to the backend's data directory.
    """
    return DataPathResponse(data_path=settings.NEXUS_DATA_PATH)