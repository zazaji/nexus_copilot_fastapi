# backend/app/api/v1/api.py
from fastapi import APIRouter
from app.api.v1.endpoints import vector, dev, backup, proxy, convert, agent

api_router = APIRouter()

api_router.include_router(vector.router, prefix="/vector", tags=["Vector Service"])
api_router.include_router(dev.router, prefix="/dev", tags=["Development"])
api_router.include_router(backup.router, prefix="/backup", tags=["Backup & Restore"])
api_router.include_router(proxy.router, prefix="/proxy", tags=["LLM Proxy"])
api_router.include_router(convert.router, prefix="/convert", tags=["Conversion"])
api_router.include_router(agent.router, prefix="/agent", tags=["Agent"])