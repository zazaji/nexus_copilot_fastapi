# backend/app/api/v1/api.py
from fastapi import APIRouter
from app.api.v1.endpoints import (
    vector, backup, proxy, convert, agent, integrations, 
    knowledge_base, tools, audio, dashboard, system, creations,
    dev
)

api_router = APIRouter()

api_router.include_router(vector.router, prefix="/vector", tags=["Vector Service"])
api_router.include_router(backup.router, prefix="/backup", tags=["Backup & Restore"])
api_router.include_router(proxy.router, prefix="/proxy", tags=["LLM Proxy"])
api_router.include_router(convert.router, prefix="/convert", tags=["Conversion"])
api_router.include_router(agent.router, prefix="/agent", tags=["Agent"])
api_router.include_router(integrations.router, prefix="/integrations", tags=["Integrations"])
api_router.include_router(knowledge_base.router, prefix="/knowledge_base", tags=["Knowledge Base"])
api_router.include_router(tools.router, prefix="/tools", tags=["Tools"])
api_router.include_router(audio.router, prefix="/audio", tags=["Audio"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["Dashboard"])
api_router.include_router(system.router, prefix="/system", tags=["System"])
api_router.include_router(creations.router, prefix="/creations", tags=["Creations"])
api_router.include_router(dev.router, prefix="/dev", tags=["dev_test"])
