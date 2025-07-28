# backend/app/api/v1/endpoints/tools.py
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from app.services import tools as tools_service

router = APIRouter()
logging.basicConfig(level=logging.INFO)

class WebhookExecutionRequest(BaseModel):
    url: str
    params: Dict[str, Any]

@router.post("/execute-webhook-tool", status_code=200)
async def execute_webhook_tool(request: WebhookExecutionRequest):
    """
    Receives a request from the Rust backend to execute a webhook tool.
    """
    try:
        result = await tools_service.execute_webhook_tool(request.url, request.params)
        return {"status": "ok", "result": result}
    except Exception as e:
        logging.error(f"Webhook tool execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))