import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Any, Dict, List
import sqlite3

from app.database import get_db_connection
from app.services import mcp_service

router = APIRouter()
logging.basicConfig(level=logging.INFO)

INTEGRATION_TEMPLATES = [
    {
        "id": "zapier_inbound_note",
        "name": "Zapier: Create Note",
        "description": "Create a new note in your knowledge base from a Zapier webhook.",
        "serviceType": "inbound_webhook"
    },
    {
        "id": "custom_inbound_note",
        "name": "Custom Webhook: Create Note",
        "description": "Create a new note from any service that can send a POST request.",
        "serviceType": "inbound_webhook"
    }
]

@router.get("/templates")
async def get_integration_templates() -> List[Dict[str, Any]]:
    """
    Returns a list of available integration templates for the frontend.
    """
    return INTEGRATION_TEMPLATES

@router.post("/webhooks/{integration_id}", status_code=202)
async def handle_inbound_webhook(integration_id: str, request: Request, conn: sqlite3.Connection = Depends(get_db_connection)):
    """
    A generic endpoint to receive data from external services like Zapier.
    """
    try:
        payload = await request.json()
        logging.info(f"Received webhook for integration_id: {integration_id} with payload: {payload}")

        # Process the webhook in the background to respond quickly
        await mcp_service.process_webhook(integration_id, payload, conn)

        return {"status": "received", "message": "Webhook payload is being processed."}
    except Exception as e:
        logging.error(f"Error processing webhook for {integration_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process webhook.")