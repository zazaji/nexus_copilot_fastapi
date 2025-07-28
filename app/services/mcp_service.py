# backend/app/services/mcp_service.py
import json
import logging
import sqlite3
import time
import uuid
import httpx
from typing import Any, Dict, List

from app.database import get_db_connection_for_bg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INTEGRATION_TEMPLATES = [
    {
        "id": "zapier_webhook_out",
        "name": "Zapier / Custom Webhook",
        "description": "Send data to any external service that provides a webhook URL.",
        "serviceType": "outbound_webhook"
    }
]

def get_templates() -> List[Dict[str, Any]]:
    return INTEGRATION_TEMPLATES

def create_task_in_db(conn: sqlite3.Connection, task_id: str, conversation_id: str, integration_name: str):
    current_time = int(time.time() * 1000)
    conn.execute(
        "INSERT INTO integration_tasks (id, conversation_id, integration_name, status, created_at) VALUES (?, ?, ?, ?, ?)",
        (task_id, conversation_id, integration_name, "running", current_time)
    )
    conn.commit()

def add_task_step(conn: sqlite3.Connection, task_id: str, step_index: int, description: str, status: str = "running", details: str = None):
    step_id = f"{task_id}-step-{step_index}"
    conn.execute(
        "INSERT INTO integration_task_steps (id, task_id, step_index, description, details, status) VALUES (?, ?, ?, ?, ?, ?)",
        (step_id, task_id, step_index, description, details, status)
    )
    conn.commit()
    return step_id

def update_step_status(conn: sqlite3.Connection, step_id: str, status: str, details: str = None):
    conn.execute(
        "UPDATE integration_task_steps SET status = ?, details = ? WHERE id = ?",
        (status, details, step_id)
    )
    conn.commit()

def update_task_status(conn: sqlite3.Connection, task_id: str, status: str, final_report: str = None):
    current_time = int(time.time() * 1000)
    conn.execute(
        "UPDATE integration_tasks SET status = ?, final_report = ?, updated_at = ? WHERE id = ?",
        (status, final_report, current_time, task_id)
    )
    conn.commit()

async def _execute_webhook(task_id: str, conn: sqlite3.Connection, config: Dict[str, Any], prompt: str):
    """
    The actual logic for the webhook integration.
    """
    step_index = 1
    
    # Step 1: Prepare data
    step1_id = add_task_step(conn, task_id, step_index, "Preparing data payload")
    payload = {
        "text": prompt,
        "source": "Nexus Integration"
    }
    update_step_status(conn, step1_id, "completed", json.dumps(payload, indent=2))
    step_index += 1

    # Step 2: Call Webhook
    webhook_url = config.get("webhookUrl")
    if not webhook_url:
        raise ValueError("Webhook URL is not configured for this integration.")
        
    step2_id = add_task_step(conn, task_id, step_index, f"Sending POST request to webhook")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(webhook_url, json=payload, timeout=30.0)
            response.raise_for_status()
            response_text = response.text
            update_step_status(conn, step2_id, "completed", f"Status: {response.status_code}\nResponse: {response_text[:500]}")
            return f"Successfully sent data to {config.get('name', 'webhook')}. Service responded with status {response.status_code}."
    except httpx.RequestError as e:
        error_details = f"Network error calling webhook: {e}"
        update_step_status(conn, step2_id, "failed", error_details)
        raise Exception(error_details)
    except httpx.HTTPStatusError as e:
        error_details = f"Webhook returned error: {e.response.status_code} - {e.response.text}"
        update_step_status(conn, step2_id, "failed", error_details)
        raise Exception(error_details)


def run_integration_task_background(task_id: str, integration_config: Dict[str, Any], user_prompt: str):
    """
    Entry point for running the integration task in a background thread.
    """
    logging.info(f"[{task_id}] Background integration task started.")
    conn = get_db_connection_for_bg()
    if not conn:
        logging.error(f"[{task_id}] FATAL: Could not get DB connection for background task.")
        return

    final_report = ""
    status = "running"
    
    try:
        service_id = integration_config.get("service")
        if service_id == "zapier_webhook_out":
            final_report = asyncio.run(_execute_webhook(task_id, conn, integration_config, user_prompt))
            status = "completed"
        else:
            raise ValueError(f"Unknown integration service type: {service_id}")

    except Exception as e:
        logging.error(f"[{task_id}] Integration task failed with an exception: {e}", exc_info=True)
        final_report = f"Task failed: {e}"
        status = "failed"
    finally:
        update_task_status(conn, task_id, status, final_report)
        if conn:
            conn.close()
        logging.info(f"[{task_id}] Background integration task finished with status: {status}")