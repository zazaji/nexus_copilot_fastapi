# backend/app/api/v1/endpoints/agent.py
import logging
import time
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List
import sqlite3

from app.database import get_db_connection
from .proxy import ApiConfig
from app.agents.runner import run_task_background

router = APIRouter()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Models ---

class StartTaskRequest(BaseModel):
    goal: str
    api_config: ApiConfig
    conversation_id: str

class PlanStep(BaseModel):
    id: str
    task_id: str
    step_index: int
    thought: str | None
    action: str
    action_input: str
    observation: str | None
    status: str

class TaskStatusResponse(BaseModel):
    id: str
    user_goal: str
    status: str
    steps: List[PlanStep]
    final_report: str | None
    updated_at: int | None

# --- API Endpoints ---

@router.post("/start-task", status_code=202)
async def start_task(payload: StartTaskRequest, background_tasks: BackgroundTasks):
    """
    Starts a new agent task in the background.
    """
    task_id = f"task-{time.time_ns()}"
    
    # The actual agent logic is now in app.agents.runner
    background_tasks.add_task(
        run_task_background, 
        task_id, 
        payload.conversation_id, 
        payload.goal, 
        payload.api_config.model_dump()
    )

    return {"task_id": task_id}

@router.get("/get-task-status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str, conn: sqlite3.Connection = Depends(get_db_connection)):
    """
    Retrieves the status of a specific agent task.
    """
    task_row = conn.execute("SELECT id, user_goal, status, final_report, updated_at FROM agent_tasks WHERE id = ?", (task_id,)).fetchone()
    if not task_row:
        raise HTTPException(status_code=404, detail="Task not found")

    steps_rows = conn.execute(
        "SELECT id, task_id, step_index, thought, action, action_input, observation, status FROM agent_task_steps WHERE task_id = ? ORDER BY step_index ASC",
        (task_id,)
    ).fetchall()
    
    steps = [PlanStep(**dict(row)) for row in steps_rows]

    return TaskStatusResponse(**dict(task_row), steps=steps)