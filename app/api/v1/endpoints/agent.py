# backend/app/api/v1/endpoints/agent.py
import json
import logging
import time
import os
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import sqlite3

from app.database import get_db_connection
from app.schemas.proxy_schemas import ApiConfig
from app.agents.runner import run_task_background
from app.agents.modes.research_mode import generate_node_content_background
from app.agents.modes.refine_mode import refine_section_background
from app.core.config import settings

router = APIRouter()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FILES_DIR = os.path.join(settings.NEXUS_DATA_PATH, "files")

# --- Models ---

class StartTaskRequest(BaseModel):
    goal: str
    api_config: Dict[str, Any]
    conversation_id: str
    mode: str = "plan"
    knowledge_base_selection: Optional[str] = None

class RestartTaskRequest(BaseModel):
    task_id: str

class ResumeWriteTaskRequest(BaseModel):
    task_id: str
    elaboration: Dict[str, Any]
    plan: List[Dict[str, Any]]

class GenerateNodeContentRequest(BaseModel):
    task_id: str
    node_id: str

class RefineSectionRequest(BaseModel):
    task_id: str
    node_id: str
    prompt: str
    model: str
    is_manual: bool = False

class LinkTaskRequest(BaseModel):
    message_id: str
    agent_task_id: str

class PlanStep(BaseModel):
    id: str
    task_id: str
    step_index: int
    thought: Optional[str] = None
    action: str
    action_input: str
    observation: Optional[str] = None
    status: str
    result: Optional[str] = None

class PlanOutlineStep(BaseModel):
    sub_goal: str
    steps: Optional[List['PlanOutlineStep']] = None

class TaskStatusResponse(BaseModel):
    id: str
    conversation_id: str = Field(alias="conversationId")
    user_goal: str = Field(alias="userGoal")
    status: str
    mode: str
    steps: List[PlanStep]
    plan: Optional[Any] = None
    final_report: Optional[str] = Field(None, alias="finalReport")
    created_at: int = Field(alias="createdAt")
    updated_at: Optional[int] = Field(None, alias="updatedAt")
    log_file_url: Optional[str] = Field(None, alias="logFileUrl")
    report_file_url: Optional[str] = Field(None, alias="reportFileUrl")
    research_content: Optional[Dict[str, Any]] = Field(None, alias="researchContent")

# --- API Endpoints ---

@router.post("/start-task", status_code=202)
async def start_task(payload: StartTaskRequest, background_tasks: BackgroundTasks):
    """
    Starts a new agent task in the background.
    """
    task_id = f"task-{time.time_ns()}"
    
    background_tasks.add_task(
        run_task_background, 
        task_id=task_id, 
        conversation_id=payload.conversation_id, 
        goal=payload.goal, 
        api_config_dict=payload.api_config,
        mode=payload.mode,
        knowledge_base_selection=payload.knowledge_base_selection,
        is_resume=False
    )
    return {"task_id": task_id}

@router.post("/restart-task", status_code=202)
async def restart_task(payload: RestartTaskRequest, background_tasks: BackgroundTasks):
    """
    Restarts a failed agent task from its last known state.
    """
    background_tasks.add_task(
        run_task_background,
        task_id=payload.task_id,
        conversation_id=None,
        goal=None,
        api_config_dict=None,
        mode=None,
        knowledge_base_selection=None,
        is_resume=True
    )
    return {"message": f"Restart signal sent to task {payload.task_id}."}

@router.post("/resume-write-task", status_code=202)
async def resume_write_task(payload: ResumeWriteTaskRequest, background_tasks: BackgroundTasks):
    """
    Resumes a 'write' mode task after the user has confirmed/edited the plan.
    """
    background_tasks.add_task(
        run_task_background,
        task_id=payload.task_id,
        conversation_id=None,
        goal=None,
        api_config_dict=None,
        mode=None,
        knowledge_base_selection=None,
        is_resume=True,
        resume_payload={"elaboration": payload.elaboration, "plan": payload.plan}
    )
    return {"message": f"Resume signal sent to write task {payload.task_id}."}


@router.post("/generate-node-content", status_code=202)
async def generate_node_content(payload: GenerateNodeContentRequest, background_tasks: BackgroundTasks):
    """
    Triggers the generation of content for a specific node in a research task.
    """
    background_tasks.add_task(
        generate_node_content_background,
        payload.task_id,
        payload.node_id
    )
    return {"message": f"Content generation for node {payload.node_id} has been queued."}

@router.post("/refine-section", status_code=202)
async def refine_section(payload: RefineSectionRequest, background_tasks: BackgroundTasks):
    """
    Triggers a refinement step for a specific section of a task.
    """
    background_tasks.add_task(
        refine_section_background,
        payload.task_id,
        payload.node_id,
        payload.prompt,
        payload.model,
        payload.is_manual
    )
    return {"message": f"Refinement for node {payload.node_id} has been queued."}

@router.post("/stop-task/{task_id}", status_code=200)
async def stop_task(task_id: str, conn: sqlite3.Connection = Depends(get_db_connection)):
    """
    Stops a running agent task by marking it as failed.
    """
    task_row = conn.execute("SELECT status FROM agent_tasks WHERE id = ?", (task_id,)).fetchone()
    if not task_row:
        raise HTTPException(status_code=404, detail="Task not found")

    current_status = task_row["status"]
    if current_status in ["running", "planning", "awaiting_user_input"]:
        current_time = int(time.time() * 1000)
        conn.execute(
            "UPDATE agent_tasks SET status = ?, final_report = ?, updated_at = ? WHERE id = ?",
            ("failed", "Task manually stopped by user.", current_time, task_id)
        )
        conn.commit()
        logging.info(f"Task {task_id} marked as 'failed' to signal stop.")
        return {"message": f"Stop signal sent to task {task_id}."}
    
    logging.warning(f"Attempted to stop task {task_id} which was not in a stoppable state (current state: {current_status}).")
    return {"message": f"Task {task_id} was not in a stoppable state (current state: {current_status})."}


@router.get("/get-task-status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str, request: Request, conn: sqlite3.Connection = Depends(get_db_connection)):
    """
    Retrieves the status of a specific agent task.
    """
    task_row = conn.execute("SELECT id, conversation_id, user_goal, status, mode, final_report, created_at, updated_at, plan, research_content FROM agent_tasks WHERE id = ?", (task_id,)).fetchone()
    if not task_row:
        raise HTTPException(status_code=404, detail="Task not found")

    steps_rows = conn.execute(
        "SELECT id, task_id, step_index, thought, action, action_input, observation, status, result FROM agent_task_steps WHERE task_id = ? ORDER BY step_index ASC",
        (task_id,)
    ).fetchall()
    
    steps = [PlanStep(**dict(row)) for row in steps_rows]
    
    task_data = dict(task_row)
    for key in ["plan", "research_content"]:
        json_str = task_data.get(key)
        if json_str:
            try:
                task_data[key] = json.loads(json_str)
            except (json.JSONDecodeError, TypeError):
                task_data[key] = None

    base_url = f"{request.url.scheme}://{request.url.netloc}"

    log_file_name = f"{task_id}_log.md"
    report_file_name = f"{task_id}_report.md"
    
    task_data["logFileUrl"] = None
    if os.path.exists(os.path.join(FILES_DIR, log_file_name)):
        task_data["logFileUrl"] = f"{base_url}/files/{log_file_name}"

    task_data["reportFileUrl"] = None
    if os.path.exists(os.path.join(FILES_DIR, report_file_name)):
        task_data["reportFileUrl"] = f"{base_url}/files/{report_file_name}"

    task_data["conversationId"] = task_data.pop("conversation_id")
    task_data["userGoal"] = task_data.pop("user_goal")
    task_data["finalReport"] = task_data.pop("final_report")
    task_data["createdAt"] = task_data.pop("created_at")
    task_data["updatedAt"] = task_data.pop("updated_at")

    return TaskStatusResponse(**task_data, steps=steps)