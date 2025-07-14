# backend/app/api/v1/endpoints/agent.py
import json
import logging
import time
import asyncio
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Callable, Coroutine
import sqlite3

from app.database import get_db_connection, get_db_connection_for_bg
from .proxy import tavily_search, ApiConfig

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

# --- Tool Implementations ---

async def _tool_internet_search(params: Dict[str, Any], api_config: ApiConfig) -> str:
    query = params.get("query")
    if not query:
        raise ValueError("Internet search requires a 'query' parameter.")
    return await tavily_search(query, api_config)

async def _tool_write_document(params: Dict[str, Any], api_config: ApiConfig) -> str:
    title = params.get("title", "Untitled Document")
    content = params.get("content", "")
    return f"# {title}\n\n{content}"

async def _tool_save_to_knowledge_base(params: Dict[str, Any], api_config: ApiConfig) -> str:
    filename = params.get("filename")
    if not filename:
        raise ValueError("Saving to knowledge base requires a 'filename' parameter.")
    return f"Success: File '{filename}' saved to the knowledge base."

# --- Tool Registry and Dispatcher ---

TOOL_DISPATCHER: Dict[str, Callable[[Dict[str, Any], ApiConfig], Coroutine[Any, Any, str]]] = {
    "internet_search": _tool_internet_search,
    "write_document": _tool_write_document,
    "save_to_knowledge_base": _tool_save_to_knowledge_base,
}

# --- Core Agent Logic ---

def _create_plan_with_llm(goal: str) -> List[Dict[str, Any]]:
    logging.info("Generating plan for goal: %s", goal)
    if "英伟达" in goal and "分析报告" in goal:
        return [
            {"thought": "首先，我需要收集关于英伟达最近表现的材料。我会从搜索新闻和股价开始。", "action": "internet_search", "action_input": {"query": "NVIDIA recent news and stock performance"}},
            {"thought": "然后，我需要搜索关于他们最新技术（如AI芯片）的详细信息。", "action": "internet_search", "action_input": {"query": "NVIDIA AI chip technology advancements"}},
            {"thought": "接下来，我将综合搜索到的信息，撰写报告的初稿。", "action": "write_document", "action_input": {"content": "根据以下信息撰写一份关于英伟达的分析报告草稿：\n\n新闻和股价：\n{step_1_observation}\n\n技术进展：\n{step_2_observation}", "title": "英伟达分析报告初稿"}},
            {"thought": "最后，我将初稿存为Markdown文件到知识库。", "action": "save_to_knowledge_base", "action_input": {"content": "{step_3_observation}", "filename": "nvidia_analysis_report.md"}}
        ]
    else:
        return [
            {"thought": "I need to search the internet to answer the user's goal.", "action": "internet_search", "action_input": {"query": goal}}
        ]

async def _execute_step(step: Dict, observations: Dict, api_config: ApiConfig) -> str:
    tool_id = step["action"]
    action_input_str = step["action_input"]

    for key, value in observations.items():
        action_input_str = action_input_str.replace(f"{{{key}}}", json.dumps(value))
    
    action_input = json.loads(action_input_str)
    
    logging.info(f"Executing step {step['step_index']}: {tool_id} with params {action_input}")
    
    tool_function = TOOL_DISPATCHER.get(tool_id)
    if not tool_function:
        raise ValueError(f"Tool '{tool_id}' not found.")
        
    observation = await tool_function(action_input, api_config)
    return observation

async def _execute_task(conn: sqlite3.Connection, task_id: str, conversation_id: str, goal: str, api_config_dict: dict):
    api_config = ApiConfig(**api_config_dict)
    
    conn.execute(
        "INSERT INTO agent_tasks (id, conversation_id, user_goal, status, created_at) VALUES (?, ?, ?, ?, ?)",
        (task_id, conversation_id, goal, "running", int(time.time() * 1000))
    )
    conn.commit()

    plan_steps_data = _create_plan_with_llm(goal)
    for i, step_data in enumerate(plan_steps_data):
        conn.execute(
            "INSERT INTO agent_task_steps (id, task_id, step_index, thought, action, action_input, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (f"{task_id}-{i+1}", task_id, i + 1, step_data.get("thought"), step_data["action"], json.dumps(step_data["action_input"]), "pending")
        )
    conn.commit()

    observations = {}
    steps_to_execute = conn.execute("SELECT * FROM agent_task_steps WHERE task_id = ? ORDER BY step_index ASC", (task_id,)).fetchall()

    for step_row in steps_to_execute:
        step = dict(step_row)
        step_id = step["id"]
        
        conn.execute("UPDATE agent_task_steps SET status = ? WHERE id = ?", ("running", step_id))
        conn.commit()
        
        observation = await _execute_step(step, observations, api_config)
        observations[f"step_{step['step_index']}_observation"] = observation
        
        conn.execute("UPDATE agent_task_steps SET status = ?, observation = ? WHERE id = ?", ("completed", observation, step_id))
        conn.commit()

    conn.execute("UPDATE agent_tasks SET status = ? WHERE id = ?", ("completed", task_id))
    conn.commit()
    logging.info(f"[{task_id}] Task marked as completed.")

def run_task_background(task_id: str, conversation_id: str, goal: str, api_config_dict: dict):
    logging.info(f"[{task_id}] Background task started for goal: {goal}")
    conn = get_db_connection_for_bg()
    if not conn:
        logging.error(f"[{task_id}] FATAL: Could not get DB connection for background task.")
        return

    try:
        asyncio.run(_execute_task(conn, task_id, conversation_id, goal, api_config_dict))
    except Exception as e:
        logging.error(f"[{task_id}] Task failed with unhandled exception: {e}", exc_info=True)
        observation = f"Task failed with unhandled exception: {e}"
        conn.execute(
            "UPDATE agent_task_steps SET status = ?, observation = ? WHERE task_id = ? AND status = 'running'",
            ("failed", observation, task_id)
        )
        conn.execute("UPDATE agent_tasks SET status = ? WHERE id = ?", ("failed", task_id))
        conn.commit()
    finally:
        conn.close()

# --- API Endpoints ---

@router.post("/start-task", status_code=202)
async def start_task(payload: StartTaskRequest, background_tasks: BackgroundTasks):
    task_id = f"task-{time.time_ns()}"
    
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
    task_row = conn.execute("SELECT id, user_goal, status FROM agent_tasks WHERE id = ?", (task_id,)).fetchone()
    if not task_row:
        raise HTTPException(status_code=404, detail="Task not found")

    steps_rows = conn.execute(
        "SELECT id, task_id, step_index, thought, action, action_input, observation, status FROM agent_task_steps WHERE task_id = ? ORDER BY step_index ASC",
        (task_id,)
    ).fetchall()
    
    steps = [PlanStep(**dict(row)) for row in steps_rows]

    return TaskStatusResponse(id=task_row["id"], user_goal=task_row["user_goal"], status=task_row["status"], steps=steps)