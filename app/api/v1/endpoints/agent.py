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
from .proxy import tavily_search, ApiConfig, get_completion

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

# Define the knowledge base path at the module level so it can be patched for tests
kb_path = "knowledge_base"

async def _tool_save_to_knowledge_base(params: Dict[str, Any], api_config: ApiConfig) -> str:
    filename = params.get("filename")
    content = params.get("content", "")

    if not filename:
        raise ValueError("Saving to knowledge base requires a 'filename' parameter.")

    # 在实际应用中，您会希望将文件保存在一个更持久、可配置的位置
    # For demonstration, we'll save it in a local 'knowledge_base' directory
    import os
    if not os.path.exists(kb_path):
        os.makedirs(kb_path)

    file_path = os.path.join(kb_path, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"Success: File '{filename}' saved to the knowledge base at '{file_path}'."

async def _tool_python_code_interpreter(params: Dict[str, Any], api_config: ApiConfig) -> str:
    code = params.get("code")
    if not code:
        raise ValueError("Python code interpreter requires 'code' parameter.")

    from io import StringIO
    import sys

    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()

    try:
        exec(code, globals())
        sys.stdout = old_stdout
        output = redirected_output.getvalue()
        return output
    except Exception as e:
        sys.stdout = old_stdout
        return f"Error executing code: {e}"

# --- Tool Registry and Dispatcher ---

TOOL_DISPATCHER: Dict[str, Callable[[Dict[str, Any], ApiConfig], Coroutine[Any, Any, str]]] = {
    "internet_search": _tool_internet_search,
    "write_document": _tool_write_document,
    "save_to_knowledge_base": _tool_save_to_knowledge_base,
    "python_code_interpreter": _tool_python_code_interpreter,
}

# --- Core Agent Logic ---

async def _create_plan_with_llm(goal: str, api_config: ApiConfig) -> List[Dict[str, Any]]:
    logging.info("Generating plan for goal: %s", goal)

    prompt = f"""
    As a smart task planning assistant, your role is to break down complex user goals into a series of simple, actionable steps.
    For the user's goal: "{goal}", please generate a JSON-formatted plan.
    Each step in the plan should be an object with the following keys: "thought", "action", and "action_input".
    - "thought": Briefly describe your thinking process for this step.
    - "action": The name of the tool to use for this step.
    - "action_input": A JSON object containing the parameters for the action.

    Available tools:
    - "internet_search": Searches online for the given query. Parameters: {{"query": "search query"}}
    - "python_code_interpreter": Executes Python code. Parameters: {{"code": "python code to execute"}}
    - "write_document": Writes content to a document. Parameters: {{"title": "document title", "content": "document content"}}
    - "save_to_knowledge_base": Saves a document to the knowledge base. Parameters: {{"filename": "file name", "content": "file content"}}

    Important Notes:
    - For "action_input", ensure it is a valid JSON object. For example, use {{"query": "user's query"}} instead of just a string.
    - When a step depends on the output of a previous step, use placeholders like {{{{step_1_observation}}}} in the "action_input". The system will automatically substitute these placeholders with the actual observations from the corresponding steps.
    - The final step should usually be "save_to_knowledge_base" to store the complete report.

    Example Plan:
    If the goal is "Research NVIDIA's latest stock price and write a brief report.", the plan could be:
    [
        {{
            "thought": "I need to find NVIDIA's current stock price.",
            "action": "internet_search",
            "action_input": {{"query": "NVIDIA stock price"}}
        }},
        {{
            "thought": "Now I will write a brief report with the stock price I found.",
            "action": "write_document",
            "action_input": {{"title": "NVIDIA Stock Report", "content": "NVIDIA's current stock price is {{{{step_1_observation}}}}."}}
        }},
        {{
            "thought": "Finally, I will save the report to the knowledge base.",
            "action": "save_to_knowledge_base",
            "action_input": {{"filename": "nvidia_stock_report.md", "content": "{{{{step_2_observation}}}}"}}
        }}
    ]

    Now, please generate the plan for the goal: "{goal}"
    """

    messages = [{"role": "user", "content": prompt}]
    llm_response = await get_completion(messages, api_config)

    try:
        plan = json.loads(llm_response)
        logging.info(f"Generated plan: {plan}")
        return plan
    except json.JSONDecodeError:
        logging.error(f"Failed to decode LLM response into JSON: {llm_response}")
        raise ValueError("Failed to generate a valid plan from LLM.")

async def _execute_step(step: Dict, observations: Dict, api_config: ApiConfig) -> str:
    tool_id = step["action"]
    action_input_str = step["action_input"]

    for key, value in observations.items():
        # JSON-encode the value to handle special characters, then strip the outer quotes
        # to embed it as a raw string value in the larger JSON structure.
        if value is not None:
            encoded_value = json.dumps(str(value))[1:-1]
        else:
            encoded_value = ""
        action_input_str = action_input_str.replace(f"{{{key}}}", encoded_value)

    # Now, the string should be a valid JSON object
    try:
        action_input = json.loads(action_input_str)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode action_input: {action_input_str}. Error: {e}")
        raise ValueError(f"Invalid JSON in action_input after substitution: {action_input_str}")

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

    plan_steps_data = await _create_plan_with_llm(goal, api_config)
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

    # --- Final Report Generation ---
    logging.info(f"[{task_id}] All steps completed. Generating final report.")

    observations_summary = "\n".join(
        f"Step {i+1} ({step['action']}):\n{obs}\n"
        for i, (step, obs) in enumerate(zip(steps_to_execute, observations.values()))
    )

    report_prompt = f"""
    The user's original goal was: "{goal}"

    Based on the following observations gathered from the executed steps, please synthesize a comprehensive and well-structured final report in Markdown format.
    The report should directly address the user's goal.

    Observations:
    {observations_summary}

    Please now generate the final report.
    """

    report_messages = [{"role": "user", "content": report_prompt}]
    final_report = await get_completion(report_messages, api_config)

    # Sanitize the goal to create a valid filename
    sanitized_goal = "".join(c for c in goal if c.isalnum() or c in (' ', '_')).rstrip()
    filename = f"{sanitized_goal.replace(' ', '_')}_report.md"

    await _tool_save_to_knowledge_base({"filename": filename, "content": final_report}, api_config)
    logging.info(f"[{task_id}] Final report saved as '{filename}'.")

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