# backend/app/agents/runner.py
import json
import logging
import time
import asyncio
import sqlite3
from typing import List, Dict, Any, Callable, Coroutine

from app.database import get_db_connection_for_bg
from app.api.v1.endpoints.proxy import tavily_search, ApiConfig, get_completion

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Tool Implementations ---

async def _tool_internet_search(params: Dict[str, Any], api_config: ApiConfig) -> str:
    """Searches the internet for a given query."""
    query = params.get("query")
    if not query:
        raise ValueError("Internet search requires a 'query' parameter.")
    return await tavily_search(query, api_config)

kb_path = "knowledge_base"

async def _tool_save_to_knowledge_base(params: Dict[str, Any], api_config: ApiConfig) -> str:
    """Saves a string of content to a file in the knowledge base."""
    filename = params.get("filename")
    content = params.get("content", "")
    if not filename:
        raise ValueError("Saving to knowledge base requires a 'filename' parameter.")
    import os
    if not os.path.exists(kb_path):
        os.makedirs(kb_path)
    file_path = os.path.join(kb_path, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Success: File '{filename}' saved to the knowledge base at '{file_path}'."

async def _tool_python_code_interpreter(params: Dict[str, Any], api_config: ApiConfig) -> str:
    """Executes a string of Python code and returns the output."""
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

async def _tool_generate_report_outline(params: Dict[str, Any], api_config: ApiConfig) -> str:
    """Generates a structured outline for a report based on a goal."""
    goal = params.get("goal")
    if not goal:
        raise ValueError("Generating an outline requires a 'goal' parameter.")

    prompt = f"Based on the following goal, please generate a detailed report outline in JSON format. The outline should be a list of strings, where each string is a chapter title. Goal: {goal}"
    messages = [{"role": "user", "content": prompt}]

    outline_str = await get_completion(messages, api_config)

    # We expect the LLM to return a JSON list of strings.
    try:
        outline = json.loads(outline_str)
        if isinstance(outline, list) and all(isinstance(i, str) for i in outline):
            return json.dumps(outline)
        else:
            raise ValueError("LLM did not return a valid JSON list of strings for the outline.")
    except json.JSONDecodeError:
        # If the LLM fails to produce valid JSON, return the raw string as an error observation.
        return f"Error: Failed to generate a valid JSON outline. Raw LLM output: {outline_str}"

async def _tool_write_report_chapter(params: Dict[str, Any], api_config: ApiConfig) -> str:
    """Writes a single chapter of a report, given the goal, outline, and chapter title."""
    goal = params.get("goal")
    outline = params.get("outline")
    chapter_title = params.get("chapter_title")
    previous_chapters = params.get("previous_chapters", "")

    if not all([goal, outline, chapter_title]):
        raise ValueError("Writing a chapter requires 'goal', 'outline', and 'chapter_title' parameters.")

    prompt = f"""
    You are a chapter writer for a larger report.
    Overall Goal: {goal}
    Report Outline: {outline}

    You are now writing the chapter titled: "{chapter_title}".

    Previously written chapters (for context):
    {previous_chapters}

    Please write the content for the "{chapter_title}" chapter now. Ensure it flows logically from the previous content and fits within the overall outline.
    """
    messages = [{"role": "user", "content": prompt}]

    chapter_content = await get_completion(messages, api_config)
    return chapter_content


TOOL_DISPATCHER: Dict[str, Callable[[Dict[str, Any], ApiConfig], Coroutine[Any, Any, str]]] = {
    "internet_search": _tool_internet_search,
    "python_code_interpreter": _tool_python_code_interpreter,
    "generate_report_outline": _tool_generate_report_outline,
    "write_report_chapter": _tool_write_report_chapter,
    "save_to_knowledge_base": _tool_save_to_knowledge_base,
    # "write_document" is now superseded by the more advanced report generation tools
}

# --- New Agent Core Logic ---

async def _determine_next_step_with_llm(goal: str, history: List[Dict], api_config: ApiConfig) -> Dict:
    """
    Determines the next step for the agent to take by consulting the LLM.
    This function is the "brain" of the agent.
    """
    logging.info("Determining next step for goal: %s", goal)

    # Construct the prompt
    prompt = _build_agent_prompt(goal, history)

    messages = [{"role": "user", "content": prompt}]

    # Get response from LLM
    llm_response_str = await get_completion(messages, api_config)

    # Extract the JSON part of the response
    try:
        # The LLM might sometimes add explanatory text before or after the JSON block.
        # We need to find the JSON block to parse it reliably.
        json_start = llm_response_str.find("```json")
        json_end = llm_response_str.rfind("```")

        if json_start != -1:
            # Adjust start position to be after "```json\n"
            json_str = llm_response_str[json_start + 7:json_end].strip()
        else:
            # If no JSON block is found, assume the whole response is the JSON
            json_str = llm_response_str

        next_step = json.loads(json_str)
        logging.info(f"LLM decided next step: {next_step}")
        return next_step
    except (json.JSONDecodeError, IndexError) as e:
        logging.error(f"Failed to decode LLM response into JSON. Response: {llm_response_str}. Error: {e}")
        # Fallback action if parsing fails
        return {
            "thought": "I received an invalid response from the LLM. I will try to recover or ask for clarification.",
            "action": "FINISH",
            "action_input": {"final_answer": f"Error: The agent's internal thought process failed. Last response was: {llm_response_str}"}
        }

def _build_agent_prompt(goal: str, history: List[Dict]) -> str:
    """Builds the full prompt for the agent's decision-making LLM call."""

    # Serialize the history into a readable format
    history_str = "\n".join([
        f"Step {item['step']}:\nThought: {item['thought']}\nAction: {item['action']}({json.dumps(item['action_input'])})\nObservation: {item['observation']}"
        for item in history
    ]) if history else "No steps taken yet."

    # Get the list of available tools
    tools_str = ""
    for name, func in TOOL_DISPATCHER.items():
        # A simple way to get parameter names from the function signature
        import inspect
        sig = inspect.signature(func)
        params = [p for p in sig.parameters if p != 'api_config']
        tools_str += f'- "{name}": {func.__doc__ or "No description available."} Parameters: {params}\n'


    return f"""
You are an autonomous agent responsible for achieving a specific goal.

**Goal:**
{goal}

**Available Tools:**
You have access to the following tools. Only use these tools.
{tools_str}

**History of Actions:**
Here is the history of the actions you have taken so far, and the observations you have gathered.
{history_str}

**Your Task:**
Based on the goal and the history, decide what to do next. You have two options:
1.  **Use a tool:** If you need more information or need to perform an action, choose one of the available tools.
2.  **Finish the task:** If you have achieved the goal and have a complete answer, use the special action "FINISH".

**Output Format:**
You MUST provide your response in a single JSON object, enclosed in a ```json ... ``` block.
The JSON object must have the following keys:
- "thought": A string explaining your reasoning and plan for the next step.
- "action": A string with the name of the tool you want to use (e.g., "internet_search") or "FINISH".
- "action_input": An object containing the parameters for the chosen action. For the "FINISH" action, this should be `{{ "final_answer": "Your complete, final answer here." }}`.

**Example:**
```json
{{
    "thought": "I need to find out the latest news about the goal. I will use the internet search tool for this.",
    "action": "internet_search",
    "action_input": {{
        "query": "latest news on {goal}"
    }}
}}
```

Now, based on the current goal and history, what is your next step?
"""

async def _execute_step(action: str, action_input: Dict, api_config: ApiConfig) -> str:
    tool_function = TOOL_DISPATCHER.get(action)
    if not tool_function:
        # This is a critical error in the agent's logic, it shouldn't happen if the LLM
        # is prompted correctly. We return an error observation.
        return f"Error: Tool '{action}' not found. Please choose from the available tools."

    logging.info(f"Executing tool '{action}' with input {action_input}")
    try:
        observation = await tool_function(action_input, api_config)
        return str(observation)
    except ValueError as e:
        # This error is often due to bad input from the LLM.
        # We return the error message so the agent can try to fix it.
        logging.warning(f"Tool '{action}' raised a ValueError: {e}")
        return f"Error executing tool '{action}': {e}. Please check your action_input."
    except Exception as e:
        # For other exceptions, we log it as a more severe error and return a generic message.
        # This might indicate a bug in the tool itself or an unexpected external issue.
        logging.error(f"An unexpected error occurred in tool '{action}': {e}", exc_info=True)
        return f"Error: An unexpected error occurred while executing tool '{action}'."

async def _execute_task(conn: sqlite3.Connection, task_id: str, conversation_id: str, goal: str, api_config_dict: dict):
    api_config = ApiConfig(**api_config_dict)

    # 1. Initialize Task in DB
    current_time = int(time.time() * 1000)
    conn.execute(
        "INSERT INTO agent_tasks (id, conversation_id, user_goal, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (task_id, conversation_id, goal, "running", current_time, current_time)
    )
    conn.commit()

    step_index = 1
    history = []

    while True:
        # 2. Determine Next Step
        next_step_decision = await _determine_next_step_with_llm(goal, history, api_config)
        thought = next_step_decision.get("thought")
        action = next_step_decision.get("action")
        action_input = next_step_decision.get("action_input", {})

        if not action:
            logging.error(f"[{task_id}] LLM failed to provide an action. Ending task.")
            break

        # 3. Check for Finish Condition
        if action.upper() == "FINISH":
            final_answer = action_input.get("final_answer", "No final answer provided.")
            logging.info(f"[{task_id}] FINISH action received. Final answer: {final_answer}")

            # Save final report to DB
            current_time = int(time.time() * 1000)
            conn.execute(
                "UPDATE agent_tasks SET status = ?, final_report = ?, updated_at = ? WHERE id = ?",
                ("completed", final_answer, current_time, task_id)
            )
            conn.commit()
            break

        # 4. Execute Action
        step_id = f"{task_id}-{step_index}"
        conn.execute(
            "INSERT INTO agent_task_steps (id, task_id, step_index, thought, action, action_input, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (step_id, task_id, step_index, thought, action, json.dumps(action_input), "running")
        )
        conn.commit()

        observation = await _execute_step(action, action_input, api_config)

        # 5. Save Observation
        conn.execute(
            "UPDATE agent_task_steps SET status = ?, observation = ? WHERE id = ?",
            ("completed", observation, step_id)
        )
        current_time = int(time.time() * 1000)
        conn.execute("UPDATE agent_tasks SET status = ?, updated_at = ? WHERE id = ?", ("running", current_time, task_id))
        conn.commit()

        # 6. Update History
        history.append({
            "step": step_index,
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "observation": observation
        })
        step_index += 1

    logging.info(f"[{task_id}] Task execution loop finished.")


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
        current_time = int(time.time() * 1000)
        conn.execute(
            "UPDATE agent_tasks SET status = ?, updated_at = ? WHERE id = ?",
            ("failed", current_time, task_id)
        )
        conn.commit()
    finally:
        if conn:
            conn.close()
