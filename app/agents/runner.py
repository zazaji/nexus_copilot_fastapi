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

# --- Task Context ---

class TaskContext:
    """A class to hold the state of a task."""
    def __init__(self, task_id: str, conversation_id: str, goal: str, api_config: ApiConfig, conn: sqlite3.Connection):
        self.task_id = task_id
        self.conversation_id = conversation_id
        self.goal = goal
        self.api_config = api_config
        self.conn = conn
        self.plan: List[Dict] = []
        self.previous_steps_summary = ""
        self.final_report_content = ""
        self.max_replans = 2
        self.replan_attempts = 0

# --- Context Management ---

async def _summarize_history(history: List[Dict], api_config: ApiConfig) -> str:
    """Summarizes a history of actions and observations to save context space."""
    if not history:
        return "No history yet."

    if len(json.dumps(history)) < 2000: # If history is short, just return it
        return json.dumps(history)

    # For longer histories, use an LLM to summarize
    prompt = f"Please summarize the following history of actions and observations into a concise summary:\n\n{json.dumps(history, indent=2)}"
    messages = [{"role": "user", "content": prompt}]
    summary = await get_completion(messages, api_config)
    return summary

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

async def _tool_read_from_knowledge_base(params: Dict[str, Any], api_config: ApiConfig) -> str:
    """Reads content from a file in the knowledge base."""
    filename = params.get("filename")
    if not filename:
        raise ValueError("Reading from knowledge base requires a 'filename' parameter.")
    import os
    file_path = os.path.join(kb_path, filename)
    if not os.path.exists(file_path):
        return f"Error: File '{filename}' not found in the knowledge base."
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

async def _tool_python_code_interpreter(params: Dict[str, Any], api_config: ApiConfig) -> str:
    """
    Executes a string of Python code in a restricted environment and returns the output.
    Can handle multi-line scripts.
    """
    code = params.get("code")
    if not code:
        raise ValueError("Python code interpreter requires 'code' parameter.")

    from io import StringIO
    import sys

    # Create a restricted global environment for exec
    restricted_globals = {
        "__builtins__": {
            "print": print,
            "len": len,
            "range": range,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "set": set,
            "sorted": sorted,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "__import__": __import__
        },
        # You can add safe libraries here, e.g., 'math', 'datetime'
        'json': json,
        'time': time,
    }

    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()

    try:
        # Execute the code with the restricted environment
        exec(code, restricted_globals)
        sys.stdout = old_stdout
        output = redirected_output.getvalue()
        return output if output else "Code executed successfully with no output."
    except Exception as e:
        sys.stdout = old_stdout
        # Provide a more informative error message
        return f"Error executing code: {type(e).__name__}: {e}"


async def _tool_generate_report_outline(params: Dict[str, Any], api_config: ApiConfig) -> str:
    """Generates a structured outline for a report based on a goal."""
    goal = params.get("goal")
    if not goal:
        raise ValueError("Generating an outline requires a 'goal' parameter.")

    prompt = f"Based on the following goal, please generate a detailed report outline in JSON format. The outline should be a list of strings, where each string is a chapter title. Goal: {goal}"
    messages = [{"role": "user", "content": prompt}]

    outline_str = await get_completion(messages, api_config)

    try:
        # The LLM might return the JSON wrapped in markdown, so we extract it.
        json_start = outline_str.find('[')
        json_end = outline_str.rfind(']') + 1
        if json_start != -1 and json_end != 0:
            outline_str = outline_str[json_start:json_end]

        outline = json.loads(outline_str)
        if isinstance(outline, list) and all(isinstance(i, str) for i in outline):
            return json.dumps(outline)
        else:
            # Attempt to re-prompt or fix the format if necessary
            return f"Error: LLM returned a valid JSON but not a list of strings. Outline: {outline}"
    except json.JSONDecodeError:
        return f"Error: Failed to generate a valid JSON outline. Raw LLM output: {outline_str}"

async def _tool_write_report_chapter(params: Dict[str, Any], api_config: ApiConfig) -> str:
    """Writes a single chapter of a report, given the goal, outline, and chapter title."""
    goal = params.get("goal")
    outline = params.get("outline")
    chapter_title = params.get("chapter_title")
    previous_chapters = params.get("previous_chapters", "") # This provides context of what has been written

    if not all([goal, outline, chapter_title]):
        raise ValueError("Writing a chapter requires 'goal', 'outline', and 'chapter_title' parameters.")

    prompt = f"""
    You are a chapter writer for a larger report.
    Overall Goal: {goal}
    Report Outline: {outline}
    You are now writing the chapter titled: "{chapter_title}".
    Previously written chapters (for context, do not repeat):
    {previous_chapters}
    Please write the content for the "{chapter_title}" chapter now. Ensure it flows logically from the previous content and fits within the overall outline.
    """
    messages = [{"role": "user", "content": prompt}]
    chapter_content = await get_completion(messages, api_config)
    return chapter_content

async def _tool_finish_task(params: Dict[str, Any], api_config: ApiConfig, context: TaskContext) -> str:
    """
    Use this tool to signify that the entire task is complete.
    The 'final_answer' parameter should contain the complete and final response to the user's goal.
    If the 'final_answer' parameter is not provided, the accumulated report content will be used.
    """
    final_answer = params.get("final_answer")
    if not final_answer:
        return context.final_report_content
    return final_answer

# --- Dynamic Re-planning Tool ---
async def _tool_replan(context: TaskContext, failed_step_summary: str) -> List[Dict]:
    """
    When the agent is stuck or a step fails, this tool is called to generate a new plan.
    """
    logging.info("Re-planning task...")

    replan_prompt = f"""
    You are a master planner. The original goal was: "{context.goal}".
    The execution so far has resulted in the following summary:
    {context.previous_steps_summary}

    The last step failed with the following summary:
    {failed_step_summary}

    However, the agent is stuck. Please create a new, revised plan to achieve the original goal, taking into account what has already been done and the recent failure.
    The new plan should start from the current state and lead to the completion of the goal.
    Provide the new plan in the same JSON format as before.
    """

    messages = [{"role": "user", "content": replan_prompt}]
    llm_response = await _call_llm_with_retry(messages, context.api_config)
    new_plan = llm_response.get("plan", [])
    if not isinstance(new_plan, list) or not all("sub_goal" in step for step in new_plan):
        raise ValueError("Re-planner LLM did not return a valid plan structure.")

    logging.info(f"Generated new plan with {len(new_plan)} steps.")
    return new_plan


TOOL_DISPATCHER: Dict[str, Callable[..., Coroutine[Any, Any, str]]] = {
    "internet_search": _tool_internet_search,
    "python_code_interpreter": _tool_python_code_interpreter,
    "generate_report_outline": _tool_generate_report_outline,
    "write_report_chapter": _tool_write_report_chapter,
    "save_to_knowledge_base": _tool_save_to_knowledge_base,
    "read_from_knowledge_base": _tool_read_from_knowledge_base,
    "finish_task": _tool_finish_task,
}

# --- LLM Interaction ---

async def _call_llm_with_retry(messages: List[Dict], api_config: ApiConfig, max_retries=3) -> Dict:
    """Calls the LLM and retries on JSON parsing errors."""
    for attempt in range(max_retries):
        try:
            llm_response_str = await get_completion(messages, api_config)

            # More robust JSON extraction
            json_start = llm_response_str.find("```json")
            if json_start != -1:
                json_str = llm_response_str[json_start + 7:]
                json_end = json_str.rfind("```")
                if json_end != -1:
                    json_str = json_str[:json_end].strip()
            else:
                json_str = llm_response_str.strip()

            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.warning(f"Attempt {attempt + 1} failed: JSONDecodeError. Response: {llm_response_str}. Retrying...")
            if attempt + 1 == max_retries:
                raise ValueError(f"LLM failed to return valid JSON after {max_retries} attempts. Last response: {llm_response_str}") from e
        except Exception as e:
            logging.error(f"An unexpected error occurred during LLM call: {e}")
            raise e
    raise ValueError("LLM call failed after multiple retries.")

# --- Planner Logic ---

def _build_planner_prompt(goal: str) -> str:
    """Builds the prompt for the Planner LLM call."""
    tools_str = ""
    for name, func in TOOL_DISPATCHER.items():
        import inspect
        sig = inspect.signature(func)
        params = [p for p in sig.parameters if p not in ['api_config', 'context']]
        tools_str += f'- "{name}": {func.__doc__ or "No description available."} Parameters: {params}\n'

    return f"""
You are a master planner AI. Your job is to create a detailed, step-by-step plan to achieve a user's goal.
The plan should be a sequence of sub-goals that can be executed by another AI agent. Break down the goal into small, manageable steps.

**User's Goal:**
{goal}

**Available Tools for the Executor Agent:**
The agent that executes your plan will have access to these tools:
{tools_str}

**Your Task:**
Generate a JSON object that represents the plan. The JSON object should have a "plan" key, which is a list of steps.
Each step in the list should be an object with a single key: "sub_goal".
The "sub_goal" should be a clear, concise instruction for the executor agent.
When writing a report, you should first generate an outline, then write each chapter, and finally combine them.
The final step should always be to call the `finish_task` tool.

**Example:**
For the goal "Write a report on the financial performance of Tesla in 2023", a good plan would be:
```json
{{
    "plan": [
        {{
            "sub_goal": "Search the internet for Tesla's 2023 annual financial reports and Q4 earnings call transcripts."
        }},
        {{
            "sub_goal": "Analyze the gathered financial data to identify key metrics like revenue, net income, and profit margins. Use the python interpreter for calculations if needed."
        }},
        {{
            "sub_goal": "Generate a report outline with sections for Introduction, Financial Performance, Key Challenges, and Future Outlook."
        }},
        {{
            "sub_goal": "Write the 'Introduction' chapter, summarizing the report's purpose."
        }},
        {{
            "sub_goal": "Write the 'Financial Performance' chapter, detailing the key metrics."
        }},
        {{
            "sub_goal": "Write the 'Key Challenges' chapter, discussing any obstacles Tesla faced."
        }},
        {{
            "sub_goal": "Write the 'Future Outlook' chapter, providing a forward-looking analysis."
        }},
        {{
            "sub_goal": "Combine all written chapters into a single final report and call the finish_task tool."
        }}
    ]
}}
```

Now, create a plan for the user's goal.
"""

async def _generate_initial_plan(context: TaskContext) -> List[Dict]:
    """Uses the LLM to generate the initial high-level plan."""
    logging.info("Generating initial plan for goal: %s", context.goal)
    prompt = _build_planner_prompt(context.goal)
    messages = [{"role": "user", "content": prompt}]

    try:
        llm_response = await _call_llm_with_retry(messages, context.api_config)
        plan = llm_response.get("plan", [])
        if not isinstance(plan, list) or not all("sub_goal" in step for step in plan):
            raise ValueError("Planner LLM did not return a valid plan structure.")
        logging.info(f"Generated plan with {len(plan)} steps.")
        return plan
    except Exception as e:
        logging.error(f"Failed to generate initial plan: {e}")
        return [{"sub_goal": f"Task failed during planning phase. Error: {e}"}]

# --- Executor Logic ---

def _build_executor_prompt(context: TaskContext, sub_goal: str, history: List[Dict]) -> str:
    """Builds the prompt for the Executor LLM call."""
    history_str = "\n".join([
        f"Action: {item['action']}({json.dumps(item['action_input'])})\nObservation: {item['observation']}"
        for item in history
    ]) if history else "No actions taken yet for this sub-goal."

    tools_str = "\n".join([f"- {name}" for name in TOOL_DISPATCHER.keys()])

    return f"""
You are an executor AI. Your job is to perform actions to achieve a specific sub-goal, which is part of a larger plan.

**Overall Goal:**
{context.goal}

**Summary of Previous Steps:**
{context.previous_steps_summary}

**Current Sub-Goal:**
{sub_goal}

**Available Tools:**
{tools_str}
(Refer to the planner's tool list for detailed descriptions and parameters.)

**History of Actions for this Sub-Goal:**
{history_str}

**Your Task:**
Based on the current sub-goal and the history, decide the next immediate action. You have two options:
1.  **Use a tool:** Choose one of the available tools to make progress on the sub-goal.
2.  **COMPLETE_SUB_GOAL:** If you have fully achieved the current sub-goal, use this special action. Do not use `finish_task` unless the sub-goal explicitly tells you to.

**Output Format:**
You MUST provide your response in a single JSON object, enclosed in a ```json ... ``` block.
The JSON object must have the following keys:
- "thought": A string explaining your reasoning.
- "action": The name of the tool to use or "COMPLETE_SUB_GOAL".
- "action_input": An object with parameters for the action. For "COMPLETE_SUB_GOAL", this can be an empty object `{{}}`.

**Example:**
```json
{{
    "thought": "I need to search for the financial reports mentioned in the sub-goal.",
    "action": "internet_search",
    "action_input": {{
        "query": "Tesla 2023 annual financial report"
    }}
}}
```

Now, what is your next action to achieve the sub-goal?
"""

async def _determine_next_action_for_sub_goal(context: TaskContext, sub_goal: str, history: List[Dict]) -> Dict:
    """Determines the next action for the agent to take for a given sub-goal."""
    prompt = _build_executor_prompt(context, sub_goal, history)
    messages = [{"role": "user", "content": prompt}]

    try:
        return await _call_llm_with_retry(messages, context.api_config)
    except Exception as e:
        logging.error(f"Executor LLM call failed: {e}")
        return {
            "thought": f"Error: The agent's internal thought process failed. Error: {e}",
            "action": "COMPLETE_SUB_GOAL",
            "action_input": {}
        }

async def _execute_tool(context: TaskContext, action: str, action_input: Dict) -> str:
    """Executes a single tool action."""
    tool_function = TOOL_DISPATCHER.get(action)
    if not tool_function:
        raise ValueError(f"Tool '{action}' not found.")

    logging.info(f"Executing tool '{action}' with input {action_input}")
    try:
        # Pass context to tools that need it
        if action in ["finish_task"]:
            observation = await tool_function(action_input, context.api_config, context=context)
        else:
            observation = await tool_function(action_input, context.api_config)
        return str(observation)
    except Exception as e:
        logging.error(f"An unexpected error occurred in tool '{action}': {e}", exc_info=True)
        raise e

# --- Main Task Orchestration ---

async def _execute_step(context: TaskContext, step_index: int, sub_goal: str) -> dict:
    """Executes all actions for a single plan step until the sub-goal is met."""
    logging.info(f"[{context.task_id}] Executing Step {step_index}: {sub_goal}")
    step_id = f"{context.task_id}-{step_index}"
    history = []
    max_actions_per_step = 10
    final_answer = None
    step_failed = False

    for i in range(max_actions_per_step):
        try:
            decision = await _determine_next_action_for_sub_goal(context, sub_goal, history)
            action = decision.get("action")
            action_input = decision.get("action_input", {})

            if not action or action.upper() == "COMPLETE_SUB_GOAL":
                logging.info(f"[{context.task_id}] Sub-goal '{sub_goal}' completed.")
                observation = "Sub-goal completed."
                break

            observation = await _execute_tool(context, action, action_input)
            history.append({"action": action, "action_input": action_input, "observation": observation})

            if action == "write_report_chapter":
                context.final_report_content += f"\n\n## {action_input.get('chapter_title', 'Chapter')}\n\n{observation}"

            if action == "finish_task":
                logging.info(f"[{context.task_id}] Finish task action received.")
                final_answer = observation
                break

            if "error" in observation.lower():
                 # Simple error check, can be improved
                logging.warning(f"Tool returned an error, considering self-correction: {observation}")

        except Exception as e:
            logging.error(f"Error executing action for sub-goal '{sub_goal}': {e}")
            observation = f"Error: {e}"
            history.append({"action": "error", "action_input": {}, "observation": str(e)})

            # If we are on the last action, mark the step as failed
            if i == max_actions_per_step - 1:
                step_failed = True
            continue

        if i == max_actions_per_step - 1:
            logging.warning(f"[{context.task_id}] Max actions reached for sub-goal '{sub_goal}'. Marking as failed.")
            step_failed = True
            break

    # Summarize the step's outcome
    step_summary = f"Sub-goal: {sub_goal}\nOutcome: {observation}"
    status = "failed" if step_failed else "completed"
    context.conn.execute(
        "UPDATE agent_task_steps SET status = ?, observation = ?, history = ? WHERE id = ?",
        (status, step_summary, json.dumps(history), step_id)
    )
    context.conn.commit()

    return {"step_summary": step_summary, "final_answer": final_answer, "failed": step_failed}

async def _execute_plan(context: TaskContext) -> tuple[bool, str]:
    """
    Executes the current plan in the context.
    Returns a tuple of (success, final_answer).
    """
    # 1. Save current plan to DB
    context.conn.execute("DELETE FROM agent_task_steps WHERE task_id = ?", (context.task_id,))
    for i, step_data in enumerate(context.plan):
        step_id = f"{context.task_id}-{i + 1}"
        sub_goal = step_data.get("sub_goal", "No sub-goal defined.")
        context.conn.execute(
            "INSERT INTO agent_task_steps (id, task_id, step_index, thought, action, action_input, status, history) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (step_id, context.task_id, i + 1, "Planner-generated step", sub_goal, "{}", "pending", "[]")
        )
    context.conn.commit()
    context.conn.execute("UPDATE agent_tasks SET status = ? WHERE id = ?", ("running", context.task_id))
    context.conn.commit()

    # 2. Execute Plan
    failed_step_summary = ""
    for i, step_data in enumerate(context.plan):
        step_index = i + 1
        sub_goal = step_data.get("sub_goal", "No sub-goal defined.")

        # Mark step as running
        step_id = f"{context.task_id}-{step_index}"
        context.conn.execute("UPDATE agent_task_steps SET status = ? WHERE id = ?", ("running", step_id))
        context.conn.commit()

        step_result = await _execute_step(context, step_index, sub_goal)

        context.previous_steps_summary += f"Step {step_index} Summary:\n{step_result['step_summary']}\n\n"

        if step_result["final_answer"]:
            context.final_report_content = step_result["final_answer"]
            return True, "" # Task is finished

        if step_result["failed"]:
            logging.warning(f"Step {step_index} failed. Current plan is not working.")
            failed_step_summary = step_result["step_summary"]
            return False, failed_step_summary # Stop executing current plan

    if not context.final_report_content:
        # If the plan completed but didn't call finish_task, something is wrong.
        context.final_report_content = context.previous_steps_summary + "\n\nWarning: The plan completed, but the 'finish_task' tool was not called. The above is a summary of the work done."

    return True, ""

async def _execute_task(conn: sqlite3.Connection, task_id: str, conversation_id: str, goal: str, api_config_dict: dict):
    api_config = ApiConfig(**api_config_dict)
    context = TaskContext(task_id, conversation_id, goal, api_config, conn)

    # 1. Initialize Task in DB
    current_time = int(time.time() * 1000)
    conn.execute(
        "INSERT INTO agent_tasks (id, conversation_id, user_goal, status, created_at, updated_at, plan) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (task_id, conversation_id, goal, "planning", current_time, current_time, "[]")
    )
    conn.commit()

    # 2. Generate Initial Plan
    context.plan = await _generate_initial_plan(context)
    conn.execute("UPDATE agent_tasks SET plan = ? WHERE id = ?", (json.dumps(context.plan), task_id))
    conn.commit()

    # 3. Execute plans until success or max replans reached
    while context.replan_attempts <= context.max_replans:
        plan_successful, failed_step_summary = await _execute_plan(context)

        if plan_successful:
            conn.execute("UPDATE agent_task_steps SET status = ? WHERE task_id = ?", ("completed", task_id))
            conn.commit()
            break # The plan was successful

        context.replan_attempts += 1
        if context.replan_attempts <= context.max_replans:
            logging.info(f"Attempting to re-plan... ({context.replan_attempts}/{context.max_replans})")
            context.plan = await _tool_replan(context, failed_step_summary)
        else:
            logging.error("Max re-plan attempts reached. Task failed.")
            context.final_report_content = context.previous_steps_summary + "\n\nERROR: The agent failed to complete the task after multiple attempts and re-plans."
            break

    # 4. Finalize Task
    logging.info(f"[{task_id}] Task execution finished. Final report generated.")
    current_time = int(time.time() * 1000)
    status = "completed" if context.final_report_content and "ERROR" not in context.final_report_content else "failed"
    conn.execute(
        "UPDATE agent_tasks SET status = ?, final_report = ?, updated_at = ? WHERE id = ?",
        (status, context.final_report_content, current_time, task_id)
    )
    conn.commit()


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
            "UPDATE agent_tasks SET status = ?, final_report = ?, updated_at = ? WHERE id = ?",
            ("failed", f"Task failed with an unhandled exception: {e}", current_time, task_id)
        )
        conn.commit()
    finally:
        if conn:
            conn.close()
