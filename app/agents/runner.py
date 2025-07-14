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

    try:
        outline = json.loads(outline_str)
        if isinstance(outline, list) and all(isinstance(i, str) for i in outline):
            return json.dumps(outline)
        else:
            raise ValueError("LLM did not return a valid JSON list of strings for the outline.")
    except json.JSONDecodeError:
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

async def _tool_finish_task(params: Dict[str, Any], api_config: ApiConfig) -> str:
    """
    Use this tool to signify that the entire task is complete.
    The 'final_answer' parameter should contain the complete and final response to the user's goal.
    """
    final_answer = params.get("final_answer")
    if not final_answer:
        raise ValueError("Finishing the task requires a 'final_answer' parameter.")
    return final_answer


TOOL_DISPATCHER: Dict[str, Callable[[Dict[str, Any], ApiConfig], Coroutine[Any, Any, str]]] = {
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
            json_start = llm_response_str.find("```json")
            json_end = llm_response_str.rfind("```")
            if json_start != -1:
                json_str = llm_response_str[json_start + 7:json_end].strip()
            else:
                json_str = llm_response_str
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.warning(f"Attempt {attempt + 1} failed: JSONDecodeError. Response: {llm_response_str}. Retrying...")
            if attempt + 1 == max_retries:
                raise e
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
        params = [p for p in sig.parameters if p != 'api_config']
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
The "sub_goal" should be a clear, concise instruction for the executor agent. The final step should always be to call the `finish_task` tool.

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
            "sub_goal": "Combine all written chapters into a single final report and call the finish_task tool with the report as the 'final_answer'."
        }}
    ]
}}
```

Now, create a plan for the user's goal.
"""

async def _generate_initial_plan(goal: str, api_config: ApiConfig) -> List[Dict]:
    """Uses the LLM to generate the initial high-level plan."""
    logging.info("Generating initial plan for goal: %s", goal)
    prompt = _build_planner_prompt(goal)
    messages = [{"role": "user", "content": prompt}]

    try:
        llm_response = await _call_llm_with_retry(messages, api_config)
        plan = llm_response.get("plan", [])
        if not isinstance(plan, list) or not all("sub_goal" in step for step in plan):
            raise ValueError("Planner LLM did not return a valid plan structure.")
        logging.info(f"Generated plan with {len(plan)} steps.")
        return plan
    except Exception as e:
        logging.error(f"Failed to generate initial plan: {e}")
        return [{"sub_goal": f"Task failed during planning phase. Error: {e}"}]

# --- Executor Logic ---

def _build_executor_prompt(goal: str, sub_goal: str, history: List[Dict], previous_steps_summary: str) -> str:
    """Builds the prompt for the Executor LLM call."""
    history_str = "\n".join([
        f"Action: {item['action']}({json.dumps(item['action_input'])})\nObservation: {item['observation']}"
        for item in history
    ]) if history else "No actions taken yet for this sub-goal."

    tools_str = "\n".join([f"- {name}" for name in TOOL_DISPATCHER.keys()])

    return f"""
You are an executor AI. Your job is to perform actions to achieve a specific sub-goal, which is part of a larger plan.

**Overall Goal:**
{goal}

**Summary of Previous Steps:**
{previous_steps_summary}

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

async def _determine_next_action_for_sub_goal(goal: str, sub_goal: str, history: List[Dict], api_config: ApiConfig, previous_steps_summary: str) -> Dict:
    """Determines the next action for the agent to take for a given sub-goal."""
    prompt = _build_executor_prompt(goal, sub_goal, history, previous_steps_summary)
    messages = [{"role": "user", "content": prompt}]

    try:
        return await _call_llm_with_retry(messages, api_config)
    except Exception as e:
        logging.error(f"Executor LLM call failed: {e}")
        return {
            "thought": f"Error: The agent's internal thought process failed. Error: {e}",
            "action": "COMPLETE_SUB_GOAL",
            "action_input": {}
        }

async def _execute_tool(action: str, action_input: Dict, api_config: ApiConfig) -> str:
    """Executes a single tool action."""
    tool_function = TOOL_DISPATCHER.get(action)
    if not tool_function:
        raise ValueError(f"Tool '{action}' not found.")

    logging.info(f"Executing tool '{action}' with input {action_input}")
    try:
        observation = await tool_function(action_input, api_config)
        return str(observation)
    except Exception as e:
        logging.error(f"An unexpected error occurred in tool '{action}': {e}", exc_info=True)
        raise e

# --- Main Task Orchestration ---

async def _execute_step(conn: sqlite3.Connection, task_id: str, step_index: int, goal: str, sub_goal: str, api_config: ApiConfig, previous_steps_summary: str) -> dict:
    """Executes all actions for a single plan step until the sub-goal is met."""
    logging.info(f"[{task_id}] Executing Step {step_index}: {sub_goal}")
    step_id = f"{task_id}-{step_index}"
    history = []
    max_actions_per_step = 10
    final_answer = None

    for i in range(max_actions_per_step):
        decision = await _determine_next_action_for_sub_goal(goal, sub_goal, history, api_config, previous_steps_summary)
        action = decision.get("action")
        action_input = decision.get("action_input", {})

        if not action or action.upper() == "COMPLETE_SUB_GOAL":
            logging.info(f"[{task_id}] Sub-goal '{sub_goal}' completed.")
            observation = "Sub-goal completed."
            break

        if action.upper() == "FINISH_TASK":
            logging.info(f"[{task_id}] Finish task action received.")
            final_answer = await _execute_tool(action, action_input, api_config)
            observation = f"Task finished with final answer: {final_answer}"
            break
        try:
            observation = await _execute_tool(action, action_input, api_config)
            history.append({"action": action, "action_input": action_input, "observation": observation})
        except Exception as e:
            logging.error(f"Error executing tool {action}: {e}")
            history.append({"action": action, "action_input": action_input, "observation": f"Error: {e}"})
            decision = await _self_correct(goal, sub_goal, history, api_config, previous_steps_summary)
            action = decision.get("action")
            action_input = decision.get("action_input", {})
            observation = await _execute_tool(action, action_input, api_config)
            history.append({"action": action, "action_input": action_input, "observation": observation})


        if i == max_actions_per_step - 1:
            logging.warning(f"[{task_id}] Max actions reached for sub-goal '{sub_goal}'. Moving on.")
            break

    # Summarize the step's outcome
    step_summary = f"Sub-goal: {sub_goal}\nOutcome: {observation}"
    conn.execute(
        "UPDATE agent_task_steps SET status = ?, observation = ? WHERE id = ?",
        ("completed", step_summary, step_id)
    )
    conn.commit()

    return {"step_summary": step_summary, "final_answer": final_answer}

def _build_self_correction_prompt(goal: str, sub_goal: str, history: List[Dict], previous_steps_summary: str) -> str:
    """Builds the prompt for the self-correction LLM call."""
    history_str = "\n".join([
        f"Action: {item['action']}({json.dumps(item['action_input'])})\nObservation: {item['observation']}"
        for item in history
    ]) if history else "No actions taken yet for this sub-goal."

    tools_str = "\n".join([f"- {name}" for name in TOOL_DISPATCHER.keys()])

    return f"""
You are an executor AI. Your job is to perform actions to achieve a specific sub-goal, which is part of a larger plan.
You have just encountered an error. Analyze the error and the history of your actions to decide on a new course of action.

**Overall Goal:**
{goal}

**Summary of Previous Steps:**
{previous_steps_summary}

**Current Sub-Goal:**
{sub_goal}

**Available Tools:**
{tools_str}
(Refer to the planner's tool list for detailed descriptions and parameters.)

**History of Actions for this Sub-Goal:**
{history_str}

**Your Task:**
You have encountered an error. Analyze the error and the history of your actions and decide the next immediate action. You have two options:
1.  **Use a tool:** Choose one of the available tools to make progress on the sub-goal.
2.  **COMPLETE_SUB_GOAL:** If you have fully achieved the current sub-goal, use this special action. Do not use `finish_task` unless the sub-goal explicitly tells you to.

**Output Format:**
You MUST provide your response in a single JSON object, enclosed in a ```json ... ``` block.
The JSON object must have the following keys:
- "thought": A string explaining your reasoning for the new action.
- "action": The name of the tool to use or "COMPLETE_SUB_GOAL".
- "action_input": An object with parameters for the action. For "COMPLETE_SUB_GOAL", this can be an empty object `{{}}`.

Now, what is your next action to achieve the sub-goal?
"""

async def _self_correct(goal: str, sub_goal: str, history: List[Dict], api_config: ApiConfig, previous_steps_summary: str) -> Dict:
    """Uses the LLM to self-correct after an error."""
    logging.info("Self-correcting for sub-goal: %s", sub_goal)
    prompt = _build_self_correction_prompt(goal, sub_goal, history, previous_steps_summary)
    messages = [{"role": "user", "content": prompt}]

    try:
        return await _call_llm_with_retry(messages, api_config)
    except Exception as e:
        logging.error(f"Self-correction LLM call failed: {e}")
        return {
            "thought": f"Error: The agent's internal thought process failed during self-correction. Error: {e}",
            "action": "COMPLETE_SUB_GOAL",
            "action_input": {}
        }


async def _execute_task(conn: sqlite3.Connection, task_id: str, conversation_id: str, goal: str, api_config_dict: dict):
    api_config = ApiConfig(**api_config_dict)

    # 1. Initialize Task in DB
    current_time = int(time.time() * 1000)
    conn.execute(
        "INSERT INTO agent_tasks (id, conversation_id, user_goal, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (task_id, conversation_id, goal, "planning", current_time, current_time)
    )
    conn.commit()

    # 2. Generate Plan
    plan = await _generate_initial_plan(goal, api_config)

    # 3. Save Plan to DB
    for i, step_data in enumerate(plan):
        step_id = f"{task_id}-{i + 1}"
        sub_goal = step_data.get("sub_goal", "No sub-goal defined.")
        conn.execute(
            "INSERT INTO agent_task_steps (id, task_id, step_index, thought, action, action_input, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (step_id, task_id, i + 1, "Planner-generated step", sub_goal, "{}", "pending")
        )
    conn.commit()
    conn.execute("UPDATE agent_tasks SET status = ? WHERE id = ?", ("running", task_id))
    conn.commit()

    # 4. Execute Plan
    previous_steps_summary = ""
    final_report = ""
    outline = None
    chapters = []
    for i, step_data in enumerate(plan):
        step_index = i + 1
        sub_goal = step_data.get("sub_goal", "No sub-goal defined.")

        # Mark step as running
        step_id = f"{task_id}-{step_index}"
        conn.execute("UPDATE agent_task_steps SET status = ? WHERE id = ?", ("running", step_id))
        conn.commit()

        step_result = await _execute_step(conn, task_id, step_index, goal, sub_goal, api_config, previous_steps_summary)
        step_summary = step_result["step_summary"]
        final_answer = step_result["final_answer"]


        if "generate_report_outline" in sub_goal.lower():
            try:
                outline = json.loads(step_summary.split("Outcome: ")[1])
            except (json.JSONDecodeError, IndexError):
                pass

        if "write_report_chapter" in sub_goal.lower():
            chapters.append(step_summary.split("Outcome: ")[1])

        if final_answer:
            final_report = final_answer
            break

        previous_steps_summary += f"Step {step_index} Summary:\n{step_summary}\n\n"
        # final_report = previous_steps_summary  # For now, the report is just a concatenation of summaries

    # 5. Finalize Task
    if chapters and not final_report:
        final_report = "\n\n".join(chapters)

    if not final_report:
        final_report = previous_steps_summary

    logging.info(f"[{task_id}] Task execution finished. Final report generated.")
    current_time = int(time.time() * 1000)
    conn.execute(
        "UPDATE agent_tasks SET status = ?, final_report = ?, updated_at = ? WHERE id = ?",
        ("completed", final_report, current_time, task_id)
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
