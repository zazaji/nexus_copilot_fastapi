# backend/app/agents/context.py
import logging
import inspect
import json
import re
from typing import List, Dict, Any, Callable, Coroutine, Optional
import sqlite3
import uuid

from app.schemas.proxy_schemas import ApiConfig
from app.services import shared_services
from app.core.config import settings
import os

class TaskContext:
    """A class to hold the state of a task."""
    def __init__(self, task_id: str, conversation_id: str, goal: str, api_config: ApiConfig, conn: sqlite3.Connection, mode: str, knowledge_base_selection: Optional[str]):
        self.task_id = task_id
        self.conversation_id = conversation_id
        self.goal = goal
        self.api_config = api_config
        self.conn = conn
        self.mode = mode
        self.knowledge_base_selection = knowledge_base_selection
        self.plan: Any = [] # Always initialize as a list for plan/explore, other modes will overwrite.
        self.step_outputs: Dict[int, str] = {}
        self.step_results: List[str] = []
        self.is_finished = False
        self.log_file_path = os.path.join(settings.NEXUS_DATA_PATH, "files", f"{task_id}_log.md")
        self.initial_context = ""
        self.research_content: Dict[str, Dict[str, Any]] = {}
        self.action_history: List[Dict[str, Any]] = []
        self.elaboration: Dict[str, Any] = {}

async def _tool_retrieve_knowledge(query: str, api_config: ApiConfig, context: TaskContext) -> str:
    """Retrieves information from the selected knowledge base or the internet based on a query. Use this to gather information needed to accomplish a sub-goal."""
    logging.info(f"Executing retrieve_knowledge with query: '{query}' and selection: '{context.knowledge_base_selection}'")
    if not query:
        raise ValueError("Tool 'retrieve_knowledge' requires a non-empty 'query' parameter.")

    selection = context.knowledge_base_selection
    if not selection or selection == "none":
        return "No knowledge source selected. Cannot retrieve information."

    sources = []
    if selection == "internet_search":
        search_results = await shared_services.internet_search(query, api_config)
        return "\n\n".join([f"Title: {res.get('title')}\nURL: {res.get('url')}\nContent: {res.get('content')}" for res in search_results])
    elif selection.startswith("online::"):
        kb_id = selection.split("::")[1]
        online_kb_config = next((kb for kb in (api_config.onlineKbs or []) if kb.id == kb_id), None)
        if online_kb_config:
            sources = await shared_services.query_online_kb(query, online_kb_config, 5, 0.6)
    else: # Local search
        if not api_config.execution or not api_config.execution.backendUrl:
             raise ValueError("Backend URL is not configured in the provided API config for the agent.")

        class MockRequest:
            def __init__(self, base_url: str):
                self.base_url = base_url
            
            def url_for(self, name: str) -> str:
                return f"{self.base_url}/api/v1/vector/{name.replace('_vectors', '')}"
        
        mock_request = MockRequest(api_config.execution.backendUrl)
        query_vector = await shared_services.get_embeddings(query, api_config, mock_request)
        sources = await shared_services.query_knowledge_base(query_vector, selection, mock_request, 5, 0.6, api_config)

    if not sources:
        return "No relevant information found in the knowledge base."
    
    return "\n\n".join([f"Source: {s.get('source_name') or s.get('file_path')}\nContent: {s['content_snippet']}" for s in sources])


async def _tool_reasoning_step(thought: str) -> str:
    """Use this tool when no external information is needed. Provide your detailed thought process or the next step of your analysis as the 'thought' parameter. This will be added to the history to guide your next action."""
    return thought

async def _tool_finish_task(context: TaskContext, conclusion: str) -> str:
    """
    Use this to provide the final answer or an intermediate reasoning step.
    The 'conclusion' parameter should be a comprehensive summary or the next piece of reasoning.
    This tool itself does NOT terminate the task; the task is terminated by a separate critique step.
    """
    logging.info(f"Executing finish_task.")
    # Note: We no longer set context.is_finished = True here.
    # The decision to finish is made by the critique step in the explore loop.
    
    if not conclusion and not context.step_results:
        return "The task is complete, but no conclusion was provided and no results were generated."
    
    return conclusion or "\n\n".join(context.step_results)

TOOL_DISPATCHER: Dict[str, Callable[..., Coroutine[Any, Any, str]]] = {
    "retrieve_knowledge": _tool_retrieve_knowledge,
    "reasoning_step": _tool_reasoning_step,
    "finish_task": _tool_finish_task,
}

async def _dispatch_tool_call(context: TaskContext, action: str, action_input: Dict) -> str:
    """
    Safely dispatches a tool call by filtering arguments based on the function's signature
    and validating that all required arguments are present.
    """
    if action not in TOOL_DISPATCHER:
        raise ValueError(f"Executor LLM chose an invalid tool: '{action}'")
    if not isinstance(action_input, dict):
        raise ValueError("Executor LLM failed to provide a valid 'action_input' object.")

    tool_function = TOOL_DISPATCHER[action]
    sig = inspect.signature(tool_function)
    
    # Filter action_input to only include parameters the function actually accepts
    valid_param_names = {p.name for p in sig.parameters.values()}
    filtered_action_input = {k: v for k, v in action_input.items() if k in valid_param_names}

    # Prepare the final arguments for the call, including context-injected ones
    call_kwargs = {**filtered_action_input}
    if 'api_config' in sig.parameters:
        call_kwargs['api_config'] = context.api_config
    if 'context' in sig.parameters:
        call_kwargs['context'] = context
        
    # Validate that all required parameters are present before calling
    for param in sig.parameters.values():
        if param.default is inspect.Parameter.empty and param.name not in call_kwargs:
            raise ValueError(f"Tool '{action}' is missing required parameter: '{param.name}'. LLM provided: {action_input}")
            
    return await tool_function(**call_kwargs)

def _clean_unicode_string(s: str) -> str:
    """
    Cleans a string by removing or replacing invalid Unicode characters,
    specifically surrogate pairs that cause encoding errors.
    """
    return s.encode('utf-8', 'replace').decode('utf-8')

async def _call_llm_with_retry(messages: List[Dict], api_config: ApiConfig, max_retries=3, max_tokens: Optional[int] = None) -> Dict:
    """Calls the LLM and robustly parses its JSON response, ignoring surrounding text."""
    # If max_tokens is not explicitly passed, try to get it from the model's config
    final_max_tokens = max_tokens
    if final_max_tokens is None and api_config.assignments.chat:
        chat_assignment = api_config.assignments.chat
        provider = next((p for p in api_config.providers if p.id == chat_assignment.providerId), None)
        if provider:
            model_info = next((m for m in provider.models if m.name == chat_assignment.modelName), None)
            if model_info and model_info.max_tokens:
                final_max_tokens = model_info.max_tokens

    for attempt in range(max_retries):
        response_message = {}
        try:
            response_message = await shared_services.get_completion(messages, api_config, max_tokens=final_max_tokens)
            
            if not isinstance(response_message, dict):
                raise TypeError(f"LLM response was not a dictionary, but a {type(response_message).__name__}.")
            llm_response_str = response_message.get("content")
            if llm_response_str is None:
                raise ValueError("LLM response did not contain a 'content' field.")
            
            # Clean the string to remove invalid unicode characters before parsing
            cleaned_llm_response_str = _clean_unicode_string(llm_response_str)

            if "</think>" in cleaned_llm_response_str:
                cleaned_llm_response_str = cleaned_llm_response_str.split("</think>")[-1]
            
            try:
                return json.loads(cleaned_llm_response_str.strip())
            except:
                pass
            
            json_match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', cleaned_llm_response_str, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logging.warning(f"Regex found a JSON block, but it was invalid: {e}")
            
            try:
                start = cleaned_llm_response_str.find('{')
                end = cleaned_llm_response_str.rfind('}') + 1
                if start != -1 and end != 0:
                    return json.loads(cleaned_llm_response_str[start:end])
            except json.JSONDecodeError:
                pass
                
            raise json.JSONDecodeError("No valid JSON object found in the response.", cleaned_llm_response_str, 0)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
            if attempt + 1 == max_retries:
                raise ValueError(f"LLM failed to return valid JSON after {max_retries} attempts. Last response: {response_message}") from e
        except Exception as e:
            logging.error(f"An unexpected error occurred during LLM call: {e}")
            raise e
    raise ValueError("LLM call failed after multiple retries.")


def _check_if_task_stopped(conn: sqlite3.Connection, task_id: str) -> bool:
    """Checks the database to see if the task has been stopped."""
    cursor = conn.cursor()
    cursor.execute("SELECT status FROM agent_tasks WHERE id = ?", (task_id,))
    result = cursor.fetchone()
    if result and result[0] != "running" and result[0] != "planning":
        logging.info(f"Task {task_id} has been stopped externally (status: {result[0]}). Terminating execution.")
        return True
    return False

def _save_step(context: TaskContext, action: str, result: Dict, status: str = "completed"):
    """Saves a new, unique, and ordered step to the database."""
    cursor = context.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM agent_task_steps WHERE task_id = ?", (context.task_id,))
    step_index = cursor.fetchone()[0] + 1
    
    step_id = str(uuid.uuid4())
    
    context.conn.execute(
        "INSERT INTO agent_task_steps (id, task_id, step_index, action, action_input, status, result) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (step_id, context.task_id, step_index, action, "{}", status, json.dumps(result, ensure_ascii=False))
    )
    context.conn.commit()

def _update_step_result(context: TaskContext, action: str, result: Dict):
    """Updates the result of an existing step."""
    sanitized_action = re.sub(r'[^a-zA-Z0-9_-]', '', action.replace(' ', '-'))
    step_id = f"{context.task_id}-{sanitized_action}"
    context.conn.execute(
        "UPDATE agent_task_steps SET result = ? WHERE id = ?",
        (json.dumps(result, ensure_ascii=False), step_id)
    )
    context.conn.commit()

async def _call_llm_and_save(context: TaskContext, action: str, messages: List[Dict], max_tokens: Optional[int] = None) -> Dict:
    """Helper to call LLM and save the step atomically."""
    _save_step(context, action, {}, status="running")
    data = await _call_llm_with_retry(messages, context.api_config, max_tokens=max_tokens)
    _save_step(context, action, data)
    return data

def _assemble_final_report(context: TaskContext) -> str:
    """Assembles the final report from the generated content for research/write modes."""
    report = f"# {context.goal}\n\n"
    def assemble_level(chapters, level=2):
        nonlocal report
        for chapter in chapters:
            report += f"\n{'#' * level} {chapter.get('id', '')} {chapter['sub_goal']}\n\n"
            if 'steps' in chapter and chapter['steps']:
                assemble_level(chapter['steps'], level + 1)
            else:
                # Correctly access the nested 'current' content
                node_content = context.research_content.get(chapter.get('id', ''), {})
                content_text = node_content.get('current', "*Content for this section is missing.*\n\n")
                report += content_text
    
    if isinstance(context.plan, list):
        assemble_level(context.plan)
    return report