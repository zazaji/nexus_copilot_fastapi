# backend/app/agents/runner.py
import json
import logging
import time
import asyncio
import sqlite3
import os
from typing import Optional, Dict, Any

from app.database import get_db_connection_for_bg
from app.schemas.proxy_schemas import ApiConfig
from app.core.config import settings
from .context import TaskContext, _call_llm_with_retry
from .modes import run_plan_mode, run_explore_mode, run_write_mode, run_debate_mode
from .modes.refine_mode import refine_section_background
from .prompts import build_final_synthesis_prompt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FILES_DIR = os.path.join(settings.NEXUS_DATA_PATH, "files")

# --- Final Synthesis Step ---

def _parse_step_result(result_str: str) -> str:
    """Safely extracts content from a step result, which might be a JSON string."""
    try:
        # Attempt to parse the string as JSON
        data = json.loads(result_str)
        # If it's a dict and has a 'content' key, return that content
        if isinstance(data, dict) and 'content' in data:
            return data['content']
        # If it's JSON but not in the expected format, stringify it cleanly
        return json.dumps(data, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        # If it's not valid JSON, return the original string
        return result_str

async def _synthesize_final_report(context: TaskContext) -> str:
    """
    Takes all intermediate results and synthesizes them into a final, polished report.
    For write/research modes, it assembles the structured content.
    For other modes, it uses an LLM to synthesize a report from step results.
    """
    logging.info(f"[{context.task_id}] Starting final synthesis step.")
    
    # Gather all generated content
    if context.mode in ['write', 'research']:
        from .context import _assemble_final_report
        history = _assemble_final_report(context)
        # For write/research, the assembled report is the final report. No extra LLM call needed.
        logging.info(f"[{context.task_id}] Final report assembled directly from structured content.")
        return history
    else:
        # Parse each step result to extract clean content before joining
        parsed_results = [_parse_step_result(res) for res in context.step_results]
        history = "\n\n---\n\n".join(parsed_results)

    if not history.strip():
        logging.warning(f"[{context.task_id}] No content available for final synthesis. Returning a simple completion message.")
        return "The agent task is complete, but no content was generated to synthesize."

    prompt = build_final_synthesis_prompt(context, history)
    messages = [{"role": "user", "content": prompt}]
    
    # Use the user's primary chat model for the high-quality final synthesis
    synthesis_data = await _call_llm_with_retry(messages, context.api_config, max_tokens=4096)
    
    final_report = synthesis_data.get("report", "Failed to synthesize the final report.")
    logging.info(f"[{context.task_id}] Final synthesis complete.")
    return final_report

# --- Main Task Orchestrator ---

async def _execute_task(context: TaskContext, is_resume: bool = False):
    """Main orchestration logic: delegates to the appropriate mode runner."""
    if not is_resume:
        with open(context.log_file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Agent Task Log: {context.task_id}\n\n**Goal:** {context.goal}\n\n**Mode:** {context.mode}\n\n---\n\n")

    final_report = ""
    status = "running"
    error_message = None

    try:
        if context.mode == 'plan':
            await run_plan_mode(context)
        elif context.mode == 'explore':
            await run_explore_mode(context)
        elif context.mode == 'write' or context.mode == 'research':
            if is_resume:
                from .modes.write_mode import resume_write_mode
                await resume_write_mode(context)
            else:
                await run_write_mode(context)
        elif context.mode == 'debate':
            await run_debate_mode(context)

        # If the task is paused for user input, we exit early.
        cursor = context.conn.cursor()
        cursor.execute("SELECT status FROM agent_tasks WHERE id = ?", (context.task_id,))
        current_status = cursor.fetchone()[0]
        if current_status == 'awaiting_user_input':
            return

        # After the main logic, if not failed, perform the final synthesis
        if not context.is_finished and status != "failed":
             logging.warning(f"Task {context.task_id} completed all steps but was not marked as finished. This may be expected for some modes.")

        final_report = await _synthesize_final_report(context)

    except Exception as e:
        logging.error(f"Task execution failed with an exception: {e}", exc_info=True)
        error_message = f"Task failed during execution: {e}"
        status = "failed"

    if status != "failed":
        status = "completed"
    
    report_content = error_message if error_message else final_report

    report_file_path = os.path.join(FILES_DIR, f"{context.task_id}_report.md")
    with open(report_file_path, 'w', encoding='utf-8') as f:
        f.write(f"# Agent Final Report: {context.task_id}\n\n**Goal:** {context.goal}\n\n**Status:** {status.capitalize()}\n\n---\n\n{report_content}")

    current_time = int(time.time() * 1000)
    context.conn.execute("UPDATE agent_tasks SET status = ?, final_report = ?, updated_at = ? WHERE id = ?",
                 (status, report_content, current_time, context.task_id))
    context.conn.commit()

def run_task_background(task_id: str, conversation_id: Optional[str], goal: Optional[str], api_config_dict: Optional[Dict[str, Any]], mode: Optional[str], knowledge_base_selection: Optional[str], is_resume: bool = False, resume_payload: Optional[Dict[str, Any]] = None):
    """Entry point for running the agent task in a background thread."""
    logging.info(f"[{task_id}] Background task started. Resume: {is_resume}")
    conn = get_db_connection_for_bg()
    if not conn:
        logging.error(f"[{task_id}] FATAL: Could not get DB connection for background task.")
        return
    
    try:
        if is_resume:
            cursor = conn.cursor()
            cursor.execute("SELECT conversation_id, user_goal, mode, api_config, plan, research_content FROM agent_tasks WHERE id = ?", (task_id,))
            task_info = cursor.fetchone()
            if not task_info:
                raise Exception(f"Task {task_id} not found for resume.")
            
            db_conversation_id, db_goal, db_mode, api_config_json, plan_json, research_content_json = task_info
            
            if not api_config_json:
                raise Exception(f"API config not found for task {task_id}. Cannot resume.")
            
            api_config_dict_from_db = json.loads(api_config_json)
            api_config = ApiConfig(**api_config_dict_from_db)
            
            db_kb_selection = None 

            conn.execute("UPDATE agent_tasks SET status = 'running' WHERE id = ?", (task_id,))
            if resume_payload and 'plan' in resume_payload:
                conn.execute("UPDATE agent_tasks SET plan = ? WHERE id = ?", (json.dumps(resume_payload['plan']), task_id))

            conn.commit()

            context = TaskContext(task_id, db_conversation_id, db_goal, api_config, conn, db_mode, db_kb_selection)
            
            # Load resume payload into context
            if resume_payload:
                if 'plan' in resume_payload:
                    context.plan = resume_payload['plan']
                if 'elaboration' in resume_payload:
                    context.elaboration = resume_payload['elaboration']
            elif plan_json: 
                context.plan = json.loads(plan_json)
            
            if research_content_json: context.research_content = json.loads(research_content_json)
            
            asyncio.run(_execute_task(context, is_resume=True))

        else:
            current_time = int(time.time() * 1000)
            api_config_json = json.dumps(api_config_dict, ensure_ascii=False)
            conn.execute("INSERT INTO agent_tasks (id, conversation_id, user_goal, status, mode, created_at, api_config) VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (task_id, conversation_id, goal, "planning", mode, current_time, api_config_json))
            conn.commit()
            api_config = ApiConfig(**api_config_dict)
            context = TaskContext(task_id, conversation_id, goal, api_config, conn, mode, knowledge_base_selection)
            asyncio.run(_execute_task(context))

    except Exception as e:
        logging.error(f"[{task_id}] Task failed with unhandled exception: {e}", exc_info=True)
        current_time = int(time.time() * 1000)
        conn.execute("UPDATE agent_tasks SET status = ?, final_report = ?, updated_at = ? WHERE id = ?",
                     ("failed", f"Task failed with an unhandled exception: {e}", current_time, task_id))
        conn.commit()
    finally:
        if conn:
            conn.close()