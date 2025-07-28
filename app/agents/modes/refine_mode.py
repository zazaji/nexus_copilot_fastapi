# backend/app/agents/modes/refine_mode.py
from re import Match


import json
import logging
import time
import asyncio
import sqlite3
import re
from typing import Dict, Any

from app.database import get_db_connection_for_bg
from app.schemas.proxy_schemas import ApiConfig
from ..context import TaskContext, _assemble_final_report, _save_step
from ..prompts import build_refine_section_prompt
from app.services import shared_services

def _extract_clean_content_from_response(response_message: Dict[str, Any]) -> str:
    """
    Robustly extracts the final, clean content string from a potentially nested
    and markdown-formatted LLM JSON response.
    """
    try:
        # Step 1: Get the outer content, which is expected to be a JSON string.
        outer_content_str = response_message.get("content")
        if not isinstance(outer_content_str, str):
            return "Refinement failed: LLM response content was not a string."

        json_match: Match[str] | None = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', outer_content_str, re.DOTALL)
        if not json_match:
            # If no JSON block is found, maybe the content is already clean text.
            # This can happen if the LLM doesn't follow instructions perfectly.
            # We return the inner string directly as a fallback.
            return outer_content_str.strip()

        outer_content_str = json_match.group(1) or json_match.group(2)


        # Step 2: Parse the outer JSON string.
        outer_data = json.loads(outer_content_str)
        if not isinstance(outer_data, dict) or "content" not in outer_data:
            return "Refinement failed: Outer JSON is missing 'content' key."

        # Step 3: Extract the inner content, which might be a markdown code block.
        inner_content_str = outer_data["content"]
        print('======inner_content_str',inner_content_str)
        # Step 4: Find and extract the JSON from within the markdown code block.
        json_match: Match[str] | None = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', inner_content_str, re.DOTALL)
        if not json_match:
            # If no JSON block is found, maybe the content is already clean text.
            # This can happen if the LLM doesn't follow instructions perfectly.
            # We return the inner string directly as a fallback.
            return inner_content_str.strip()

        json_str = json_match.group(1) or json_match.group(2)
        
        # Step 5: Parse the inner JSON.
        inner_data = json.loads(json_str)
        if not isinstance(inner_data, dict) or "content" not in inner_data:
            return "Refinement failed: Inner JSON is missing 'content' key."
            
        # Step 6: Return the final, clean content.
        return inner_data["content"]

    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logging.error(f"Error parsing refined content: {e}. Raw response content: {response_message.get('content')}")
        # Fallback to returning the raw inner content if parsing fails at any step
        return response_message.get('content', "Refinement failed due to a parsing error.")


async def _refine_node_content(task_id: str, node_id: str, user_prompt: str, model_identifier: str, is_manual: bool):
    """The actual logic to refine content for a single node."""
    conn = get_db_connection_for_bg()
    if not conn:
        logging.error(f"[{task_id}] Could not get DB connection for node refinement.")
        return

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT user_goal, plan, research_content, api_config, conversation_id, mode, knowledge_base_selection FROM agent_tasks WHERE id = ?", (task_id,))
        task_row = cursor.fetchone()
        if not task_row:
            raise Exception("Task not found in database.")

        db_goal, plan_json, research_content_json, api_config_json, conversation_id, mode, kb_selection = task_row
        
        if not api_config_json:
            raise Exception("API config not found for task. Cannot refine.")

        plan = json.loads(plan_json) if plan_json else []
        research_content = json.loads(research_content_json) if research_content_json else {}
        api_config = ApiConfig(**json.loads(api_config_json))
        
        temp_context = TaskContext(task_id, conversation_id, db_goal, api_config, conn, mode, kb_selection)

        content_node = research_content.get(node_id)
        if not content_node or "current" not in content_node:
            raise Exception(f"Node {node_id} has no content to refine.")
        
        current_content = content_node["current"]

        target_node_title = "Unknown Section"
        def find_node_info(nodes):
            nonlocal target_node_title
            for node in nodes:
                if node.get('id') == node_id:
                    target_node_title = f"{node.get('id', '')} {node.get('sub_goal', '')}".strip()
                    return True
                if 'steps' in node and node['steps']:
                    if find_node_info(node['steps']):
                        return True
            return False
        find_node_info(plan)

        refined_content = ""
        history_prompt = ""

        if is_manual:
            refined_content = user_prompt
            history_prompt = "Manual Replacement"
            logging.info(f"[{task_id}] Performing manual content replacement for node {node_id}.")
        else:
            history_prompt = user_prompt
            planned_word_count = 0
            def find_word_count(nodes):
                nonlocal planned_word_count
                for node in nodes:
                    if node.get('id') == node_id:
                        planned_word_count = node.get('word_count', 0)
                        return True
                    if 'steps' in node and node['steps']:
                        if find_word_count(node['steps']):
                            return True
                return False
            find_word_count(plan)

            current_word_count = len(current_content)
            prompt = build_refine_section_prompt(temp_context, json.dumps(plan, ensure_ascii=False), target_node_title, current_content, user_prompt, planned_word_count, current_word_count)
            
            provider_id, model_name = model_identifier.split("::")
            api_config.assignments.chat.providerId = provider_id
            api_config.assignments.chat.modelName = model_name

            response_message = await shared_services.get_completion([{"role": "user", "content": prompt}], api_config, max_tokens=4096)
            refined_content = _extract_clean_content_from_response(response_message)

        refine_action = f"Phase 5: Refine content for '{target_node_title}'"
        _save_step(temp_context, refine_action, {"content": refined_content})

        if "history" not in content_node:
            content_node["history"] = []
        
        content_node["history"].append({
            "prompt": history_prompt,
            "content": current_content,
            "timestamp": int(time.time() * 1000)
        })
        content_node["current"] = refined_content
        research_content[node_id] = content_node

        temp_context.plan = plan
        temp_context.research_content = research_content
        progressive_report = _assemble_final_report(temp_context)

        conn.execute(
            "UPDATE agent_tasks SET research_content = ?, final_report = ? WHERE id = ?",
            (json.dumps(research_content, ensure_ascii=False), progressive_report, task_id)
        )
        conn.commit()
        logging.info(f"[{task_id}] Content for section '{node_id}' refined and report updated.")

    except Exception as e:
        logging.error(f"[{task_id}] Failed to refine content for node {node_id}: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()

def refine_section_background(task_id: str, node_id: str, prompt: str, model: str, is_manual: bool):
    """Entry point for the background refinement task."""
    asyncio.run(_refine_node_content(task_id, node_id, prompt, model, is_manual))