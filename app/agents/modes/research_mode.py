# backend/app/agents/modes/research_mode.py
import json
import logging
import asyncio
from typing import Dict, Any

from ..context import TaskContext, _assemble_final_report, _check_if_task_stopped, _call_llm_and_save
from ..prompts import build_writer_section_content_prompt
from app.database import get_db_connection_for_bg
from app.schemas.proxy_schemas import ApiConfig

async def run_research_mode(context: TaskContext):
    """Orchestrates the research process, starting with outline generation."""
    logging.info(f"[{context.task_id}] Starting research mode.")
    
    # Phase 1: Generate outline
    from ..prompts import build_writer_outline_prompt
    prompt = build_writer_outline_prompt(context, f"Style: In-depth research report. Strategy: Comprehensive and structured.", levels=4)
    outline_data = await _call_llm_and_save(context, "Phase 1: Generate Outline", [{"role": "user", "content": prompt}], max_tokens=4096)
    
    def add_metadata_to_outline(nodes, prefix=""):
        for i, node in enumerate(nodes, 1):
            current_id = f"{prefix}{i}" if prefix else str(i)
            node['id'] = current_id
            node['status'] = 'pending'
            if 'steps' in node and node['steps']:
                add_metadata_to_outline(node['steps'], f"{current_id}.")
    
    plan = outline_data.get("plan", [])
    add_metadata_to_outline(plan)
    context.plan = plan
    
    context.conn.execute("UPDATE agent_tasks SET plan = ?, status = ? WHERE id = ?", (json.dumps(context.plan), "running", context.task_id))
    context.conn.commit()
    logging.info(f"[{context.task_id}] Research outline generated and saved.")

    context.is_finished = False

async def _generate_node_content(task_id: str, node_id: str):
    """The actual logic to generate content for a single node."""
    conn = get_db_connection_for_bg()
    if not conn:
        logging.error(f"[{task_id}] Could not get DB connection for node generation.")
        return

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT user_goal, plan, research_content, api_config FROM agent_tasks WHERE id = ?", (task_id,))
        task_row = cursor.fetchone()
        if not task_row:
            raise Exception("Task not found in database.")

        db_goal, plan_json, research_content_json, api_config_json = task_row
        plan = json.loads(plan_json) if plan_json else []
        research_content = json.loads(research_content_json) if research_content_json else {}
        api_config = ApiConfig(**json.loads(api_config_json))

        target_node = None
        target_node_title = "Unknown Section"
        history = ""

        def find_node_and_build_history(nodes, prefix=""):
            nonlocal target_node, target_node_title
            for i, node in enumerate(nodes, 1):
                current_id = f"{prefix}{i}" if prefix else str(i)
                if node.get('id') == node_id:
                    target_node = node
                    target_node_title = f"{node.get('id', '')} {node.get('sub_goal', '')}".strip()
                    return True
                
                # Add content of already completed sibling/parent sections to history
                if 'steps' not in node or not node['steps']:
                    if node.get('id') in research_content:
                        history_content = research_content[node['id']].get('current', '')
                        history += f"## {node.get('id')} {node.get('sub_goal')}\n\n{history_content}\n\n"

                if 'steps' in node and node['steps']:
                    if find_node_and_build_history(node['steps'], f"{current_id}."):
                        return True
            return False
        
        find_node_and_build_history(plan)

        if not target_node:
            raise Exception(f"Node with id {node_id} not found in the plan.")

        target_node['status'] = 'writing'
        conn.execute("UPDATE agent_tasks SET plan = ? WHERE id = ?", (json.dumps(plan), task_id))
        conn.commit()

        # Generate content
        prompt = build_writer_section_content_prompt(
            context=None, # Pass None as context is reconstructed
            elaboration=f"Style: In-depth research report for goal: {db_goal}",
            outline=json.dumps(plan),
            chapter_strategy="Write a detailed and well-researched section for a comprehensive report.",
            section_title=target_node_title,
            history=history
        )
        
        # Temporarily override the chat model for this call
        original_chat_assignment = api_config.assignments.chat
        api_config.assignments.chat = api_config.assignments.suggestion # Use a faster model for generation
        
        data = await _call_llm_and_save(
            TaskContext(task_id, "", db_goal, api_config, conn, "research", None), # Mock context for saving
            f"Generate content for '{target_node_title}'", 
            [{"role": "user", "content": prompt}], 
            max_tokens=4096
        )
        content = data.get('content', f"Content generation failed for section {node_id}.")
        
        api_config.assignments.chat = original_chat_assignment # Restore original model

        research_content[node_id] = {"current": content, "history": []}
        target_node['status'] = 'completed'
        
        class MockContext:
            def __init__(self, goal: str, plan: list, research_content: Dict[str, Any]):
                self.goal = goal
                self.plan = plan
                self.research_content = research_content
        
        mock_context = MockContext(goal=db_goal, plan=plan, research_content=research_content)
        
        progressive_report = _assemble_final_report(mock_context)
        
        conn.execute(
            "UPDATE agent_tasks SET plan = ?, research_content = ?, final_report = ? WHERE id = ?",
            (json.dumps(plan), json.dumps(research_content), progressive_report, task_id)
        )
        conn.commit()
        logging.info(f"[{task_id}] Content for section '{node_id}' generated and report updated.")

    except Exception as e:
        logging.error(f"[{task_id}] Failed to generate content for node {node_id}: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()

def generate_node_content_background(task_id: str, node_id: str):
    """Entry point for the background task."""
    asyncio.run(_generate_node_content(task_id, node_id))