# backend/app/agents/modes/write_mode.py
import json
import logging
from typing import List
import time

from ..context import TaskContext, _call_llm_and_save, _check_if_task_stopped, _update_step_result, _assemble_final_report, _call_llm_with_retry, _save_step
from ..prompts import (
    build_writer_elaboration_prompt,
    build_writer_outline_prompt,
    build_writer_chapter_strategy_prompt,
    build_writer_section_content_prompt,
    build_writer_refine_prompt,
)

async def run_write_mode(context: TaskContext):
    """Orchestrates the multi-stage writing process with stateful recovery."""
    
    cursor = context.conn.cursor()
    cursor.execute("SELECT action, result FROM agent_task_steps WHERE task_id = ? AND status = 'completed'", (context.task_id,))
    completed_steps = {row[0]: json.loads(row[1]) for row in cursor.fetchall()}

    # Phase 1: Elaboration
    elaboration_action = "Phase 1: Generate Elaboration"
    if elaboration_action not in completed_steps:
        prompt = build_writer_elaboration_prompt(context)
        data = await _call_llm_and_save(context, elaboration_action, [{"role": "user", "content": prompt}])
        completed_steps[elaboration_action] = data
    elaboration = completed_steps[elaboration_action]['elaboration']
    elaboration_str = f"Summary: {elaboration['summary']}\nStyle: {elaboration['style']}\nStrategy: {elaboration['strategy']}\nWord Count: {elaboration.get('word_count', 1500)}"

    # Phase 2: Outline
    outline_action = "Phase 2: Generate Outline"
    if outline_action not in completed_steps:
        prompt = build_writer_outline_prompt(context, elaboration_str)
        data = await _call_llm_and_save(context, outline_action, [{"role": "user", "content": prompt}], max_tokens=4096)
        completed_steps[outline_action] = data
    
    context.plan = completed_steps[outline_action].get("plan", [])
    
    def add_metadata_to_outline(nodes, prefix=""):
        for i, node in enumerate(nodes, 1):
            current_id = f"{prefix}{i}" if prefix else str(i)
            node['id'] = current_id
            if 'steps' in node and node['steps']:
                add_metadata_to_outline(node['steps'], f"{current_id}.")
    
    add_metadata_to_outline(context.plan)
    
    # Save the plan and pause for user input
    context.conn.execute("UPDATE agent_tasks SET plan = ?, status = ? WHERE id = ?", (json.dumps(context.plan, ensure_ascii=False), "awaiting_user_input", context.task_id))
    context.conn.commit()
    logging.info(f"[{context.task_id}] Outline generated. Awaiting user confirmation.")
    return


async def resume_write_mode(context: TaskContext):
    """Resumes the writing process after user has confirmed the plan and elaboration."""
    logging.info(f"[{context.task_id}] Resuming write mode with user-confirmed plan and elaboration.")
    
    # The user-confirmed plan and elaboration are already on the context object from the runner
    # We need to update the elaboration step in the database for consistency
    _save_step(context, "Phase 1: Generate Elaboration", {"elaboration": context.elaboration})
    
    elaboration_str = f"Summary: {context.elaboration['summary']}\nStyle: {context.elaboration['style']}\nStrategy: {context.elaboration['strategy']}\nWord Count: {context.elaboration.get('word_count', 1500)}"

    # Phase 3: Chapter Strategies
    chapter_strategies = {}
    async def get_strategies(nodes):
        for node in nodes:
            if 'steps' in node and node['steps']:
                node_title = f"{node['id']} {node['sub_goal']}"
                strategy_action = f"Phase 3: Strategy for '{node_title}'"
                
                if _check_if_task_stopped(context.conn, context.task_id): raise Exception("Task stopped by user.")
                prompt = build_writer_chapter_strategy_prompt(context, elaboration_str, json.dumps(context.plan, ensure_ascii=False), node_title)
                data = await _call_llm_and_save(context, strategy_action, [{"role": "user", "content": prompt}])
                strategy = data['strategy']
                chapter_strategies[node['id']] = strategy
                await get_strategies(node['steps'])
    
    await get_strategies(context.plan)

    # Phase 4 & 5: Content Generation and Refinement
    writing_tasks = []
    def collect_writing_tasks(nodes):
        for node in nodes:
            if "steps" in node and node["steps"]:
                collect_writing_tasks(node["steps"])
            else:
                writing_tasks.append(node)
    
    collect_writing_tasks(context.plan)

    history = ""
    for section_node in writing_tasks:
        if _check_if_task_stopped(context.conn, context.task_id): raise Exception("Task stopped by user.")

        section_id = section_node['id']
        section_title = f"{section_id} {section_node['sub_goal']}"
        write_action = f"Phase 4: Write content for '{section_title}'"
        refine_action = f"Phase 5: Refine content for '{section_title}'"
        
        logging.info(f"[{context.task_id}] Starting content generation for section: {section_title}")

        parent_id = ".".join(section_id.split('.')[:-1])
        chapter_strategy = chapter_strategies.get(parent_id, elaboration_str)
        planned_word_count = section_node.get('word_count', 0)
        
        prompt = build_writer_section_content_prompt(context, elaboration_str, json.dumps(context.plan, ensure_ascii=False), chapter_strategy, section_title, history, planned_word_count)
        data = await _call_llm_and_save(context, write_action, [{"role": "user", "content": prompt}], max_tokens=4096)
        content = data['content']

        # Refine with word count awareness, just count length, important,don't remove
        current_word_count = len(content)
        prompt = build_writer_refine_prompt(context, section_title, content, planned_word_count, current_word_count)
        refine_data = await _call_llm_with_retry([{"role": "user", "content": prompt}], context.api_config, max_tokens=4096)
        refined_content = refine_data.get('content', content)
        
        _save_step(context, refine_action, {"content": refined_content})
        
        context.research_content[section_id] = {
            "current": refined_content,
            "history": [{
                "prompt": "Initial generation",
                "content": content,
                "timestamp": int(time.time() * 1000)
            }]
        }
        history += f"## {section_title}\n\n{refined_content}\n\n"
        
        progressive_report = _assemble_final_report(context)
        context.conn.execute(
            "UPDATE agent_tasks SET research_content = ?, final_report = ? WHERE id = ?", 
            (json.dumps(context.research_content, ensure_ascii=False), progressive_report, context.task_id)
        )
        context.conn.commit()

    final_report = _assemble_final_report(context)
    context.step_results.append(final_report)
    context.is_finished = True