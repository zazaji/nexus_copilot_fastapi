# backend/app/agents/modes/write_mode.py
import json
import logging
from typing import List, Dict, Any
import time

from ..context import TaskContext, _call_llm_with_retry, _check_if_task_stopped, _assemble_final_report
from ..prompts import (
    build_writer_elaboration_prompt,
    build_writer_outline_prompt,
    build_writer_chapter_strategy_prompt,
    build_writer_section_content_prompt,
    build_writer_critique_prompt,
    build_writer_refine_prompt,
)

MAX_REFINE_ITERATIONS = 10

async def _call_llm_with_critique_and_refine(
    context: TaskContext,
    action_name: str,
    generation_prompt: str,
    critique_prompt_builder: callable,
    refine_prompt_builder: callable,
    max_tokens: int = 2048
) -> Dict[str, Any]:
    """
    A robust wrapper for LLM calls that includes a critique and refinement loop.
    """
    # Initial Generation
    logging.info(f"[{context.task_id}] Generating initial content for '{action_name}'...")
    generated_data = await _call_llm_with_retry([{"role": "user", "content": generation_prompt}], context.api_config, max_tokens=max_tokens)
    
    current_content = generated_data
    
    for i in range(MAX_REFINE_ITERATIONS):
        if _check_if_task_stopped(context.conn, context.task_id):
            raise Exception("Task stopped by user.")

        # Critique Step
        logging.info(f"[{context.task_id}] Critiquing '{action_name}' - Attempt {i + 1}")
        critique_prompt = critique_prompt_builder(context, current_content)
        critique_data = await _call_llm_with_retry([{"role": "user", "content": critique_prompt}], context.api_config)
        
        if critique_data.get("passed", False):
            logging.info(f"[{context.task_id}] Critique passed for '{action_name}'.")
            return current_content
        
        critique_feedback = critique_data.get("overall_assessment", "No feedback provided.")
        logging.warning(f"[{context.task_id}] Critique failed for '{action_name}': {critique_feedback}")

        # Refine Step
        logging.info(f"[{context.task_id}] Refining '{action_name}' based on critique...")
        refine_prompt = refine_prompt_builder(context, current_content, critique_feedback)
        refined_data = await _call_llm_with_retry([{"role": "user", "content": refine_prompt}], context.api_config, max_tokens=max_tokens)
        
        current_content = refined_data

    logging.error(f"[{context.task_id}] Failed to meet quality standards for '{action_name}' after {MAX_REFINE_ITERATIONS} attempts.")
    return current_content # Return the last attempt even if it failed

def _save_step(context: TaskContext, action: str, result: Dict, status: str = "completed"):
    """Saves a step to the database."""
    cursor = context.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM agent_task_steps WHERE task_id = ?", (context.task_id,))
    step_index = cursor.fetchone()[0] + 1
    
    step_id = f"{context.task_id}-step-{step_index}"
    
    context.conn.execute(
        "INSERT INTO agent_task_steps (id, task_id, step_index, action, action_input, status, result) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (step_id, context.task_id, step_index, action, "{}", status, json.dumps(result, ensure_ascii=False))
    )
    context.conn.commit()

async def run_write_mode(context: TaskContext):
    """Orchestrates the multi-stage writing process with a critique-refine loop."""
    
    # Phase 1: Elaboration
    elaboration_action = "Phase 1: Generate Elaboration"
    elaboration_prompt = build_writer_elaboration_prompt(context)
    elaboration_data = await _call_llm_with_retry([{"role": "user", "content": elaboration_prompt}], context.api_config)
    context.elaboration = elaboration_data['elaboration']
    _save_step(context, elaboration_action, elaboration_data)

    elaboration_str = f"Summary: {context.elaboration['summary']}\nStyle: {context.elaboration['style']}\nStrategy: {context.elaboration['strategy']}\nWord Count: {context.elaboration.get('word_count', 1500)}"

    # Phase 2: Outline
    outline_action = "Phase 2: Generate Outline"
    outline_prompt = build_writer_outline_prompt(context, elaboration_str)
    outline_data = await _call_llm_with_retry([{"role": "user", "content": outline_prompt}], context.api_config, max_tokens=4096)
    
    context.plan = outline_data.get("plan", [])
    
    def add_metadata_to_outline(nodes, prefix=""):
        for i, node in enumerate(nodes, 1):
            current_id = f"{prefix}{i}" if prefix else str(i)
            node['id'] = current_id
            node['status'] = 'pending'
            if 'steps' in node and node['steps']:
                add_metadata_to_outline(node['steps'], f"{current_id}.")
    
    add_metadata_to_outline(context.plan)
    _save_step(context, outline_action, outline_data)
    
    context.conn.execute("UPDATE agent_tasks SET plan = ?, status = ? WHERE id = ?", (json.dumps(context.plan, ensure_ascii=False), "awaiting_user_input", context.task_id))
    context.conn.commit()
    logging.info(f"[{context.task_id}] Outline generated. Awaiting user confirmation.")
    return

async def resume_write_mode(context: TaskContext):
    """Resumes the writing process after user has confirmed the plan and elaboration."""
    logging.info(f"[{context.task_id}] Resuming write mode with user-confirmed plan.")
    
    elaboration_str = f"Summary: {context.elaboration['summary']}\nStyle: {context.elaboration['style']}\nStrategy: {context.elaboration['strategy']}"

    # Phase 3: Chapter Strategies (Simplified, no critique loop for this)
    chapter_strategies = {}
    async def get_strategies(nodes):
        for node in nodes:
            if 'steps' in node and node['steps']:
                node_title = f"{node['id']} {node['sub_goal']}"
                strategy_action = f"Phase 3: Strategy for '{node_title}'"
                
                if _check_if_task_stopped(context.conn, context.task_id): raise Exception("Task stopped by user.")
                prompt = build_writer_chapter_strategy_prompt(context, elaboration_str, json.dumps(context.plan, ensure_ascii=False), node_title)
                data = await _call_llm_with_retry([{"role": "user", "content": prompt}], context.api_config)
                _save_step(context, strategy_action, data)
                chapter_strategies[node['id']] = data['strategy']
                await get_strategies(node['steps'])
    
    await get_strategies(context.plan)

    # Phase 4: Content Generation with Critique-Refine Loop
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
        
        def update_node_status(status: str):
            def find_and_update(nodes):
                for node in nodes:
                    if node['id'] == section_id:
                        node['status'] = status
                        return True
                    if 'steps' in node and node['steps']:
                        if find_and_update(node['steps']):
                            return True
                return False
            find_and_update(context.plan)
            context.conn.execute("UPDATE agent_tasks SET plan = ? WHERE id = ?", (json.dumps(context.plan, ensure_ascii=False), context.task_id))
            context.conn.commit()

        update_node_status('writing')

        parent_id = ".".join(section_id.split('.')[:-1])
        chapter_strategy = chapter_strategies.get(parent_id, elaboration_str)
        planned_word_count = section_node.get('word_count', 0)
        
        # Define builders for the critique-refine loop
        def critique_builder(context, content):
            return build_writer_critique_prompt(context, section_title, content['content'], elaboration_str, json.dumps(context.plan, ensure_ascii=False), planned_word_count)
        
        def refine_builder(context, content, critique):
            return build_writer_refine_prompt(context, section_title, content['content'], critique, elaboration_str, json.dumps(context.plan, ensure_ascii=False))

        # Execute the loop
        generation_prompt = build_writer_section_content_prompt(context, elaboration_str, json.dumps(context.plan, ensure_ascii=False), chapter_strategy, section_title, history, planned_word_count)
        final_data = await _call_llm_with_critique_and_refine(context, f"Content for '{section_title}'", generation_prompt, critique_builder, refine_builder, max_tokens=4096)
        
        refined_content = final_data['content']
        
        _save_step(context, f"Final Content for '{section_title}'", {"content": refined_content})
        
        context.research_content[section_id] = {"current": refined_content, "history": []} # History is now implicit in the steps
        history += f"## {section_title}\n\n{refined_content}\n\n"
        
        update_node_status('completed')
        
        progressive_report = _assemble_final_report(context)
        context.conn.execute(
            "UPDATE agent_tasks SET research_content = ?, final_report = ? WHERE id = ?", 
            (json.dumps(context.research_content, ensure_ascii=False), progressive_report, context.task_id)
        )
        context.conn.commit()

    final_report = _assemble_final_report(context)
    context.step_results.append(final_report)
    context.is_finished = True