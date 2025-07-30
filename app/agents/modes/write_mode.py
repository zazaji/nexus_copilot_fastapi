# backend/app/agents/modes/write_mode.py
import json
import logging
from typing import List, Dict, Any
import time

from ..context import TaskContext, _check_if_task_stopped, _save_step, _assemble_final_report, _call_llm_with_retry
from ..prompts import (
    build_writer_elaboration_prompt,
    build_writer_outline_prompt,
    build_writer_section_content_prompt,
    build_writer_critique_prompt,
    build_writer_refine_prompt,
)

MAX_REFINE_LOOPS = 10

async def _generate_critique_and_refine(
    context: TaskContext,
    task_description: str,
    requirements: str,
    generation_prompt: str,
    expected_json_key: str
) -> Dict[str, Any]:
    """
    A generic loop for generating, critiquing, and refining any piece of content.
    """
    content = ""
    critique_result = {}
    for i in range(MAX_REFINE_LOOPS):
        logging.info(f"[{context.task_id}] Starting generation/refinement loop {i+1}/{MAX_REFINE_LOOPS} for '{task_description}'")
        
        # 1. GENERATE (or REFINE)
        if i == 0:
            # Initial generation
            generation_messages = [{"role": "user", "content": generation_prompt}]
        else:
            # Refinement based on previous critique
            refine_prompt = build_writer_refine_prompt(context, task_description, requirements, content, critique_result.get('feedback', 'No feedback provided.'))
            generation_messages = [{"role": "user", "content": refine_prompt}]
        
        generated_data = await _call_llm_with_retry(generation_messages, context.api_config, max_tokens=4096)
        content = json.dumps(generated_data, ensure_ascii=False)

        # 2. CRITIQUE
        critique_prompt = build_writer_critique_prompt(context, task_description, requirements, content)
        critique_messages = [{"role": "user", "content": critique_prompt}]
        critique_result = await _call_llm_with_retry(critique_messages, context.api_config)

        if critique_result.get("meets_requirements", False):
            logging.info(f"[{context.task_id}] Content for '{task_description}' passed critique. Loop finished.")
            return generated_data
        else:
            logging.warning(f"[{context.task_id}] Content for '{task_description}' failed critique. Feedback: {critique_result.get('feedback')}. Retrying...")

    raise Exception(f"Failed to generate valid content for '{task_description}' after {MAX_REFINE_LOOPS} attempts.")


async def run_write_mode(context: TaskContext):
    """Orchestrates the multi-stage writing process with a critique/refine loop."""
    
    # Phase 1: Elaboration
    elaboration_action = "Phase 1: Generate Elaboration"
    elaboration_requirements = "The output must be a JSON object with an 'elaboration' key, containing 'summary', 'style', 'word_count', and 'strategy'."
    elaboration_prompt = build_writer_elaboration_prompt(context)
    elaboration_data = await _generate_critique_and_refine(context, elaboration_action, elaboration_requirements, elaboration_prompt, "elaboration")
    _save_step(context, elaboration_action, elaboration_data)
    elaboration = elaboration_data['elaboration']
    elaboration_str = f"Summary: {elaboration['summary']}\nStyle: {elaboration['style']}\nStrategy: {elaboration['strategy']}\nWord Count: {elaboration.get('word_count', 1500)}"

    # Phase 2: Outline
    outline_action = "Phase 2: Generate Outline"
    outline_requirements = "The output must be a JSON object with a 'plan' key, which is a list of objects. Each object must have 'sub_goal' and 'word_count' keys. Nested objects must have a 'steps' key."
    outline_prompt = build_writer_outline_prompt(context, elaboration_str)
    outline_data = await _generate_critique_and_refine(context, outline_action, outline_requirements, outline_prompt, "plan")
    _save_step(context, outline_action, outline_data)
    
    context.plan = outline_data.get("plan", [])
    
    def add_metadata_to_outline(nodes, prefix=""):
        for i, node in enumerate(nodes, 1):
            current_id = f"{prefix}{i}" if prefix else str(i)
            node['id'] = current_id
            if 'steps' in node and node['steps']:
                add_metadata_to_outline(node['steps'], f"{current_id}.")
    
    add_metadata_to_outline(context.plan)
    
    context.conn.execute("UPDATE agent_tasks SET plan = ?, status = ? WHERE id = ?", (json.dumps(context.plan, ensure_ascii=False), "awaiting_user_input", context.task_id))
    context.conn.commit()
    logging.info(f"[{context.task_id}] Outline generated. Awaiting user confirmation.")
    return


async def resume_write_mode(context: TaskContext):
    """Resumes the writing process after user has confirmed the plan and elaboration."""
    logging.info(f"[{context.task_id}] Resuming write mode with user-confirmed plan and elaboration.")
    
    _save_step(context, "Phase 1: Generate Elaboration", {"elaboration": context.elaboration})
    elaboration_str = f"Summary: {context.elaboration['summary']}\nStyle: {context.elaboration['style']}\nStrategy: {context.elaboration['strategy']}\nWord Count: {context.elaboration.get('word_count', 1500)}"

    # Phase 3 & 4: Content Generation
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
        
        logging.info(f"[{context.task_id}] Starting content generation for section: {section_title}")

        planned_word_count = section_node.get('word_count', 200)
        
        content_requirements = f"The output must be a JSON object with a 'content' key. The content should be approximately {planned_word_count} words and written in a {context.elaboration['style']} style. It must not include the section title."
        content_prompt = build_writer_section_content_prompt(context, elaboration_str, json.dumps(context.plan, ensure_ascii=False), section_title, history, planned_word_count)
        
        content_data = await _generate_critique_and_refine(context, write_action, content_requirements, content_prompt, "content")
        _save_step(context, write_action, content_data)
        
        content = content_data['content']
        
        context.research_content[section_id] = {
            "current": content,
            "history": []
        }
        history += f"## {section_title}\n\n{content}\n\n"
        
        progressive_report = _assemble_final_report(context)
        context.conn.execute(
            "UPDATE agent_tasks SET research_content = ?, final_report = ? WHERE id = ?", 
            (json.dumps(context.research_content, ensure_ascii=False), progressive_report, context.task_id)
        )
        context.conn.commit()

    final_report = _assemble_final_report(context)
    context.step_results.append(final_report)
    context.is_finished = True