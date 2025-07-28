# backend/app/agents/modes/plan_explore_mode.py
import json
import logging
import inspect
import uuid
from typing import Dict, List

from app.services import shared_services
from ..context import TaskContext, TOOL_DISPATCHER, _call_llm_with_retry, _check_if_task_stopped, _dispatch_tool_call
from ..prompts import (
    build_planner_prompt,
    build_executor_prompt,
    build_explorer_act_prompt,
    build_explorer_reflect_prompt,
    build_explorer_critique_prompt,
)

async def _generate_and_save_plan(context: TaskContext) -> List[Dict]:
    """Uses the LLM to generate the initial plan and saves it."""
    logging.info("Generating initial plan for goal: %s", context.goal)
    prompt = build_planner_prompt(context)
    messages = [{"role": "user", "content": prompt}]

    plan_data = await _call_llm_with_retry(messages, context.api_config)
    plan = plan_data.get("plan", [])

    if not isinstance(plan, list) or (plan and not all("sub_goal" in step for step in plan)):
        raise ValueError("Planner LLM did not return a valid plan structure.")

    context.plan = plan
    context.conn.execute("UPDATE agent_tasks SET plan = ? WHERE id = ?", (json.dumps(context.plan, ensure_ascii=False), context.task_id))
    context.conn.commit()
    logging.info(f"Generated and saved plan with {len(plan)} steps.")
    return plan

async def _execute_plan_step(context: TaskContext, sub_goal: str, step_index: int):
    """Executes a single step of the plan, gets a result, and stores it."""
    step_id = str(uuid.uuid4())
    logging.info(f"[{context.task_id}] Executing Step {step_index} ({step_id}): {sub_goal}")
    
    context.conn.execute(
        "INSERT INTO agent_task_steps (id, task_id, step_index, action, action_input, status) VALUES (?, ?, ?, ?, ?, ?)",
        (step_id, context.task_id, step_index, "Planning...", "{}", "running")
    )
    context.conn.commit()

    if sub_goal == "finish_task":
        observation = await _dispatch_tool_call(context, "finish_task", {"conclusion": "\n\n".join(context.step_results)})
        context.step_outputs[step_index] = observation
        context.conn.execute("UPDATE agent_task_steps SET thought = ?, action = ?, action_input = ?, observation = ?, status = ?, result = ? WHERE id = ?",
                             ("Compiling final report from all previous steps.", "finish_task", "{}", observation, "completed", "Final report compiled.", step_id))
        context.conn.commit()
        return

    prompt = build_executor_prompt(context, sub_goal, step_index, TOOL_DISPATCHER)
    messages = [{"role": "user", "content": prompt}]

    decision = await _call_llm_with_retry(messages, context.api_config)
    action = decision.get("action")
    action_input = decision.get("action_input", {})
    thought = decision.get("thought", "")
    result_md = decision.get("result", f"## Step {step_index} Result\n\nNo result was generated for this step.")

    context.step_results.append(result_md)
    
    observation = "No tool executed."
    if action and action != "none":
        observation = await _dispatch_tool_call(context, action, action_input)
    
    context.step_outputs[step_index] = observation

    # Progressive final report update
    progressive_report = "\n\n".join(context.step_results)
    context.conn.execute("UPDATE agent_tasks SET final_report = ? WHERE id = ?", (progressive_report, context.task_id))

    context.conn.execute("UPDATE agent_task_steps SET thought = ?, action = ?, action_input = ?, observation = ?, status = ?, result = ? WHERE id = ?",
                         (thought, action, json.dumps(action_input, ensure_ascii=False), observation, "completed", result_md, step_id))
    context.conn.commit()

    with open(context.log_file_path, 'a', encoding='utf-8') as f:
        f.write(f"## Step {step_index}: {sub_goal}\n\n")
        f.write(f"### Thought\n\n> {thought}\n\n")
        f.write(f"### Action: `{action}`\n\n")
        f.write("#### Input\n\n```json\n")
        f.write(json.dumps(action_input, indent=2, ensure_ascii=False))
        f.write("\n```\n\n")
        f.write("#### Observation\n\n")
        f.write(f"```\n{observation}\n```\n\n")
        f.write(f"### Result\n\n{result_md}\n\n")
        f.write("---\n\n")

async def _execute_explore_step(context: TaskContext, step_index: int):
    """Executes a single step in explore mode using a Act-Reflect-Critique cycle."""
    step_id = str(uuid.uuid4())
    logging.info(f"[{context.task_id}] Exploring Step {step_index} ({step_id}): Deciding action...")
    
    context.conn.execute("INSERT INTO agent_task_steps (id, task_id, step_index, action, action_input, status) VALUES (?, ?, ?, ?, ?, ?)",
                         (step_id, context.task_id, step_index, "Thinking...", "{}", "running"))
    context.conn.commit()

    has_retrieval_tool = context.knowledge_base_selection and context.knowledge_base_selection != "none"

    # --- ACT ---
    act_prompt = build_explorer_act_prompt(context, TOOL_DISPATCHER, has_retrieval_tool)
    act_messages = [{"role": "user", "content": act_prompt}]
    
    if has_retrieval_tool:
        decision = await _call_llm_with_retry(act_messages, context.api_config)
    else:
        response_message = await shared_services.get_completion(act_messages, context.api_config)
        reasoning_text = response_message.get("content", "Could not generate a reasoning step.")
        decision = {
            "thought": "The user has not provided a knowledge source, so I must rely on my internal knowledge. My next step is to reason about the problem directly.",
            "action": "reasoning_step",
            "action_input": {"thought": reasoning_text}
        }

    action = decision.get("action")
    action_input = decision.get("action_input", {})
    thought = decision.get("thought", "")

    observation = "No tool executed."
    action_succeeded = True
    if action and action != "none":
        observation = await _dispatch_tool_call(context, action, action_input)
        if "No knowledge source selected" in observation:
            action_succeeded = False
    
    context.action_history.append({"name": action, "success": action_succeeded})
    context.step_outputs[step_index] = observation

    # --- REFLECT ---
    logging.info(f"[{context.task_id}] Reflecting on observation for step {step_index}...")
    reflect_prompt = build_explorer_reflect_prompt(context, context.goal, action, action_input, observation)
    reflect_messages = [{"role": "user", "content": reflect_prompt}]
    
    reflection = await _call_llm_with_retry(reflect_messages, context.api_config)
    result_md = reflection.get("result", f"Action '{action}' was performed, but no summary was generated.")
    
    context.step_results.append(result_md)

    # --- CRITIQUE ---
    logging.info(f"[{context.task_id}] Critiquing progress after step {step_index}...")
    full_history = "\n\n".join(context.step_results)
    critique_prompt = build_explorer_critique_prompt(context, context.goal, full_history)
    critique_messages = [{"role": "user", "content": critique_prompt}]
    
    critique_decision = await _call_llm_with_retry(critique_messages, context.api_config)
    critique_text = critique_decision.get("critique", "Critique failed.")
    is_finished = critique_decision.get("is_finished", False)
    
    if is_finished:
        context.is_finished = True
        logging.info(f"[{context.task_id}] Critique determined the task is finished.")
    else:
        # Add critique to history to guide the next step
        context.step_results.append(f"**Critique:** {critique_text}")

    # --- SAVE STEP ---
    progressive_report = "\n\n".join(context.step_results)
    context.conn.execute("UPDATE agent_tasks SET final_report = ? WHERE id = ?", (progressive_report, context.task_id))

    context.conn.execute("UPDATE agent_task_steps SET thought = ?, action = ?, action_input = ?, observation = ?, status = ?, result = ? WHERE id = ?",
                         (thought, action, json.dumps(action_input, ensure_ascii=False), observation, "completed", result_md, step_id))
    context.conn.commit()

    with open(context.log_file_path, 'a', encoding='utf-8') as f:
        f.write(f"## Step {step_index}\n\n")
        f.write(f"### Thought\n\n> {thought}\n\n")
        f.write(f"### Action: `{action}`\n\n")
        f.write("#### Input\n\n```json\n")
        f.write(json.dumps(action_input, indent=2, ensure_ascii=False))
        f.write("\n```\n\n")
        f.write("#### Observation\n\n")
        f.write(f"```\n{observation}\n```\n\n")
        f.write(f"### Result\n\n{result_md}\n\n")
        f.write(f"### Critique\n\n> {critique_text}\n\n")
        f.write("---\n\n")

async def run_plan_mode(context: TaskContext):
    plan = await _generate_and_save_plan(context)
    for i, step in enumerate(plan):
        if _check_if_task_stopped(context.conn, context.task_id):
            raise Exception("Task stopped by user.")
        await _execute_plan_step(context, step["sub_goal"], i + 1)

async def run_explore_mode(context: TaskContext):
    consecutive_failures = 0
    for i in range(1, 11): # Max 10 steps for explore mode
        if _check_if_task_stopped(context.conn, context.task_id):
            raise Exception("Task stopped by user.")
        
        await _execute_explore_step(context, i)

        # Check for consecutive failures of the same type
        last_action = context.action_history[-1] if context.action_history else None
        if last_action and not last_action["success"]:
            consecutive_failures += 1
        else:
            consecutive_failures = 0
        
        if consecutive_failures >= 2:
            logging.error(f"[{context.task_id}] Agent is stuck in a failure loop. Terminating task.")
            raise Exception("Agent is stuck trying the same failing action repeatedly.")

        if context.is_finished:
            break