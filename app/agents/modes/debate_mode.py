# backend/app/agents/modes/debate_mode.py
import json
import logging
from typing import Dict

from ..context import TaskContext, _call_llm_and_save, _check_if_task_stopped
from ..prompts import (
    build_debate_persona_prompt,
    build_debate_judge_rules_prompt,
    build_debate_argument_prompt,
    build_debate_judge_verdict_prompt,
)

async def run_debate_mode(context: TaskContext):
    """Orchestrates the multi-stage debate process with dynamic round evaluation."""
    logging.info(f"[{context.task_id}] Starting debate mode.")
    
    debate_state = context.plan if isinstance(context.plan, dict) and context.plan else {}

    # Phase 1: Generate Personas and Complexity
    if 'personas' not in debate_state:
        prompt = build_debate_persona_prompt(context)
        initial_data = await _call_llm_and_save(context, "Phase 1: Generate Personas & Complexity", [{"role": "user", "content": prompt}])
        debate_state['personas'] = initial_data.get('personas', {})
        debate_state['complexity'] = initial_data.get('complexity', {'max_rounds': 8, 'score_diff_threshold': 8})
        debate_state['rounds'] = []
        context.conn.execute("UPDATE agent_tasks SET plan = ?, status = ? WHERE id = ?", (json.dumps(debate_state, ensure_ascii=False), "running", context.task_id))
        context.conn.commit()
    
    history = ""
    complexity = debate_state.get('complexity', {'max_rounds': 8, 'score_diff_threshold': 8})
    max_rounds = complexity.get('max_rounds', 8)
    score_diff_threshold = complexity.get('score_diff_threshold', 8)
    
    # Phase 2: Round-based Debate Loop
    while len(debate_state.get('rounds', [])) < max_rounds:
        round_num = len(debate_state.get('rounds', [])) + 1
        if _check_if_task_stopped(context.conn, context.task_id): raise Exception("Task stopped by user.")

        # Judge sets rules for the new round
        prompt = build_debate_judge_rules_prompt(context, debate_state['personas'], history, round_num)
        rules_data = await _call_llm_and_save(context, f"Phase 2.{round_num}.1: Judge Sets Rules", [{"role": "user", "content": prompt}])
        round_rules = rules_data.get('rules', f'Round {round_num} begins.')
        
        current_round = {"round": round_num, "rules": round_rules}
        debate_state['rounds'].append(current_round)
        context.conn.execute("UPDATE agent_tasks SET plan = ? WHERE id = ?", (json.dumps(debate_state, ensure_ascii=False), context.task_id))
        context.conn.commit()
        history += f"### Round {round_num}: {round_rules}\n\n"
        
        # Pro argues
        if _check_if_task_stopped(context.conn, context.task_id): raise Exception("Task stopped by user.")
        prompt = build_debate_argument_prompt(context, debate_state['personas'], history, round_num, round_rules, 'pro')
        pro_data = await _call_llm_and_save(context, f"Phase 2.{round_num}.2: Pro Argues", [{"role": "user", "content": prompt}], max_tokens=1024)
        pro_argument = pro_data.get('argument', 'The Pro side has no argument for this round.')
        current_round['pro_argument'] = pro_argument
        context.conn.execute("UPDATE agent_tasks SET plan = ? WHERE id = ?", (json.dumps(debate_state, ensure_ascii=False), context.task_id))
        context.conn.commit()
        history += f"**Pro's Argument:**\n{pro_argument}\n\n"
        
        # Con argues
        if _check_if_task_stopped(context.conn, context.task_id): raise Exception("Task stopped by user.")
        prompt = build_debate_argument_prompt(context, debate_state['personas'], history, round_num, round_rules, 'con')
        con_data = await _call_llm_and_save(context, f"Phase 2.{round_num}.3: Con Argues", [{"role": "user", "content": prompt}], max_tokens=1024)
        con_argument = con_data.get('argument', 'The Con side has no argument for this round.')
        current_round['con_argument'] = con_argument
        context.conn.execute("UPDATE agent_tasks SET plan = ? WHERE id = ?", (json.dumps(debate_state, ensure_ascii=False), context.task_id))
        context.conn.commit()
        history += f"**Con's Argument:**\n{con_argument}\n\n"

        # Judge evaluates round
        if _check_if_task_stopped(context.conn, context.task_id): raise Exception("Task stopped by user.")
        prompt = build_debate_judge_verdict_prompt(context, debate_state['personas'], history, is_final=False)
        evaluation_data = await _call_llm_and_save(context, f"Phase 2.{round_num}.4: Judge Evaluates", [{"role": "user", "content": prompt}], max_tokens=1024)
        current_round['evaluation'] = evaluation_data
        context.conn.execute("UPDATE agent_tasks SET plan = ? WHERE id = ?", (json.dumps(debate_state, ensure_ascii=False), context.task_id))
        context.conn.commit()
        history += f"**Judge's Evaluation:**\n{evaluation_data.get('justification', '')}\n\n"

        # Check for early finish
        total_pro_score = sum(r.get('evaluation', {}).get('score', {}).get('pro', 0) for r in debate_state['rounds'])
        total_con_score = sum(r.get('evaluation', {}).get('score', {}).get('con', 0) for r in debate_state['rounds'])
        if abs(total_pro_score - total_con_score) >= score_diff_threshold:
            logging.info(f"[{context.task_id}] Score difference threshold of {score_diff_threshold} reached. Ending debate early.")
            break

    # Phase 3: Final Verdict
    if 'verdict' not in debate_state:
        if _check_if_task_stopped(context.conn, context.task_id): raise Exception("Task stopped by user.")
        prompt = build_debate_judge_verdict_prompt(context, debate_state['personas'], history, is_final=True)
        verdict_data = await _call_llm_and_save(context, "Phase 3: Final Verdict", [{"role": "user", "content": prompt}], max_tokens=2048)
        debate_state['verdict'] = verdict_data
        context.conn.execute("UPDATE agent_tasks SET plan = ? WHERE id = ?", (json.dumps(debate_state, ensure_ascii=False), context.task_id))
        context.conn.commit()
    
    verdict_data = debate_state['verdict']
    final_report = f"## Final Verdict on '{context.goal}'\n\n**Winner:** {verdict_data.get('winner', 'N/A').upper()}\n\n{verdict_data.get('justification', 'No justification provided.')}"
    context.step_results.append(final_report)
    context.is_finished = True