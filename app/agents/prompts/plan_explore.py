# backend/app/agents/prompts/plan_explore.py
from __future__ import annotations
import json
import inspect
from typing import TYPE_CHECKING, Dict

from .utils import get_language_instruction

if TYPE_CHECKING:
    from ..context import TaskContext

def build_planner_prompt(context: 'TaskContext') -> str:
    """Builds the prompt for the Planner LLM call."""
    language_instruction = get_language_instruction(context)
    context_section = ""
    if context.initial_context:
        context_section = f"""
**Background Information:**
Use this information to inform your plan.
---
{context.initial_context}
---
"""
    return f"""
You are a master planner AI. Your job is to create a logical, step-by-step plan to achieve a user's goal.
The plan should be a sequence of sub-goals that break down the main goal into manageable parts.
Focus on the logical steps of research, analysis, and composition. Do not include explicit tool calls like "search the internet". The Executor will decide which tool to use for each step.
The final step should always be to "finish_task" to compile all previous results into a final answer.
{language_instruction}

**User's Goal:** {context.goal}
{context_section}
**Your Task:**
Generate a JSON object with a "plan" key, which is a list of steps. Each step is an object with a "sub_goal" key.

**Example:**
-```json
{{
    "plan": [
        {{"sub_goal": "Identify the key aspects of 'AI applications in scientific computing'."}},
        {{"sub_goal": "Research and summarize the role of AI in drug discovery."}},
        {{"sub_goal": "Research and summarize how AI is used in material science simulations."}},
        {{"sub_goal": "Compile the findings from the previous steps into a comprehensive report."}},
        {{"sub_goal": "finish_task"}}
    ]
}}
-```
Now, create a plan for the user's goal. Your output must be ONLY the JSON object, with no other text before or after it. 
"""

def build_executor_prompt(context: 'TaskContext', sub_goal: str, step_index: int, tool_dispatcher: Dict) -> str:
    """Builds the prompt for the Executor to choose a tool and generate a result for a sub-goal."""
    
    previous_results = "\n".join(context.step_results)
    if not previous_results:
        previous_results = "No results from previous steps yet."

    tools_with_params = []
    for name, func in tool_dispatcher.items():
        doc = inspect.getdoc(func) or ""
        tools_with_params.append(f'- "{name}": {doc}')
    tools_str = "\n".join(tools_with_params)
    
    language_instruction = get_language_instruction(context)

    return f"""
You are an expert executor AI. Your job is to achieve the current sub-goal by using the available tools and then provide a summary of your findings for that step.

**Overall Goal:** {context.goal}
** Initial Plan:**
---
{json.dumps(context.plan, indent=2)}
---
**Previous Steps' Results:**
---
{previous_results}
---

**Current Sub-Goal to Accomplish:**
"{sub_goal}"

**Available Tools:**
{tools_str}

**Your Task:**
1.  **Think**: Analyze the sub-goal and the previous results. Decide if you need to use a tool.
2.  **Act**: If a tool is needed, choose the single best tool and its parameters. The `action_input` keys **MUST** match the parameter names described in the tool's documentation. If no tool is needed (e.g., you are just summarizing previous results), set "action" to "none".
3.  **Result**: Based on your action (or lack thereof) and its observation, provide a result for the **current sub-goal**. The result MUST be in Markdown format, starting with a level 2 heading (##). {language_instruction}

**Output Format:**
You MUST provide your response in a single, valid JSON object. Do not add any other text before or after the JSON. The JSON object must have four keys: "thought", "action", "action_input", and "result".

**Example Response (if a tool is needed):**
-```json
{{
  "thought": "I need to find information about AI in drug discovery. The 'retrieve_knowledge' tool is perfect for this. I will use the sub-goal as the query.",
  "action": "retrieve_knowledge",
  "action_input": {{
    "query": "the role of AI in drug discovery"
  }},
  "result": "## AI in Drug Discovery\\n\\nBased on the retrieved information, AI significantly accelerates drug discovery by..."
}}
-```

**Example Response (if no tool is needed):**
-```json
{{
  "thought": "The sub-goal is to compile previous findings. I have the results from step 1 and 2. I don't need a new tool, I can synthesize the information I already have.",
  "action": "none",
  "action_input": {{}},
  "result": "## Combined Findings on AI Applications\\n\\nAI is pivotal in both scientific computing and drug discovery. In scientific computing, it is used for... In drug discovery, it helps by..."
}}
-```

Now, what is your action for the current sub-goal? Your output must be ONLY the JSON object.
"""

def build_explorer_act_prompt(context: 'TaskContext', tool_dispatcher: Dict, has_retrieval_tool: bool) -> str:
    """Builds the prompt for the Explorer to decide the next action, aware of past failures."""
    
    history = ""
    if not context.step_results:
        history = "This is the first step. No history yet."
    else:
        history_parts = []
        for i, result in enumerate(context.step_results):
            history_parts.append(f"### Step {i+1} Result:\n{result}")
        history = "\n\n".join(history_parts)

    failure_context = ""
    if any(action.get("name") == "retrieve_knowledge" and not action.get("success") for action in context.action_history):
        failure_context = "**CRITICAL CONTEXT:** You have previously attempted to use `retrieve_knowledge` and it failed because no knowledge source (like internet search or a local directory) was selected by the user. You **MUST NOT** attempt to use `retrieve_knowledge` again. Instead, you must rely on your internal knowledge to break down the problem and answer the user's goal step-by-step."

    if has_retrieval_tool:
        tools_with_params = []
        for name, func in tool_dispatcher.items():
            doc = inspect.getdoc(func) or ""
            tools_with_params.append(f'- "{name}": {doc}')
        tools_str = "\n".join(tools_with_params)
        
        return f"""
You are an autonomous AI explorer. Your mission is to achieve the user's goal through a series of iterative steps. You will be given the overall goal and the history of your previous actions and their results. Your task is to decide the single best next action to take.

**Overall Goal:** {context.goal}
**History of Previous Steps:**
---
{history}
---
{failure_context}

**Available Tools:**
{tools_str}
**CRITICAL INSTRUCTION:** You MUST choose an action from the 'Available Tools' list. Do not invent tools.

**Your Task:**
1.  **Think**: Analyze the overall goal, the history, and any critical context. What is the current status? What is the most critical piece of missing information or the next logical action to move closer to the goal? If a previous action failed, you must change your strategy.
2.  **Act**: Choose the single best tool and its parameters to execute your thought. The `action_input` keys **MUST** match the parameter names described in the tool's documentation.

**Output Format:**
You MUST provide your response in a single, valid JSON object. The JSON object must have three keys: "thought", "action", and "action_input".

**Example Response:**
-```json
{{
  "thought": "The goal is to create a presentation on Mars rovers. I have already researched the history of rovers. Now I need to find details about the Perseverance rover's instruments.",
  "action": "retrieve_knowledge",
  "action_input": {{
    "query": "instruments on NASA Perseverance Mars rover"
  }}
}}
-```
Now, what is your next step to achieve the goal? Your output must be ONLY the JSON object.
"""
    else:
        return f"""
You are an autonomous AI analyst. Your mission is to achieve the user's goal by breaking it down and reasoning through it step-by-step. You have no external tools and must rely solely on your internal knowledge.

**Overall Goal:** {context.goal}
**History of Your Reasoning So Far:**
---
{history}
---
{failure_context}

**Your Task:**
Based on the goal and your reasoning history, provide the very next step in your thought process to move closer to a complete answer. Your response should be a concise paragraph of pure reasoning.

**Example:**
If the goal is "explain the meaning of a state" and the history is empty, a good next step would be:
"First, I should define what a state is in the context of political science. A state is a polity under a system of governance that has a monopoly on the legitimate use of physical force within a defined territory."

Now, provide the next step of your reasoning. Your output should be ONLY the text of your thought process.
"""

def build_explorer_reflect_prompt(context: 'TaskContext', goal: str, action: str, action_input: Dict, observation: str) -> str:
    """Builds the prompt for the Explorer to reflect on an action's result."""
    language_instruction = get_language_instruction(context)
    return f"""
You are an AI assistant responsible for summarizing the results of an action. {language_instruction}

**Overall Goal:** {goal}
**Action Taken:**
`{action}` with input `{json.dumps(action_input)}`
**Observation from Action:**
---
{observation}
---
**Your Task:**
Based on the observation, write a concise, natural language summary that contributes to the overall goal. This summary will be the "result" of the current step and will be used in the history for future steps.
**CRITICAL INSTRUCTION:** Do NOT use any Markdown headings (like ##). Just provide the plain text summary.

**Output Format:**
You MUST provide your response as a single, valid JSON object with one key: "result".

**Example Response:**
-```json
{{
  "result": "The Perseverance rover is equipped with a suite of advanced instruments including Mastcam-Z for panoramic imaging, SuperCam for chemical analysis, and MOXIE for testing oxygen production on Mars. These tools are crucial for its mission to seek signs of ancient life."
}}
-```
Now, generate the result for the action taken. Your output must be ONLY the JSON object.
"""

def build_explorer_critique_prompt(context: 'TaskContext', goal: str, history: str) -> str:
    """Builds the prompt for the Explorer to critique its own progress and decide whether to finish."""
    language_instruction = get_language_instruction(context)
    return f"""
You are a meticulous and critical AI assistant. Your role is to evaluate the progress of an AI explorer and determine if its work is complete. {language_instruction}

**Original User Goal:** "{goal}"

**Explorer's Progress So Far (History of all steps):**
---
{history}
---

**Your Task:**
1.  **Critique:** Carefully compare the explorer's progress against the original user goal. Is the goal fully and comprehensively answered? Have all aspects of the query been addressed?
2.  **Decide:** Based on your critique, decide if the task is finished.

**Output Format:**
You MUST provide your response as a single, valid JSON object with two keys: "critique" and "is_finished".
- `critique`: A short, critical analysis of the current progress. If the task is not finished, this should explain what is still missing.
- `is_finished`: A boolean value (`true` or `false`).

**Example Response (Not Finished):**
-```json
{{
  "critique": "The definition of a state has been provided, but its key attributes like sovereignty, territory, and government have not been explained yet.",
  "is_finished": false
}}
-```

**Example Response (Finished):**
-```json
{{
  "critique": "The explorer has defined the state, explained its attributes, and provided a summary. The user's goal is fully achieved.",
  "is_finished": true
}}
-```

Now, provide your critique and decision. Your output must be ONLY the JSON object.
"""

def build_final_synthesis_prompt(context: 'TaskContext', history: str) -> str:
    """Builds the prompt for the final synthesizer LLM call."""
    language_instruction = get_language_instruction(context)
    return f"""
You are an expert report writer AI. Your task is to synthesize a collection of research notes and intermediate results into a single, final, comprehensive, and well-structured report that directly answers the user's original goal.

**User's Original Goal:**
"{context.goal}"

**Collected Information and Reasoning Steps:**
---
{history}
---

**Your Task:**
1.  Review all the collected information and reasoning steps.
2.  Write a final, high-quality report that directly and completely answers the user's goal.
3.  The report MUST be in Markdown format.
4.  The report MUST start with a level 1 heading (`#`) that is the user's original goal.
5.  Structure the report logically with appropriate subheadings (##, ###, etc.), lists, and formatting to be clear and readable.
6.  Do not include any meta-commentary like "Based on the information provided...". Just write the report itself.
7.  {language_instruction}

**Output Format:**
You MUST provide your response as a single, valid JSON object with one key: "report".

**Example Response:**
-```json
{{
  "report": "# How to Make Money with AI\\n\\nMaking money with AI can be approached from several angles, primarily focusing on developing AI-powered products, offering specialized AI services, or leveraging AI for content creation...\\n\\n## 1. Developing AI Products\\n\\nOne of the most direct ways to generate revenue is by building and selling software that solves a specific problem using AI...\\n\\n### 1.1. Identifying a Niche\\n..."
}}
-```

Now, synthesize the final report. Your output must be ONLY the JSON object.
"""