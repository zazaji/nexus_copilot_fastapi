# backend/app/agents/prompts/debate.py
from __future__ import annotations
from typing import Dict, TYPE_CHECKING
from .utils import get_language_instruction

if TYPE_CHECKING:
    from ..context import TaskContext

def build_debate_persona_prompt(context: 'TaskContext') -> str:
    """Generates personas and complexity parameters for the debate."""
    language_instruction = get_language_instruction(context)
    return f"""
You are a debate setup AI. Your task is to create distinct personas and set the debate parameters for the topic: "{context.goal}". {language_instruction}

**Instructions:**
1.  **Create Personas:** For the roles Pro, Con, and Judge, define:
    *   `style`: A short description of their debating style.
    *   `framework`: The core intellectual framework they will use.
2.  **Assess Complexity:** Based on the topic's complexity, determine the debate's structure:
    *   `max_rounds`: The maximum number of rounds (between 4 and 12). A simple topic might be 4-5 rounds, a complex one 10-12. Default to 8 for average topics.
    *   `score_diff_threshold`: The score difference that will end the debate early (between 5 and 15). A higher threshold for more nuanced topics. Default to 8.

**Output Format:**
You MUST provide your response as a single, valid JSON object with two keys: "personas" and "complexity".

**Example Response:**
-```json
{{
  "personas": {{
    "pro": {{
      "style": "Passionate and forward-looking, focusing on potential benefits and innovation.",
      "framework": "Techno-optimism and progress-oriented consequentialism."
    }},
    "con": {{
      "style": "Cautious and skeptical, emphasizing risks and unintended consequences.",
      "framework": "Precautionary principle and critical theory."
    }},
    "judge": {{
      "style": "Analytical and methodical, focused on the clarity and logical consistency of arguments.",
      "framework": "Formal logic and evidence-based reasoning."
    }}
  }},
  "complexity": {{
      "max_rounds": 8,
      "score_diff_threshold": 10
  }}
}}
-```

Now, generate the personas and complexity for the debate on "{context.goal}". Your output must be ONLY the JSON object.
"""

def build_debate_judge_rules_prompt(context: 'TaskContext', personas: Dict, history: str, round_num: int) -> str:
    """Generates the rules for a specific round of the debate."""
    language_instruction = get_language_instruction(context)
    return f"""
You are the Judge in a formal debate. It is your duty to set the rules and focus for each round. {language_instruction}

**Debate Topic:** "{context.goal}"
**Debate History (Previous Rounds):**
---
{history if history else "This is the first round. No history yet."}
---

**Your Task:**
You are now beginning **Round {round_num}**. Based on the debate so far, define a clear, concise, and neutral focus for this round.

**Output Format:**
You MUST provide your response as a single, valid JSON object with one key: "rules".

**Example:** `{{"rules": "Direct Rebuttal: Each side will directly address and refute the key points made by their opponent in the previous round."}}`

Now, set the rules for **Round {round_num}**. Your output must be ONLY the JSON object.
"""

def build_debate_argument_prompt(context: 'TaskContext', personas: Dict, history: str, round_num: int, round_rules: str, role: str) -> str:
    """Generates an argument for a debater in a specific round."""
    my_persona = personas[role]
    language_instruction = get_language_instruction(context)

    return f"""
You are an expert debater. You must argue your position convincingly, adhering strictly to your assigned persona and the rules of the current round. {language_instruction}

**Debate Topic:** "{context.goal}"
**Your Assigned Role:** {role.upper()}
**Your Persona:**
- **Style:** {my_persona['style']}
- **Framework:** {my_persona['framework']}

**Debate History & Current Round ({round_num}) Rules:**
---
{history}
**Current Round Rules:** "{round_rules}"
---

**Your Task:**
Construct a powerful and persuasive argument for your position.
1.  Your argument **MUST** follow the rules and focus for Round {round_num}.
2.  Your argument **MUST** be consistent with your assigned persona.
3.  Your argument **MUST** be concise, approximately 100 words.
4.  Your argument should be well-structured and presented in Markdown format.

**Output Format:**
You MUST provide your response as a single, valid JSON object with one key: "argument".

**Example:** `{{"argument": "My opponent's argument ignores the core ethical implications. From a deontological perspective, the action itself is unjust, regardless of its supposed utilitarian benefits. The data they cited is irrelevant when fundamental rights are at stake."}}`

Now, generate your argument for **Round {round_num}**. Your output must be ONLY the JSON object.
"""

def build_debate_judge_verdict_prompt(context: 'TaskContext', personas: Dict, history: str, is_final: bool) -> str:
    """Generates a verdict/evaluation from the judge, either for a round or for the final conclusion."""
    language_instruction = get_language_instruction(context)
    task_description = "deliver a final verdict for the entire debate" if is_final else "evaluate the latest round of arguments"

    return f"""
You are the Judge of a formal debate. It is now your duty to {task_description}. {language_instruction}

**Debate Topic:** "{context.goal}"
**Full Debate Transcript So Far:**
---
{history}
---

**Your Task:**
1.  **Analyze the Arguments:** Review the latest arguments from the Pro and Con sides. Evaluate them based on logical consistency, evidence, and adherence to the round's rules.
2.  **Declare a Winner:** Clearly state whether "pro" or "con" won {"the debate" if is_final else "this specific round"}.
3.  **Assign Scores:** Provide a score out of 10 for both the Pro and Con sides for their performance in {"the entire debate" if is_final else "this round"}.
4.  **Provide Justification:** Write a detailed justification for your decision in Markdown format, explaining your scoring.

**Output Format:**
You MUST provide your response as a single, valid JSON object with three keys: "winner", "score", and "justification". The "score" key should be an object with "pro" and "con" keys.

**Example Response (for a round):**
-```json
{{
  "winner": "con",
  "score": {{
    "pro": 7,
    "con": 8
  }},
  "justification": "In this round, the Con side was more effective. They successfully dismantled the Pro's primary analogy and presented a compelling counter-example. The Pro side's rebuttal was passionate but lacked sufficient data to counter the Con's point about economic impact."
}}
-```

Now, {task_description}. Your output must be ONLY the JSON object.
"""