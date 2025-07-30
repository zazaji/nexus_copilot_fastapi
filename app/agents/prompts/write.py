# backend/app/agents/prompts/write.py
from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from .utils import get_language_instruction

if TYPE_CHECKING:
    from ..context import TaskContext

def build_writer_elaboration_prompt(context: 'TaskContext') -> str:
    """Phase 1: Generate the core summary, style, and strategy, including word count."""
    language_instruction = get_language_instruction(context)
    return f"""
You are a master strategist and writer. Your task is to elaborate on the user's goal for an article, paying close attention to any constraints provided. {language_instruction}

**User's Goal:** Write an article about "{context.goal}"

**Instructions:**
Generate a comprehensive elaboration covering four key areas:
1.  **Summary:** A brief, one-sentence summary of the article's core thesis.
2.  **Style:** The writing style and tone (e.g., academic, journalistic, technical, persuasive).
3.  **Word Count:** The target total word count. If the user specified a word count, use that. If not, estimate a reasonable word count based on the goal (e.g., 1500 for a standard blog post, 3000+ for a detailed report).
4.  **Strategy:** A high-level approach for structuring the article.

**Output Format:**
You MUST provide your response as a single, valid JSON object with one key: "elaboration". The value should be an object containing "summary", "style", "word_count", and "strategy".

**Example Response (User specified word count):**
-```json
{{
  "elaboration": {{
    "summary": "This article will analyze the impact of renewable energy on global geopolitics.",
    "style": "Analytical and objective, supported by data and expert opinions.",
    "word_count": 2000,
    "strategy": "Start with an introduction, followed by an analysis of key geopolitical shifts, case studies, and a conclusion on future trends."
  }}
}}
-```

Now, generate the elaboration for the goal: "{context.goal}". Your output must be ONLY the JSON object.
"""

def build_writer_outline_prompt(context: 'TaskContext', elaboration: str, levels: int = 3) -> str:
    """Phase 2: Generate a structured outline with word count allocation."""
    language_instruction = get_language_instruction(context)
    return f"""
You are a professional writer and editor AI. Your task is to create a detailed, multi-level outline for an article and intelligently allocate the target total word count across all sections. {language_instruction}

**User's Goal:** Write an article about "{context.goal}"

**Core Strategy, Style, and Word Count:**
---
{elaboration}
---

**Instructions:**
1.  Create a logical structure for the article that adheres to the core strategy.
2.  The outline should have a depth of approximately {levels} levels.
3.  For each section and sub-section (every node in the tree), you MUST include a "word_count" key with an estimated number of words for that specific part.
4.  The sum of all word counts for the terminal sections (those without "steps") MUST approximate the total target word count.
5.  The output must be a single JSON object with a single key "plan", which is a list of chapter objects.
6.  Each object in the hierarchy must have a "sub_goal" key (the title) and a "word_count" key.
7.  Objects that have children must have a "steps" key, which is a list of child objects.

**Example for a 1000-word article:**
-```json
{{
  "plan": [
    {{
      "sub_goal": "Chapter 1: Introduction",
      "word_count": 150,
      "steps": [
        {{
          "sub_goal": "1.1: Hook and Thesis",
          "word_count": 150
        }}
      ]
    }},
    {{
      "sub_goal": "Chapter 2: Main Body",
      "word_count": 700,
      "steps": [
        {{
          "sub_goal": "2.1: Key Point A",
          "word_count": 350
        }},
        {{
          "sub_goal": "2.2: Key Point B",
          "word_count": 350
        }}
      ]
    }},
    {{
      "sub_goal": "Chapter 3: Conclusion",
      "word_count": 150
    }}
  ]
}}
-```

Now, generate the JSON outline with word count allocation for the goal: "{context.goal}". Your output must be ONLY the JSON object.
"""

def build_writer_critique_prompt(context: 'TaskContext', task_description: str, requirements: str, content_to_critique: str) -> str:
    """Generic critique prompt for any generated content."""
    language_instruction = get_language_instruction(context)
    return f"""
You are a meticulous and demanding editor AI. Your task is to critique a piece of generated content to see if it meets a set of strict requirements. {language_instruction}

**Overall Goal:** {context.goal}
**Current Task:** {task_description}
**Requirements for this task:**
---
{requirements}
---

**Content to Critique:**
---
{content_to_critique}
---

**Your Instructions:**
1.  **Analyze:** Rigorously compare the "Content to Critique" against each point in the "Requirements".
2.  **Decision:** Determine if the content meets ALL requirements.
3.  **Feedback:** If it fails, provide specific, actionable feedback on exactly what is missing or needs to be changed. If it passes, state that it meets the requirements.

**Output Format:**
You MUST provide your response as a single, valid JSON object with two keys: "meets_requirements" (a boolean) and "feedback" (a string).

**Example (Failure):**
-```json
{{
  "meets_requirements": false,
  "feedback": "The generated outline is missing word count allocations for sub-sections 2.1 and 2.2. The total word count also does not sum up to the target of 2000 words."
}}
-```

**Example (Success):**
-```json
{{
  "meets_requirements": true,
  "feedback": "The content successfully meets all specified requirements for this stage."
}}
-```

Now, critique the provided content. Your output must be ONLY the JSON object.
"""

def build_writer_refine_prompt(context: 'TaskContext', task_description: str, requirements: str, original_content: str, critique_feedback: str) -> str:
    """Generic refinement prompt for any generated content."""
    language_instruction = get_language_instruction(context)
    return f"""
You are an expert writer AI specializing in revision and refinement. Your task is to rewrite a piece of content based on specific editorial feedback to ensure it meets all requirements. {language_instruction}

**Overall Goal:** {context.goal}
**Current Task:** {task_description}
**Original Requirements for this task:**
---
{requirements}
---

**Original Content (that failed critique):**
---
{original_content}
---

**Critique and Feedback to Address:**
---
{critique_feedback}
---

**Your Instructions:**
Rewrite the "Original Content" to fully address every point in the "Critique and Feedback". The new version must satisfy all of the "Original Requirements".

**Output Format:**
You MUST provide your response as a single, valid JSON object. The structure of this JSON should be identical to the expected format of the original content (e.g., if you are refining an outline, the output should be `{{ "plan": [...] }}`).

Now, generate the refined content. Your output must be ONLY the JSON object.
"""

def build_writer_section_content_prompt(context: 'TaskContext', elaboration: str, outline: str, section_title: str, history: str, planned_word_count: int) -> str:
    """Phase 4: Write the content for a specific section with a strong word count constraint."""
    language_instruction = get_language_instruction(context)
    return f"""
You are an expert writer AI. Your task is to write the content for a specific section of an article, adhering to all provided strategic context and constraints. {language_instruction}

**Overall Article Goal:** {context.goal}

**Core Strategy & Style:**
---
{elaboration}
---

**Full Article Outline:**
---
{outline}
---

**Previously Written Content (History):**
---
{history}
---

**Current Section to Write:** "{section_title}"

**Constraint:**
You MUST write approximately **{planned_word_count} words** for this section. Adhering to this word count is a primary requirement.

**Instructions:**
1.  Write the content for the specified section, respecting the word count constraint.
2.  The content should be comprehensive, well-structured, and engaging, following the article's overall style.
3.  The output MUST be in Markdown format.
4.  Do NOT include the section title (e.g., "### 1.1.1 Core Definition") in your output, only the body content.

**Output Format:**
You MUST provide your response as a single, valid JSON object with one key: "content".

**Example Response:**
-```json
{{
  "content": "The core definition of this topic revolves around three main pillars. Firstly, it involves the principle of... Secondly, it is characterized by... Finally, its practical application can be seen in..."
}}
-```

Now, write the content for the section "{section_title}". Your output must be ONLY the JSON object.
"""

def build_refine_section_prompt(context: 'TaskContext', outline: str, section_title: str, current_content: str, user_prompt: str, planned_word_count: int, current_word_count: int) -> str:
    """Builds a comprehensive prompt for the LLM to refine a specific section based on user input and word count analysis."""
    language_instruction = get_language_instruction(context)
    
    word_count_instruction = ""
    if planned_word_count > 0:
        ratio = current_word_count / planned_word_count if planned_word_count > 0 else 1
        if ratio < 0.75:
            word_count_instruction = f"Note: The current section is about {current_word_count} words, which is significantly shorter than the planned {planned_word_count} words. While addressing the user's request, please also try to expand the content with more details or examples."
        elif ratio > 1.25:
            word_count_instruction = f"Note: The current section is about {current_word_count} words, which is significantly longer than the planned {planned_word_count} words. While addressing the user's request, please also try to make the content more concise."
        else:
            word_count_instruction = f"Note: The current word count of {current_word_count} is appropriate for the planned {planned_word_count} words. Focus on the user's request for quality improvement."

    return f"""
You are an expert editor and writer AI. Your task is to refine a specific section of an article based on user instructions, while maintaining consistency with the overall article structure and goal. {language_instruction}

**Overall Article Goal:**
{context.goal}

**Full Article Outline (JSON):**
---
{outline}
---

**Section to Refine:**
"{section_title}"

**Current Content of the Section:**
---
{current_content}
---

**User's Refinement Instructions:**
"{user_prompt}"

**Editing Context:**
{word_count_instruction}

**Your Task:**
Rewrite the "Current Content of the Section" to incorporate the "User's Refinement Instructions" and the word count context.
- You MUST adhere to the user's instructions.
- The refined content should seamlessly fit into the article's overall structure and tone.
- The output MUST be in Markdown format.
- Do NOT include the section title (e.g., "### 1.2.1 Title") in your output, only the revised body content.

**Output Format:**
You MUST provide your response as a single, valid JSON object with one key: "content".

**Example Response:**
-```json
{{
  "content": "The refined content, rewritten according to the user's instructions, goes here. It should be a complete replacement for the original section content."
}}
-```

Now, generate the refined content. Your output must be ONLY the JSON object.
"""