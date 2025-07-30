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
You are a professional writer and editor AI. Your task is to create a detailed, multi-level outline for an article and intelligently allocate the target word count across all sections. {language_instruction}

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

def build_writer_chapter_strategy_prompt(context: 'TaskContext', elaboration: str, outline: str, chapter_title: str) -> str:
    """Phase 3: Generate the writing strategy for a specific chapter."""
    language_instruction = get_language_instruction(context)
    return f"""
You are an expert writing strategist. Your task is to devise a clear and concise writing strategy for a specific chapter of an article. {language_instruction}

**Overall Article Goal:** {context.goal}

**Core Strategy & Style:**
---
{elaboration}
---

**Full Article Outline:**
---
{outline}
---

**Current Chapter to Strategize:** "{chapter_title}"

**Instructions:**
Based on all the context provided, formulate a brief (2-3 sentences) writing strategy for this specific chapter. The strategy should guide the writing process to ensure this chapter fits coherently within the overall article structure and achieves its specific purpose.

**Output Format:**
You MUST provide your response as a single, valid JSON object with one key: "strategy".

**Example Response:**
-```json
{{
  "strategy": "This chapter will introduce the core concepts. It should start by defining the first key concept with clear examples, then transition smoothly to the second key concept, highlighting its relationship to the first."
}}
-```

Now, generate the writing strategy for the chapter "{chapter_title}". Your output must be ONLY the JSON object.
"""

def build_writer_section_content_prompt(context: 'TaskContext', elaboration: str, outline: str, chapter_strategy: str, section_title: str, history: str, planned_word_count: int) -> str:
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

**Strategy for the Current Chapter:**
---
{chapter_strategy}
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
2.  The content should be comprehensive, well-structured, and engaging, following the chapter's strategy and the article's overall style.
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

def build_writer_refine_prompt(context: 'TaskContext', section_title: str, original_content: str, planned_word_count: int, current_word_count: int) -> str:
    """Phase 5: Refine a piece of text, with strong, quantitative instructions on word count."""
    language_instruction = get_language_instruction(context)
    
    word_count_instruction = ""
    if planned_word_count > 0:
        ratio_reverse = int(10*(planned_word_count / current_word_count))/10  if planned_word_count > 0 else 1

        if ratio_reverse > 1.25:
            word_count_instruction = f"The current content is only {current_word_count} words, but the target is {planned_word_count} words. Your primary goal is to **expand the  length of content to {ratio_reverse} times**. Achieve this by adding more details, examples, deeper analysis, or elaborating on the key points."
        elif ratio_reverse < 0.75:
            ratio_reverse = int(10*(planned_word_count / current_word_count))/10
            word_count_instruction = f"The current content is {current_word_count} words, but the target is only {planned_word_count} words. Your primary goal is to **condense the  length of content to {ratio_reverse} times**. Achieve this by removing redundant phrases, combining sentences, and focusing only on the most critical information."
        else:
            word_count_instruction = f"The current word count of {current_word_count} is close to the planned {planned_word_count} words. Your primary goal is to **polish the text for clarity, flow, and depth**,focus to the subject, without significantly changing the length."

    return f"""
You are a master editor AI. Your task is to review and improve the following text for a specific article section, with a primary focus on the word count target. {language_instruction}

**Section Title:** "{section_title}"

**Original Content:**
---
{original_content}
---

**Primary Editing Goal:**
{word_count_instruction}

**Secondary Goal:**
In addition to the word count, improve the content's clarity, flow, and depth. Ensure it is polished and professional. Do not simply rephrase; add value.

**Output Format:**
You MUST provide your response as a single, valid JSON object with one key: "content".

**Example Response:**
-```json
{{
  "content": "The foundational definition of this topic is built upon three interconnected pillars. The first, and most critical, is the principle of..., which dictates that... This contrasts with the second pillar, characterized by..., a concept that emerged in the late 20th century. The third pillar, its practical application, is most vividly demonstrated in the field of..."
}}
-```

Now, refine the provided content based on your primary editing goal. Your output must be ONLY the JSON object.
"""

def build_refine_section_prompt(context: 'TaskContext', outline: str, section_title: str, current_content: str, user_prompt: str, planned_word_count: int, current_word_count: int) -> str:
    """Builds a comprehensive prompt for the LLM to refine a specific section based on user input and word count analysis."""
    language_instruction = get_language_instruction(context)
    
    word_count_instruction = ""
    if planned_word_count > 0:
        ratio = current_word_count / planned_word_count if planned_word_count > 0 else 1
        diff_percent = abs(ratio - 1) * 100
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