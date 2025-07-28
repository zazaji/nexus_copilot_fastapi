# backend/app/agents/prompts/utils.py
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import TaskContext

def get_language_instruction(context: 'TaskContext') -> str:
    """
    Generates a language instruction string based on the user's settings.
    Defaults to English if the setting is invalid or 'system'.
    """
    lang = "en"  # Default language
    if context.api_config and context.api_config.appearance:
        # Safely access the language, defaulting to 'system' then 'en'
        user_lang = getattr(context.api_config.appearance, 'language', 'system')
        if user_lang == "zh":
            lang = "zh"
    
    if lang == "zh":
        return "You MUST respond in Chinese."
    return "You MUST respond in English."