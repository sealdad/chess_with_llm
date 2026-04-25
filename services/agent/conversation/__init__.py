"""
Conversation module for chess agent with Battle and Teach modes.
"""

from .modes import (
    GameMode,
    AgentMode,  # backward-compatible alias
    DIFFICULTY_DEPTH,
    MODE_PROMPTS,
    MODE_VOICES,
    MODE_NAMES,
    build_battle_prompt,
    build_teach_prompt,
    detect_mode_from_text,
    detect_language_from_text,
    get_mode_greeting,
    get_mode_name,
    get_mode_prompt,
    get_language_greeting,
    resolve_character_emotion,
)
from .llm_service import LLMService
from .context_manager import ConversationContext
from .intent_router import IntentRouter, Intent

__all__ = [
    "GameMode",
    "AgentMode",
    "DIFFICULTY_DEPTH",
    "MODE_PROMPTS",
    "MODE_VOICES",
    "MODE_NAMES",
    "build_battle_prompt",
    "build_teach_prompt",
    "detect_mode_from_text",
    "detect_language_from_text",
    "get_mode_greeting",
    "get_mode_name",
    "get_mode_prompt",
    "get_language_greeting",
    "resolve_character_emotion",
    "LLMService",
    "ConversationContext",
    "IntentRouter",
    "Intent",
]
