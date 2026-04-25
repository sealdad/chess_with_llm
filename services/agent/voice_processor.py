"""
Voice event processor — consumes VoiceEvents and produces actions/responses.

This runs at the main.py level where game state lives, not inside the graph.
"""

import asyncio
from typing import Optional, Callable, Awaitable

from voice_events import VoiceEvent, VoiceEventPriority


# Type aliases for handler callbacks
GameCommandFn = Callable[[str, str], Awaitable[dict]]  # (command, text) -> response
ModeSwitchFn = Callable[[object, str], Awaitable[dict]]  # (mode, text) -> response
LanguageSwitchFn = Callable[[str, str], Awaitable[dict]]  # (language, text) -> response
MoveFn = Callable[[dict, str], Awaitable[dict]]  # (move_data, text) -> response
ConversationFn = Callable[[str], Awaitable[dict]]  # (text) -> response


class VoiceEventProcessor:
    """
    Processes voice events from the queue and returns action dicts.

    Each handler callback is injected from main.py where the actual
    game state and services live.
    """

    def __init__(
        self,
        on_game_command: Optional[GameCommandFn] = None,
        on_mode_switch: Optional[ModeSwitchFn] = None,
        on_language_switch: Optional[LanguageSwitchFn] = None,
        on_move: Optional[MoveFn] = None,
        on_conversation: Optional[ConversationFn] = None,
    ):
        self.on_game_command = on_game_command
        self.on_mode_switch = on_mode_switch
        self.on_language_switch = on_language_switch
        self.on_move = on_move
        self.on_conversation = on_conversation

    async def process(self, event: VoiceEvent) -> Optional[dict]:
        """
        Process a single voice event and return a response dict.

        Returns:
            dict with at least {"text": str} or None if no response needed.
        """
        intent = event.intent

        if intent == "game_command" and self.on_game_command:
            command = (event.data or {}).get("command", "")
            return await self.on_game_command(command, event.text)

        elif intent == "mode_switch" and self.on_mode_switch:
            mode = (event.data or {}).get("mode")
            return await self.on_mode_switch(mode, event.text)

        elif intent == "language_switch" and self.on_language_switch:
            language = (event.data or {}).get("language", "en")
            return await self.on_language_switch(language, event.text)

        elif intent == "move" and self.on_move:
            return await self.on_move(event.data or {}, event.text)

        elif intent == "conversation" and self.on_conversation:
            return await self.on_conversation(event.text)

        return None
