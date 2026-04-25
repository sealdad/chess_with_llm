"""
LLM service for conversational chess agent using OpenAI GPT-4.
"""

import os
import asyncio
from typing import Optional, List, Dict, AsyncGenerator
from dataclasses import dataclass


@dataclass
class GameContext:
    """Context about the current chess game."""
    fen: str
    move_number: int
    whose_turn: str
    last_human_move: Optional[str] = None
    last_robot_move: Optional[str] = None
    robot_color: str = "black"
    game_status: str = "playing"
    move_history: Optional[List[str]] = None


class LLMService:
    """
    OpenAI GPT-4 powered conversation service.

    Handles chat completions with game context and conversation history.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        max_tokens: int = 300,
        temperature: float = 0.8,
        base_url: Optional[str] = None,
    ):
        """
        Initialize LLM service.

        Args:
            api_key: API key (uses env var if not provided)
            model: Model name (gpt-4o, gemini-2.0-flash, anthropic/claude-sonnet-4, etc.)
            max_tokens: Maximum response tokens
            temperature: Response creativity (0-2)
            base_url: Custom API endpoint (for Gemini, OpenRouter, etc.)
        """
        from openai import OpenAI

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = base_url

        # Detect provider from base_url and model name
        self.provider = self._detect_provider(base_url, model)
        print(f"[LLM] Provider detected: {self.provider} (model={model})")

    @staticmethod
    def _detect_provider(base_url: Optional[str], model: str) -> str:
        """Detect provider from model name first, then base_url.

        Model name wins because it dictates the API parameter contract
        (e.g. gpt-5.x requires max_completion_tokens regardless of the
        endpoint it's pointed at).
        """
        url = (base_url or "").lower()
        model_lower = model.lower()
        if model_lower.startswith("claude"):
            return "claude"
        if model_lower.startswith("gemini"):
            return "gemini"
        if model_lower.startswith("gpt") or model_lower.startswith("o1") or model_lower.startswith("o3"):
            return "openai"
        if "anthropic.com" in url:
            return "claude"
        if "googleapis.com" in url:
            return "gemini"
        if "openrouter.ai" in url:
            return "openrouter"
        return "openai"

    def _build_optional_params(self, max_tokens=None) -> dict:
        """Build provider-compatible optional parameters.

        Each provider has different requirements:
          OpenAI (gpt-5+): max_completion_tokens, temperature OK
          Claude: max_tokens, NO temperature (deprecated for some models)
          Gemini: max_tokens, temperature OK
          OpenRouter: max_tokens, temperature OK
        """
        params = {}
        val = max_tokens or self.max_tokens

        if self.provider == "openai":
            params["max_completion_tokens"] = val
            # gpt-5.x and o-series reasoning models reject non-default temperature
            ml = self.model.lower()
            is_reasoning = (
                ml.startswith("gpt-5")
                or ml.startswith("o1")
                or ml.startswith("o3")
                or ml.startswith("o4")
            )
            if not is_reasoning:
                params["temperature"] = self.temperature
        elif self.provider == "claude":
            params["max_tokens"] = val
            # Claude opus/sonnet may reject temperature — skip it
        elif self.provider == "gemini":
            # Gemini 2.5+ uses "thinking tokens" that consume max_tokens budget
            # Need much higher limit so thinking + output both fit
            params["max_tokens"] = max(val * 10, 2048)
            params["temperature"] = self.temperature
        else:
            # OpenRouter and others: use max_tokens (widely supported)
            params["max_tokens"] = val
            params["temperature"] = self.temperature

        return params

    def _build_game_context_message(self, game_context: Optional[GameContext]) -> str:
        """Build a context message about the current game state."""
        if game_context is None:
            return "No game is currently active."

        lines = [
            f"Move number: {game_context.move_number}",
            f"Turn: {game_context.whose_turn}",
            f"You are playing: {game_context.robot_color}",
        ]

        if game_context.last_human_move:
            lines.append(f"Human's last move: {game_context.last_human_move}")

        if game_context.last_robot_move:
            lines.append(f"Your last move: {game_context.last_robot_move}")

        if game_context.game_status != "playing":
            lines.append(f"Game status: {game_context.game_status}")

        # Add visual ASCII board so LLM can see piece positions
        try:
            import chess
            board = chess.Board(game_context.fen)
            lines.append(f"\nBoard (uppercase=White, lowercase=Black):\n{board}")
        except Exception:
            pass
        lines.append(f"FEN: {game_context.fen}")

        return "\n".join(lines)

    def _build_messages(
        self,
        user_message: str,
        system_prompt: str,
        conversation_history: List[Dict],
        game_context: Optional[GameContext] = None,
    ) -> List[Dict]:
        """Build the messages list for the API call."""
        messages = []

        context_info = self._build_game_context_message(game_context)

        if self.provider == "gemini":
            # Gemini: keep system prompt short, put context in user message
            messages.append({"role": "system", "content": system_prompt})
            if game_context:
                user_message = f"[Game State]\n{context_info}\n\n{user_message}"
        else:
            # OpenAI/Claude: system prompt with game context
            full_system = f"{system_prompt}\n\n--- Current Game State ---\n{context_info}"
            messages.append({"role": "system", "content": full_system})

        # Add conversation history (last N messages)
        max_history = 10  # Keep last 10 exchanges
        recent_history = conversation_history[-max_history * 2:] if conversation_history else []

        for msg in recent_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message,
        })

        return messages

    def chat(
        self,
        user_message: str,
        system_prompt: str,
        conversation_history: Optional[List[Dict]] = None,
        game_context: Optional[GameContext] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Get a chat response from GPT-4.

        Args:
            user_message: The user's message
            system_prompt: System prompt for personality
            conversation_history: Previous conversation messages
            game_context: Current game state
            max_tokens: Override max tokens for this call (uses default if None)

        Returns:
            The assistant's response text
        """
        messages = self._build_messages(
            user_message,
            system_prompt,
            conversation_history or [],
            game_context,
        )

        response = self._call_api(messages, max_tokens)
        content = response.choices[0].message.content
        if not content:
            finish = response.choices[0].finish_reason if response.choices else "unknown"
            print(f"[LLM] Empty response from {self.model} (finish_reason={finish}, provider={self.provider})")
        return content.strip() if content else ""

    def _call_api(self, messages, max_tokens=None, stream=False):
        """Call the API with provider-correct parameters."""
        params = self._build_optional_params(max_tokens)
        if stream:
            params["stream"] = True
        print(f"[LLM._call_api] provider={self.provider}, model={self.model}, params={list(params.keys())}")
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **params,
        )

    async def chat_async(
        self,
        user_message: str,
        system_prompt: str,
        conversation_history: Optional[List[Dict]] = None,
        game_context: Optional[GameContext] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Async version of chat."""
        return await asyncio.to_thread(
            self.chat,
            user_message,
            system_prompt,
            conversation_history,
            game_context,
            max_tokens,
        )

    def stream_chat(
        self,
        user_message: str,
        system_prompt: str,
        conversation_history: Optional[List[Dict]] = None,
        game_context: Optional[GameContext] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Stream chat response from GPT-4.

        Yields:
            Text chunks as they arrive
        """
        messages = self._build_messages(
            user_message,
            system_prompt,
            conversation_history or [],
            game_context,
        )

        stream = self._call_api(messages, max_tokens, stream=True)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def stream_chat_async(
        self,
        user_message: str,
        system_prompt: str,
        conversation_history: Optional[List[Dict]] = None,
        game_context: Optional[GameContext] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Async streaming chat response.

        Yields:
            Text chunks as they arrive
        """
        messages = self._build_messages(
            user_message,
            system_prompt,
            conversation_history or [],
            game_context,
        )

        stream = await asyncio.to_thread(
            lambda: self._call_api(messages, max_tokens, stream=True)
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                await asyncio.sleep(0)

    def generate_move_comment(
        self,
        move: str,
        is_human_move: bool,
        system_prompt: str,
        game_context: Optional[GameContext] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Generate a comment about a move.

        Args:
            move: The move in UCI format (e.g., "e2e4")
            is_human_move: True if human made the move
            system_prompt: Mode-specific system prompt
            game_context: Current game state
            conversation_history: Previous conversation

        Returns:
            Comment about the move
        """
        if is_human_move:
            prompt = f"The human just played {move}. React to this move briefly in your personality style."
        else:
            prompt = f"You just played {move}. Briefly comment on your move in your personality style."

        return self.chat(
            prompt,
            system_prompt,
            conversation_history,
            game_context,
        )

    async def generate_move_comment_async(
        self,
        move: str,
        is_human_move: bool,
        system_prompt: str,
        game_context: Optional[GameContext] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """Async version of generate_move_comment."""
        return await asyncio.to_thread(
            self.generate_move_comment,
            move,
            is_human_move,
            system_prompt,
            game_context,
            conversation_history,
        )
