"""
Text-to-Speech tool for LangGraph agent.

Provides TTS capability using OpenAI TTS API with streaming support.
"""

import os
import base64
from typing import Optional, Any, Generator
from langchain.tools import BaseTool


class TTSTool(BaseTool):
    """
    Tool for converting text to speech using OpenAI TTS.

    Supports both one-shot and streaming audio generation.
    """

    name: str = "speak_text"
    description: str = """
    Convert text to spoken audio using OpenAI TTS.
    Returns audio data that can be played back.
    Use for agent responses, game commentary, move announcements.

    Available voices: alloy, echo, fable, onyx, nova, shimmer
    """

    _client: Any = None
    voice: str = "alloy"
    model: str = "tts-1"

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: str = "alloy",
        model: str = "tts-1",
        **kwargs,
    ):
        """
        Initialize TTS tool.

        Args:
            api_key: OpenAI API key. Uses OPENAI_API_KEY env var if not provided.
            voice: Default voice (alloy, echo, fable, onyx, nova, shimmer)
            model: TTS model (tts-1 for speed, tts-1-hd for quality)
        """
        super().__init__(**kwargs)
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.voice = voice
        self.model = model

    def _run(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: str = "mp3",
    ) -> dict:
        """
        Convert text to audio (non-streaming, returns complete audio).

        Args:
            text: Text to convert to speech
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            speed: Speed factor (0.25 to 4.0)
            response_format: Output format (mp3, opus, aac, flac, wav, pcm)

        Returns:
            dict with 'success', 'audio_data' (base64), 'format'
        """
        try:
            response = self._client.audio.speech.create(
                model=self.model,
                voice=voice or self.voice,
                input=text,
                speed=speed,
                response_format=response_format,
            )

            audio_bytes = response.content
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            return {
                "success": True,
                "audio_data": audio_b64,
                "format": response_format,
                "text": text,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def stream(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: str = "mp3",
        chunk_size: int = 4096,
    ) -> Generator[bytes, None, None]:
        """
        Stream audio chunks for real-time playback.

        This method is not part of the LangChain tool interface,
        but can be used directly when streaming is needed.

        Args:
            text: Text to convert to speech
            voice: Voice to use
            speed: Speed factor (0.25 to 4.0)
            response_format: Output format
            chunk_size: Size of each audio chunk

        Yields:
            Audio data chunks as bytes
        """
        response = self._client.audio.speech.create(
            model=self.model,
            voice=voice or self.voice,
            input=text,
            speed=speed,
            response_format=response_format,
        )

        for chunk in response.iter_bytes(chunk_size=chunk_size):
            yield chunk

    async def _arun(self, *args, **kwargs) -> dict:
        """Async version."""
        import asyncio

        return await asyncio.to_thread(self._run, *args, **kwargs)

    @staticmethod
    def get_voices() -> dict:
        """
        Get information about available voices.

        Returns:
            dict with voice names and descriptions
        """
        return {
            "alloy": "Neutral and balanced",
            "echo": "Warm and engaging",
            "fable": "Authoritative and clear",
            "onyx": "Deep and resonant",
            "nova": "Friendly and upbeat",
            "shimmer": "Soft and expressive",
        }


class MoveAnnouncerTool(BaseTool):
    """
    Specialized tool for announcing chess moves.

    Generates natural language descriptions and can produce audio.
    """

    name: str = "announce_move"
    description: str = """
    Announce a chess move in natural language, optionally as audio.
    Takes a UCI move and board FEN, produces human-readable announcement.
    """

    _tts_tool: Optional[TTSTool] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, tts_tool: Optional[TTSTool] = None, **kwargs):
        """
        Initialize move announcer.

        Args:
            tts_tool: Optional TTS tool for audio generation
        """
        super().__init__(**kwargs)
        self._tts_tool = tts_tool

    def _describe_move(self, uci_move: str, fen: str) -> str:
        """Generate natural language description of a move."""
        import chess

        board = chess.Board(fen)
        move = chess.Move.from_uci(uci_move)
        piece = board.piece_at(move.from_square)

        piece_names = {
            chess.PAWN: "pawn",
            chess.KNIGHT: "knight",
            chess.BISHOP: "bishop",
            chess.ROOK: "rook",
            chess.QUEEN: "queen",
            chess.KING: "king",
        }

        piece_name = piece_names.get(piece.piece_type, "piece") if piece else "piece"
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)

        if board.is_castling(move):
            if move.to_square > move.from_square:
                return "Castles kingside"
            else:
                return "Castles queenside"
        elif board.is_capture(move):
            captured = board.piece_at(move.to_square)
            cap_name = piece_names.get(captured.piece_type, "piece") if captured else "piece"
            return f"{piece_name.capitalize()} takes {cap_name} on {to_sq}"
        else:
            return f"{piece_name.capitalize()} to {to_sq}"

    def _run(
        self,
        uci_move: str,
        fen: str,
        generate_audio: bool = False,
        player: str = "robot",
    ) -> dict:
        """
        Announce a chess move.

        Args:
            uci_move: Move in UCI format (e.g., "e2e4")
            fen: Current board FEN before the move
            generate_audio: Whether to generate audio
            player: Who made the move ("robot" or "human")

        Returns:
            dict with 'text', optional 'audio_data'
        """
        try:
            description = self._describe_move(uci_move, fen)

            if player == "robot":
                text = f"I play {description}."
            else:
                text = f"You played {description}."

            result = {
                "success": True,
                "text": text,
                "move_description": description,
            }

            if generate_audio and self._tts_tool:
                audio_result = self._tts_tool._run(text)
                if audio_result.get("success"):
                    result["audio_data"] = audio_result["audio_data"]
                    result["audio_format"] = audio_result["format"]

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def _arun(self, *args, **kwargs) -> dict:
        """Async version."""
        return self._run(*args, **kwargs)
