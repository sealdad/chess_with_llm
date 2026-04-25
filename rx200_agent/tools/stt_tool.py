"""
Speech-to-Text tool for LangGraph agent.

Provides STT capability using OpenAI Whisper API.
"""

import io
import os
import base64
from typing import Optional, Any
from langchain.tools import BaseTool


class STTTool(BaseTool):
    """
    Tool for converting speech to text using OpenAI Whisper.

    This tool takes audio data (base64 encoded or raw bytes) and
    returns the transcribed text.
    """

    name: str = "transcribe_speech"
    description: str = """
    Convert spoken audio to text using OpenAI Whisper.
    Input: Audio data (base64 encoded) and optional format/language hints.
    Returns the transcribed text.
    Use this tool when you need to understand what the human said.
    """

    _client: Any = None
    model: str = "whisper-1"

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize STT tool.

        Args:
            api_key: OpenAI API key. Uses OPENAI_API_KEY env var if not provided.
        """
        super().__init__(**kwargs)
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def _run(
        self,
        audio_data: str,
        audio_format: str = "wav",
        language: str = "en",
    ) -> dict:
        """
        Transcribe audio to text.

        Args:
            audio_data: Base64 encoded audio data
            audio_format: Audio format (wav, mp3, webm, m4a)
            language: Language hint for better accuracy

        Returns:
            dict with 'success', 'text', and optional 'error'
        """
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_data)
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = f"audio.{audio_format}"

            # Call Whisper API
            transcription = self._client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language=language,
                response_format="text",
            )

            return {
                "success": True,
                "text": transcription.strip(),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "text": "",
            }

    async def _arun(self, *args, **kwargs) -> dict:
        """Async version - uses sync for now (OpenAI SDK)."""
        import asyncio

        return await asyncio.to_thread(self._run, *args, **kwargs)


class MoveParserTool(BaseTool):
    """
    Tool for parsing natural language chess moves to UCI format.

    Uses the current board state to understand and validate spoken moves.
    """

    name: str = "parse_chess_move"
    description: str = """
    Parse natural language chess move description to UCI format.
    Examples:
        "pawn to e4" -> "e2e4"
        "knight f3" -> "g1f3"
        "castle kingside" -> "e1g1"

    Requires the spoken move text and current board FEN.
    """

    def _run(self, spoken_move: str, current_fen: str) -> dict:
        """
        Parse spoken move to UCI.

        Args:
            spoken_move: Natural language move (e.g., "pawn to e4")
            current_fen: Current board position FEN

        Returns:
            dict with 'success', 'uci_move', and 'interpretation'
        """
        import re
        import chess

        board = chess.Board(current_fen)
        spoken_lower = spoken_move.lower().strip()

        # Castling
        if "castle" in spoken_lower or "castling" in spoken_lower:
            if "king" in spoken_lower or "short" in spoken_lower:
                uci = "e1g1" if board.turn == chess.WHITE else "e8g8"
            elif "queen" in spoken_lower or "long" in spoken_lower:
                uci = "e1c1" if board.turn == chess.WHITE else "e8c8"
            else:
                return {"success": False, "error": "Unclear castling direction"}

            move = chess.Move.from_uci(uci)
            if move in board.legal_moves:
                return {
                    "success": True,
                    "uci_move": uci,
                    "interpretation": spoken_move,
                }
            return {"success": False, "error": "Castling not legal"}

        # Extract squares mentioned
        squares = re.findall(r"[a-h][1-8]", spoken_lower)

        # Piece names
        piece_map = {
            "king": chess.KING,
            "queen": chess.QUEEN,
            "rook": chess.ROOK,
            "bishop": chess.BISHOP,
            "knight": chess.KNIGHT,
            "pawn": chess.PAWN,
            "horse": chess.KNIGHT,
        }

        piece_type = None
        for name, ptype in piece_map.items():
            if name in spoken_lower:
                piece_type = ptype
                break

        # Try to find matching legal move
        for move in board.legal_moves:
            from_sq = chess.square_name(move.from_square)
            to_sq = chess.square_name(move.to_square)
            piece = board.piece_at(move.from_square)

            if len(squares) >= 1:
                if squares[-1] == to_sq:
                    if len(squares) == 2 and squares[0] != from_sq:
                        continue
                    if piece_type and piece.piece_type != piece_type:
                        continue
                    return {
                        "success": True,
                        "uci_move": move.uci(),
                        "interpretation": spoken_move,
                    }

            if piece_type and piece.piece_type == piece_type:
                if to_sq in spoken_lower:
                    return {
                        "success": True,
                        "uci_move": move.uci(),
                        "interpretation": spoken_move,
                    }

        return {
            "success": False,
            "error": f"Could not parse '{spoken_move}' as a legal move",
            "uci_move": None,
        }

    async def _arun(self, *args, **kwargs) -> dict:
        """Async version."""
        return self._run(*args, **kwargs)
