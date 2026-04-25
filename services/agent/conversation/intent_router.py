"""
Intent router for classifying user input and routing to appropriate handlers.
"""

import re
from enum import Enum
from typing import Optional, Tuple
from dataclasses import dataclass

import chess

from .modes import AgentMode, detect_mode_from_text, detect_language_from_text


class Intent(Enum):
    """Types of user intents."""
    MOVE = "move"                    # Chess move (e.g., "pawn to e4")
    MODE_SWITCH = "mode_switch"      # Switch agent mode
    LANGUAGE_SWITCH = "language_switch"  # Switch language
    GAME_COMMAND = "game_command"    # Game control (status, resign, etc.)
    CONVERSATION = "conversation"    # General conversation
    IGNORE = "ignore"                # Filler words, noise, hallucinations


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent: Intent
    confidence: float
    data: Optional[dict] = None


class IntentRouter:
    """
    Routes user input to the appropriate handler.

    Classifies whether input is:
    - A chess move
    - A mode switch request
    - A language switch request
    - A game command
    - General conversation
    """

    # Game command keywords (dual-language)
    GAME_COMMANDS = {
        "status": [
            "status", "what's the position", "where are we", "board state",
            "目前局面", "現在狀態", "棋盤狀態",
        ],
        "resign": [
            "resign", "i give up", "i surrender",
            "投降", "我認輸", "我放棄",
        ],
        "draw": [
            "offer draw", "draw", "let's call it a draw",
            "和棋", "提議和棋", "算平手",
        ],
        "new_game": [
            "new game", "start over", "reset", "play again", "start a game", "let's play",
            "新局", "重新開始", "重來", "再來一局", "開始遊戲", "來下棋",
        ],
        "help": [
            "help", "what can i say", "commands",
            "幫助", "指令", "我可以說什麼",
        ],
        "undo": [
            "undo", "take back", "go back",
            "悔棋", "退回", "收回",
        ],
        "stop": [
            "stop the game", "end the game",
            "結束遊戲", "不下了",
        ],
        "your_turn": [
            "your turn", "you're up", "your move", "i moved", "i'm done",
            "take your turn", "my move is done", "i've moved",
            "done", "okay go", "ok go", "go ahead",
            "i made my move", "your go", "now you", "it's your turn",
            "換你", "你走", "輪到你", "我走了", "我下了", "我走好了",
            "走好了", "好了", "下好了", "你的回合", "輪你了",
        ],
    }

    # Common Whisper hallucinations and filler words to ignore
    IGNORE_PHRASES = {
        # English
        "thank you", "thanks", "bye", "goodbye", "you", "the", "i",
        "thank you for watching", "thanks for watching",
        "subscribe", "like and subscribe", "see you next time",
        "silence", "...", "hmm", "um", "uh", "ah", "oh", "huh",
        "you're welcome", "okay", "ok",
        # Chinese Whisper hallucinations and filler
        "謝謝", "嗯", "啊", "喔", "欸", "呃",
        "謝謝觀看", "請訂閱", "感謝收看",
        "好的", "好",
    }

    # Intent to priority mapping (used by voice event system)
    INTENT_PRIORITY = {
        Intent.GAME_COMMAND: "HIGH",
        Intent.MODE_SWITCH: "NORMAL",
        Intent.LANGUAGE_SWITCH: "NORMAL",
        Intent.MOVE: "NORMAL",
        Intent.CONVERSATION: "LOW",
        Intent.IGNORE: "IGNORE",
    }

    # Move-related keywords (dual-language)
    MOVE_KEYWORDS = [
        # English
        "pawn", "knight", "bishop", "rook", "queen", "king",
        "castle", "castling", "takes", "capture", "to",
        "move", "play", "horse",
        # Chinese
        "兵", "馬", "象", "車", "后", "王", "國王", "皇后",
        "城堡", "入堡", "王翼", "后翼", "吃", "走",
    ]

    # Square pattern
    SQUARE_PATTERN = re.compile(r"[a-h][1-8]")

    def __init__(self, language: str = "en"):
        self.language = language

    def set_language(self, language: str) -> None:
        """Update the current language for intent classification."""
        self.language = language

    def classify(self, text: str, board: Optional[chess.Board] = None) -> IntentResult:
        """
        Classify user input into an intent category.

        Args:
            text: User input text
            board: Current chess board for move validation

        Returns:
            IntentResult with intent type and any extracted data
        """
        text_lower = text.lower().strip()

        # Check for filler/noise/hallucinations first
        if self._should_ignore(text_lower):
            return IntentResult(
                intent=Intent.IGNORE,
                confidence=0.95,
                data={"reason": "filler_or_noise"},
            )

        # Check for language switch
        new_lang = detect_language_from_text(text_lower)
        if new_lang is not None:
            return IntentResult(
                intent=Intent.LANGUAGE_SWITCH,
                confidence=0.95,
                data={"language": new_lang},
            )

        # Check for mode switch (checks both languages)
        mode = detect_mode_from_text(text_lower, self.language)
        if mode is not None:
            return IntentResult(
                intent=Intent.MODE_SWITCH,
                confidence=0.95,
                data={"mode": mode},
            )

        # Check for game commands
        command = self._detect_game_command(text_lower)
        if command:
            return IntentResult(
                intent=Intent.GAME_COMMAND,
                confidence=0.9,
                data={"command": command},
            )

        # Check for chess move
        if board is not None:
            move = self._detect_move(text_lower, board)
            if move:
                return IntentResult(
                    intent=Intent.MOVE,
                    confidence=0.85,
                    data={"move": move, "original_text": text},
                )

        # Check if it looks like a move attempt (has move keywords or squares)
        if self._looks_like_move_attempt(text_lower):
            return IntentResult(
                intent=Intent.MOVE,
                confidence=0.6,
                data={"original_text": text, "move": None},
            )

        # Default to conversation
        return IntentResult(
            intent=Intent.CONVERSATION,
            confidence=0.8,
            data={"message": text},
        )

    def _should_ignore(self, text: str) -> bool:
        """Check if text is filler, noise, or a Whisper hallucination."""
        # Very short utterances (1-2 chars)
        if len(text) <= 2:
            return True
        # Exact match against known ignore phrases
        if text in self.IGNORE_PHRASES:
            return True
        # Only punctuation or whitespace
        stripped = re.sub(r"[^\w]", "", text)
        if len(stripped) <= 1:
            return True
        return False

    def _detect_game_command(self, text: str) -> Optional[str]:
        """Detect if text is a game command."""
        for command, keywords in self.GAME_COMMANDS.items():
            for keyword in keywords:
                if keyword in text:
                    return command
        return None

    def _detect_move(self, text: str, board: chess.Board) -> Optional[str]:
        """
        Try to parse text as a chess move.

        Returns UCI move string if valid, None otherwise.
        """
        return self._parse_spoken_move(text, board)

    def _looks_like_move_attempt(self, text: str) -> bool:
        """Check if text appears to be a move attempt."""
        # Has square notation
        if self.SQUARE_PATTERN.search(text):
            return True

        # Has move keywords
        for keyword in self.MOVE_KEYWORDS:
            if keyword in text:
                return True

        return False

    def _parse_spoken_move(self, spoken: str, board: chess.Board) -> Optional[str]:
        """
        Parse natural language chess move to UCI format.

        Supports both English and Chinese piece names.
        """
        spoken_lower = spoken.lower().strip()

        # Handle castling (English + Chinese)
        castling_words = ["castle", "castling", "入堡", "城堡"]
        if any(w in spoken_lower for w in castling_words):
            kingside_words = ["king", "short", "o-o", "0-0", "王翼", "短"]
            queenside_words = ["queen", "long", "o-o-o", "0-0-0", "后翼", "長"]
            if any(w in spoken_lower for w in kingside_words):
                uci = "e1g1" if board.turn == chess.WHITE else "e8g8"
            elif any(w in spoken_lower for w in queenside_words):
                uci = "e1c1" if board.turn == chess.WHITE else "e8c8"
            else:
                return None

            move = chess.Move.from_uci(uci)
            return uci if move in board.legal_moves else None

        # Extract squares
        squares = self.SQUARE_PATTERN.findall(spoken_lower)

        # Piece type mapping (English + Chinese)
        piece_map = {
            "king": chess.KING, "國王": chess.KING, "王": chess.KING,
            "queen": chess.QUEEN, "皇后": chess.QUEEN, "后": chess.QUEEN,
            "rook": chess.ROOK, "車": chess.ROOK,
            "bishop": chess.BISHOP, "象": chess.BISHOP,
            "knight": chess.KNIGHT, "horse": chess.KNIGHT, "馬": chess.KNIGHT,
            "pawn": chess.PAWN, "兵": chess.PAWN,
        }

        piece_type = None
        for name, ptype in piece_map.items():
            if name in spoken_lower:
                piece_type = ptype
                break

        capture_words = ["take", "takes", "capture", "x", "吃"]
        is_capture = any(w in spoken_lower for w in capture_words)

        # Two squares mentioned
        if len(squares) == 2:
            from_sq, to_sq = squares
            try:
                uci = from_sq + to_sq
                move = chess.Move.from_uci(uci)
                if move in board.legal_moves:
                    piece = board.piece_at(move.from_square)
                    if piece_type is None or (piece and piece.piece_type == piece_type):
                        return uci
            except ValueError:
                pass

        # One square (destination)
        if len(squares) >= 1:
            target_sq = squares[-1]

            candidates = []
            for move in board.legal_moves:
                to_sq = chess.square_name(move.to_square)
                if to_sq != target_sq:
                    continue

                piece = board.piece_at(move.from_square)
                if piece is None:
                    continue

                if piece_type is not None and piece.piece_type != piece_type:
                    continue

                if is_capture and not board.is_capture(move):
                    continue

                candidates.append(move)

            if len(candidates) == 1:
                return candidates[0].uci()

            if len(candidates) > 1 and len(squares) == 2:
                from_sq = squares[0]
                for move in candidates:
                    if chess.square_name(move.from_square) == from_sq:
                        return move.uci()

        # Piece + destination
        if piece_type is not None:
            for move in board.legal_moves:
                piece = board.piece_at(move.from_square)
                if piece is None or piece.piece_type != piece_type:
                    continue

                to_sq = chess.square_name(move.to_square)
                if to_sq in spoken_lower:
                    return move.uci()

        return None

    def extract_mode(self, text: str) -> Optional[AgentMode]:
        """Extract mode from text if present."""
        return detect_mode_from_text(text, self.language)

    def get_help_text(self, language: str = "en") -> str:
        """Return help text about available commands."""
        if language == "zh-TW":
            return """
可用的語音指令：

西洋棋走法：
  - 「兵到e4」或「e2到e4」
  - 「馬f3」或「馬到f3」
  - 「皇后吃d7」
  - 「王翼入堡」或「后翼入堡」

模式切換：
  - 「教練模式」— 獲得教學和建議
  - 「朋友模式」— 輕鬆對局
  - 「對手模式」— 認真競爭
  - 「毒舌模式」— 互相嘴砲

語言切換：
  - 「切換中文」— 繁體中文
  - 「Switch to English」— 英文

遊戲指令：
  - 「目前局面」— 查看目前棋盤
  - 「幫助」— 顯示指令說明
  - 「新局」— 重新開始
  - 「投降」— 放棄這局

或者直接聊天！問問題、討論策略，或一起玩。
"""
        return """
Available voice commands:

Chess Moves:
  - "Pawn to e4" or "e2 to e4"
  - "Knight f3" or "Horse to f3"
  - "Queen takes d7"
  - "Castle kingside" or "Castle queenside"

Mode Switching:
  - "Switch to coach mode" - Get teaching and advice
  - "Friend mode" or "Be my friend" - Casual game
  - "Opponent mode" or "Let's compete" - Competitive
  - "Trash talk mode" or "Mean friend" - Playful roasts

Language Switching:
  - "Switch to Chinese" - Traditional Chinese
  - "切換中文" - 繁體中文

Game Commands:
  - "Status" - Get current position info
  - "Help" - Show this help
  - "New game" - Start over
  - "Resign" - Give up the game

Or just chat naturally! Ask questions, discuss strategy, or have fun.
"""
