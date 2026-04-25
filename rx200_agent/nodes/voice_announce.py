# rx200_agent/nodes/voice_announce.py
"""
Voice announcement node - speaks pending responses via TTS.
"""

import os
from typing import Optional

from ..state import AgentState
from ..tools.tts_tool import TTSTool


# Global TTS tool instance
_tts_tool: Optional[TTSTool] = None

# Mode to voice mapping
MODE_VOICES = {
    "coach": "nova",       # Friendly, warm
    "friend": "alloy",     # Neutral, casual
    "opponent": "onyx",    # Deep, serious
    "mean_friend": "echo", # Playful
}


def get_tts_tool(voice: str = "alloy") -> TTSTool:
    """Get or create TTS tool instance."""
    global _tts_tool
    if _tts_tool is None:
        _tts_tool = TTSTool(voice=voice)
    return _tts_tool


def voice_announce(state: AgentState) -> dict:
    """
    Process and speak pending voice response via TTS.

    This node checks for pending_voice_response in state,
    generates audio via TTS, and clears the pending response.

    Returns:
        Updated state with cleared pending response
    """
    pending = state.get("pending_voice_response")
    voice_enabled = state.get("voice_enabled", False)

    if not pending:
        return {"current_phase": "voice_announce"}

    # Get voice based on agent mode
    agent_mode = state.get("agent_mode", "friend")
    voice = MODE_VOICES.get(agent_mode, "alloy")

    if voice_enabled:
        try:
            tts = get_tts_tool(voice)
            # Update voice if mode changed
            tts.voice = voice

            result = tts._run(text=pending, voice=voice)

            if result.get("success"):
                print(f"[Voice] Speaking: {pending[:50]}...")
            else:
                print(f"[Voice] TTS failed: {result.get('error')}")

        except Exception as e:
            print(f"[Voice] Error: {e}")
    else:
        # Voice disabled - just print the message
        print(f"[Agent] {pending}")

    # Update voice context history
    voice_context = state.get("voice_context", []).copy()
    voice_context.append({
        "type": "agent",
        "text": pending,
        "phase": state.get("current_phase", "unknown"),
    })

    return {
        "pending_voice_response": None,
        "voice_context": voice_context,
        "current_phase": "voice_announce",
    }


def generate_move_announcement(
    move_uci: str,
    fen: str,
    is_capture: bool = False,
    captured_piece: Optional[str] = None,
    is_check: bool = False,
    agent_mode: str = "friend",
) -> str:
    """
    Generate a natural language announcement for a chess move.

    Args:
        move_uci: Move in UCI format (e.g., "e2e4")
        fen: Board FEN before the move
        is_capture: Whether this is a capture
        captured_piece: Name of captured piece if capture
        is_check: Whether this move gives check
        agent_mode: Current agent personality mode

    Returns:
        Natural language announcement string
    """
    import chess

    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
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
    to_sq = chess.square_name(move.to_square)

    # Check for castling
    if board.is_castling(move):
        if move.to_square > move.from_square:
            base = "Castling kingside"
        else:
            base = "Castling queenside"
    elif is_capture and captured_piece:
        # Clean up captured piece name (e.g., "black_pawn" -> "pawn")
        cap_name = captured_piece.split("_")[-1] if "_" in captured_piece else captured_piece
        base = f"Taking your {cap_name} with my {piece_name}"
    else:
        base = f"Moving my {piece_name} to {to_sq}"

    # Add check announcement
    if is_check:
        base += ". Check!"

    # Add personality flair based on mode
    if agent_mode == "coach":
        if is_capture:
            base = f"I'm {base.lower()}. Notice how this improves my position."
        elif is_check:
            base = base  # Keep as is
        else:
            base = f"I'm {base.lower()}."
    elif agent_mode == "mean_friend":
        if is_capture:
            base = f"Ha! {base}. Thanks for the free piece!"
        elif is_check:
            base = f"{base} Feeling the pressure?"
        else:
            base = f"{base}. Let's see how you handle this."
    elif agent_mode == "opponent":
        base = base  # Keep terse
    else:  # friend
        if is_capture:
            base = f"I'll take that! {base}."
        else:
            base = f"{base}."

    return base


def generate_human_move_response(
    move: dict,
    agent_mode: str = "friend",
    is_check: bool = False,
) -> str:
    """
    Generate a response to the human's move.

    Args:
        move: ChessMove dict with from_square, to_square, piece, is_capture, etc.
        agent_mode: Current agent personality mode
        is_check: Whether the human's move gives check

    Returns:
        Natural language response string
    """
    piece = move.get("piece", "piece")
    # Clean up piece name (e.g., "white_knight" -> "knight")
    piece_name = piece.split("_")[-1] if "_" in piece else piece
    from_sq = move.get("from_square", "")
    to_sq = move.get("to_square", "")
    is_capture = move.get("is_capture", False)
    captured = move.get("captured_piece")

    # Base description
    if is_capture and captured:
        cap_name = captured.split("_")[-1] if "_" in captured else captured
        base = f"You captured my {cap_name} with your {piece_name}"
    else:
        base = f"You moved your {piece_name} to {to_sq}"

    # Add check response
    if is_check:
        base += ". I'm in check!"

    # Add personality
    if agent_mode == "coach":
        if is_capture:
            return f"{base}. Good capture! Let me think about how to respond."
        elif is_check:
            return f"{base} Let me find the best way to get out of this."
        else:
            return f"Interesting! {base}. Let me analyze this position."
    elif agent_mode == "mean_friend":
        if is_capture:
            return f"Ugh, {base}. Lucky shot! But I've got something for you..."
        elif is_check:
            return f"{base} Oh please, like that's gonna stop me!"
        else:
            return f"{base}. That's what you're going with? Okay..."
    elif agent_mode == "opponent":
        if is_capture:
            return f"{base}."
        elif is_check:
            return "Noted."
        else:
            return "Okay."
    else:  # friend
        if is_capture:
            return f"Ooh, {base}! Nice one."
        elif is_check:
            return f"{base} Good move!"
        else:
            return f"{base}. Let me see what I can do."
