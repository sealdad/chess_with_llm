# rx200_agent/nodes/observe.py
"""
Observation node - captures and analyzes the board state.
"""

import time
from typing import Any

from ..state import AgentState, BoardSnapshot
from ..tools.vision_tool import VisionTool


# Global vision tool instance (shared across calls)
_vision_tool = None


def get_vision_tool() -> VisionTool:
    """Get or create the vision tool instance."""
    global _vision_tool
    if _vision_tool is None:
        _vision_tool = VisionTool(lazy_init=True)
    return _vision_tool


def observe_board(state: AgentState) -> dict:
    """
    Capture image and analyze the board state.

    This node:
    1. Captures a frame from the camera
    2. Runs the vision pipeline to detect pieces
    3. Updates state with the new board snapshot
    4. Handles vision failures

    Returns:
        Updated state fields
    """
    vision_tool = get_vision_tool()

    # Capture and analyze
    result = vision_tool._run()

    if not result["success"]:
        # Vision failed
        failures = state.get("consecutive_vision_failures", 0) + 1
        return {
            "vision_error": result.get("error", "Unknown vision error"),
            "consecutive_vision_failures": failures,
            "current_phase": "error" if failures >= 3 else "observe",
            "error_type": "vision" if failures >= 3 else None,
            "error_message": result.get("error") if failures >= 3 else None,
        }

    # Create board snapshot
    board_snapshot: BoardSnapshot = {
        "fen": result["fen"],
        "piece_positions": result["piece_positions"],
        "ascii_board": result["ascii_board"],
        "timestamp": result["timestamp"],
        "is_valid": result["is_valid"],
        "warnings": result.get("warnings", []),
    }

    # Update state
    updates = {
        "previous_board": state.get("current_board"),
        "current_board": board_snapshot,
        "vision_error": None,
        "consecutive_vision_failures": 0,
        "current_phase": "observe",
    }

    # Update game FEN if valid
    if result["is_valid"]:
        game = state.get("game", {}).copy()
        game["previous_fen"] = game.get("current_fen")
        game["current_fen"] = result["fen"]
        updates["game"] = game

    return updates


def check_board_validity(state: AgentState) -> str:
    """
    Check if the observed board state is valid.

    Returns:
        "valid" or "error" for routing
    """
    current_board = state.get("current_board")

    if current_board is None:
        return "error"

    if not current_board.get("is_valid", False):
        return "error"

    return "valid"
