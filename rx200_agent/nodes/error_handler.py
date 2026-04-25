# rx200_agent/nodes/error_handler.py
"""
Error handling nodes for various failure scenarios.
"""

import time
from typing import Optional

from ..state import AgentState
from ..config import VISION_RETRY_LIMIT


def handle_vision_error(state: AgentState) -> dict:
    """
    Handle vision/camera errors.

    Strategies:
    - If under retry limit: wait and retry
    - If over limit: pause game and ask for help
    """
    failures = state.get("consecutive_vision_failures", 0)
    error = state.get("vision_error", "Unknown vision error")

    print(f"\n[Error] Vision failure ({failures}/{VISION_RETRY_LIMIT}): {error}")

    if failures < VISION_RETRY_LIMIT:
        print("[Agent] Retrying in 2 seconds...")
        time.sleep(2)
        return {
            "current_phase": "observe",  # Retry observation
            "vision_error": None,
        }
    else:
        print("\n[Agent] Too many vision failures. Please check:")
        print("  1. Is the camera connected?")
        print("  2. Is the chessboard visible and well-lit?")
        print("  3. Are there any obstructions?")
        print("\n[Agent] Pausing — resume via /game/resume when ready.")

        return {
            "paused": True,
            "pause_reason": "vision_failures_exceeded",
            "pause_options": ["retry"],
            "consecutive_vision_failures": 0,
            "vision_error": None,
            "current_phase": "observe",
        }


def handle_robot_error(state: AgentState) -> dict:
    """
    Handle robot movement errors.

    Strategies:
    - Log the error
    - Ask for manual intervention
    - Allow retry or skip
    """
    error = state.get("robot_error", "Unknown robot error")
    move = state.get("suggested_move", "unknown")

    print(f"\n[Error] Robot failed to execute move {move}: {error}")
    print("\nOptions: retry / skip / abort")
    print("[Agent] Pausing — resume via /game/resume with chosen action.")

    user_action = state.get("user_action")

    if user_action == "retry":
        return {
            "robot_error": None,
            "current_phase": "act",
            "paused": False,
            "pause_reason": None,
            "pause_options": None,
            "user_action": None,
        }
    elif user_action == "skip":
        # Skip - assume human moved the piece manually
        game = state["game"].copy()
        game["whose_turn"] = "human"

        return {
            "game": game,
            "robot_error": None,
            "current_phase": "verify",
            "paused": False,
            "pause_reason": None,
            "pause_options": None,
            "user_action": None,
        }
    elif user_action == "abort":
        return {
            "should_continue": False,
            "current_phase": "end",
            "error_message": "Game aborted by user after robot error",
            "paused": False,
            "pause_reason": None,
            "pause_options": None,
            "user_action": None,
        }
    else:
        # No action yet — pause and wait
        return {
            "paused": True,
            "pause_reason": "robot_error",
            "pause_options": ["retry", "skip", "abort"],
            "robot_error": error,
        }


def handle_invalid_position(state: AgentState) -> dict:
    """
    Handle invalid board position detected.

    This might happen if:
    - Pieces are knocked over
    - Board is not set up correctly
    - Vision misdetection
    """
    board = state.get("current_board", {})
    warnings = board.get("warnings", [])

    print("\n[Error] Invalid board position detected!")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")

    print("\n[Agent] Please correct the board position.")
    print("[Agent] Pausing — resume via /game/resume when ready.")

    return {
        "paused": True,
        "pause_reason": "invalid_board_position",
        "pause_options": ["retry"],
        "current_phase": "observe",
    }


def handle_timeout(state: AgentState) -> dict:
    """
    Handle timeout waiting for human move.

    Pauses the game so the UI can ask the user what to do.
    """
    msg = state.get("error_message", "Timeout waiting for human move")
    print(f"\n[Error] {msg}")
    print("[Agent] Pausing — resume via /game/resume to keep waiting or abort.")

    user_action = state.get("user_action")

    if user_action == "continue":
        # User wants to keep waiting
        return {
            "error_type": None,
            "error_message": None,
            "current_phase": "wait_human",
            "paused": False,
            "pause_reason": None,
            "pause_options": None,
            "user_action": None,
        }
    elif user_action == "abort":
        return {
            "should_continue": False,
            "current_phase": "end",
            "error_message": "Game aborted after timeout",
            "paused": False,
            "pause_reason": None,
            "pause_options": None,
            "user_action": None,
        }
    else:
        # No action yet — pause
        return {
            "paused": True,
            "pause_reason": "timeout_waiting_for_human",
            "pause_options": ["continue", "abort"],
        }


def handle_game_end(state: AgentState) -> dict:
    """
    Handle game ending conditions.
    """
    game = state["game"]
    status = game.get("game_status", "playing")

    if status == "checkmate":
        # Determine winner
        current_fen = game["current_fen"]
        # In checkmate, the side to move has lost
        side_to_move = current_fen.split()[1]
        if side_to_move == "w":
            winner = "Black"
        else:
            winner = "White"

        robot_color = game["robot_color"]
        if winner.lower() == robot_color:
            print("\n[Agent] Checkmate! I win!")
        else:
            print("\n[Agent] Checkmate! You win! Well played!")

    elif status == "stalemate":
        print("\n[Agent] Stalemate! The game is a draw.")

    elif status == "draw":
        print("\n[Agent] The game is a draw.")

    return {
        "should_continue": False,
        "current_phase": "end",
    }


def check_game_status(state: AgentState) -> str:
    """
    Check if the game should continue.

    Returns:
        "continue" or "end" for routing
    """
    game = state.get("game", {})
    status = game.get("game_status", "playing")

    if status != "playing":
        return "end"

    if not state.get("should_continue", True):
        return "end"

    return "continue"
