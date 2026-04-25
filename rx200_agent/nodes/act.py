# rx200_agent/nodes/act.py
"""
Act node - executes the robot's move.
"""

import chess
from typing import Optional

from ..state import AgentState, ChessMove
from ..tools.robot_tool import RobotTool
from ..board_tracker import BoardTracker
from ..utils.move_parser import (
    parse_uci_move,
    is_capture_move,
    get_captured_piece,
    get_game_status,
    is_in_check,
)
from .voice_announce import generate_move_announcement


# Global robot tool instance
_robot_tool = None


def get_robot_tool() -> RobotTool:
    """Get or create robot tool instance."""
    global _robot_tool
    if _robot_tool is None:
        _robot_tool = RobotTool(lazy_init=True)
    return _robot_tool


def execute_robot_move(state: AgentState) -> dict:
    """
    Execute the suggested move with the robot arm.

    This node:
    1. Gets the suggested move from state
    2. Commands the robot to execute the move
    3. Updates game state with the move
    4. Handles any robot errors

    Returns:
        Updated state fields
    """
    suggested_move = state.get("suggested_move")
    game = state["game"]
    current_fen = game["current_fen"]

    if not suggested_move:
        return {
            "error_type": "no_move",
            "error_message": "No move suggested",
            "current_phase": "error",
        }

    print(f"\n[Agent] Executing move: {suggested_move}")

    # Parse the move
    from_sq, to_sq, promotion = parse_uci_move(suggested_move)

    # Check if capture
    is_capture = is_capture_move(current_fen, suggested_move)
    captured_piece = get_captured_piece(current_fen, suggested_move) if is_capture else None

    # Execute with robot
    robot = get_robot_tool()
    result = robot._run(uci_move=suggested_move, fen=current_fen)

    if not result["success"]:
        print(f"[Error] Robot failed: {result.get('error')}")
        return {
            "robot_error": result.get("error"),
            "robot_busy": False,
            "error_type": "robot",
            "error_message": result.get("error"),
            "current_phase": "error",
        }

    print(f"[Agent] Move executed successfully!")

    # Calculate new position
    board = chess.Board(current_fen)
    move = chess.Move.from_uci(suggested_move)
    board.push(move)
    new_fen = board.fen()

    # Update board tracker
    tracker = None
    tracker_data = state.get("board_tracker_state")
    if tracker_data:
        tracker = BoardTracker.from_dict(tracker_data)
        tracker.push_uci(suggested_move)

    # Determine piece that moved
    # Get piece info from before the move
    old_board = chess.Board(current_fen)
    piece = old_board.piece_at(move.from_square)
    piece_color = "white" if piece.color == chess.WHITE else "black"
    piece_name = chess.piece_name(piece.piece_type)
    piece_full = f"{piece_color}_{piece_name}"

    # Create move record
    robot_move: ChessMove = {
        "from_square": from_sq,
        "to_square": to_sq,
        "piece": piece_full,
        "is_capture": is_capture,
        "captured_piece": captured_piece,
        "promotion": promotion,
    }

    # Update game state
    new_game = game.copy()
    new_game["previous_fen"] = current_fen
    new_game["current_fen"] = new_fen
    new_game["last_robot_move"] = robot_move
    new_game["whose_turn"] = "human"
    new_game["move_number"] = game["move_number"] + 1
    new_game["is_check"] = is_in_check(new_fen)
    new_game["game_status"] = get_game_status(new_fen)

    # Add to history
    history = game.get("board_history", []).copy()
    history.append(current_fen)
    new_game["board_history"] = history

    # Generate voice announcement for the move
    agent_mode = state.get("agent_mode", "friend")
    voice_text = generate_move_announcement(
        move_uci=suggested_move,
        fen=current_fen,
        is_capture=is_capture,
        captured_piece=captured_piece,
        is_check=new_game["is_check"],
        agent_mode=agent_mode,
    )

    result = {
        "game": new_game,
        "robot_busy": False,
        "robot_error": None,
        "last_robot_action": suggested_move,
        "current_phase": "act",
        "pending_voice_response": voice_text,
    }
    if tracker:
        result["board_tracker_state"] = tracker.to_dict()
    return result


def verify_robot_move(state: AgentState) -> dict:
    """
    Verify the robot's move was executed correctly.

    This node observes the board after the robot move
    and confirms the position matches expectations.

    Returns:
        Updated state with verification result
    """
    # Import here to avoid circular dependency
    from .observe import observe_board

    # Observe the board
    obs_result = observe_board(state)

    if obs_result.get("vision_error"):
        print("[Warning] Could not verify move - vision error")
        return {
            **obs_result,
            "current_phase": "verify",
        }

    # Compare observed FEN with expected FEN
    expected_fen = state["game"]["current_fen"]
    observed_board = obs_result.get("current_board")

    if observed_board:
        observed_fen = observed_board["fen"]
        expected_position = expected_fen.split()[0]
        observed_position = observed_fen.split()[0]

        if expected_position == observed_position:
            print("[Agent] Move verified successfully!")
        else:
            print("[Warning] Board position doesn't match expected state")
            print(f"  Expected: {expected_position}")
            print(f"  Observed: {observed_position}")

    # Validate tracker vs vision (advisory - tracker is authoritative)
    tracker_data = state.get("board_tracker_state")
    tracker_discrepancies = []
    if tracker_data and observed_board:
        tracker = BoardTracker.from_dict(tracker_data)
        vision_pieces = observed_board.get("piece_positions", {})
        if vision_pieces:
            discrepancies = tracker.validate_against_vision(vision_pieces)
            if discrepancies:
                print(f"[Agent] Tracker vs vision: {len(discrepancies)} discrepancies (tracker authoritative)")
                for d in discrepancies:
                    print(f"  {d.square}: {d.type.value} - tracker={d.tracker_piece}, vision={d.vision_piece}")
                tracker_discrepancies = [
                    {"square": d.square, "type": d.type.value,
                     "tracker_piece": d.tracker_piece, "vision_piece": d.vision_piece}
                    for d in discrepancies
                ]

    return {
        **obs_result,
        "current_phase": "verify",
        "tracker_discrepancies": tracker_discrepancies,
    }
