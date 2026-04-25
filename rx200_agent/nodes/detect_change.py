# rx200_agent/nodes/detect_change.py
"""
Human move detection node - detects when human has made a move.

Uses BoardTracker for occupancy-based move detection instead of
FEN diff. Vision provides occupancy signals; the tracker knows
what pieces are where.
"""

import time
from typing import Optional, Set

from ..state import AgentState, ChessMove, BoardSnapshot
from ..tools.vision_tool import VisionTool
from ..board_tracker import BoardTracker, MoveDetectionResult
from ..utils.move_parser import get_game_status, is_in_check
from ..config import POLL_INTERVAL, STABILITY_THRESHOLD, MAX_WAIT_TIME
from .voice_announce import generate_human_move_response


# Global vision tool instance
_vision_tool = None


def get_vision_tool() -> VisionTool:
    """Get or create the vision tool instance."""
    global _vision_tool
    if _vision_tool is None:
        _vision_tool = VisionTool(lazy_init=True)
    return _vision_tool


def _get_occupancy_from_vision(result: dict) -> Set[str]:
    """Extract occupied square names from vision result."""
    piece_positions = result.get("piece_positions", {})
    return set(piece_positions.keys())


def _restore_tracker(state: AgentState) -> Optional[BoardTracker]:
    """Restore BoardTracker from state."""
    tracker_data = state.get("board_tracker_state")
    if tracker_data is None:
        return None
    return BoardTracker.from_dict(tracker_data)


def wait_for_human_move(state: AgentState) -> dict:
    """
    Wait for human to make a move (polling mode).

    Uses occupancy-based detection with the BoardTracker:
    1. Get baseline occupancy from tracker
    2. Poll vision for current occupancy
    3. On stable change, use tracker.detect_human_move() to match legal moves
    4. Apply matched move via tracker.push_move()

    Returns:
        Updated state fields
    """
    vision_tool = get_vision_tool()

    # Restore tracker from state
    tracker = _restore_tracker(state)
    if tracker is None:
        return {
            "error_type": "tracker_missing",
            "error_message": "BoardTracker not found in state",
            "current_phase": "error",
        }

    # Baseline occupancy from tracker (authoritative)
    baseline_occupancy = tracker.get_occupancy()

    print("\n[Agent] Waiting for your move...")

    stable_count = 0
    last_candidate_occupancy = None
    last_vision_result = None
    start_time = time.time()

    while time.time() - start_time < MAX_WAIT_TIME:
        # Capture current board
        result = vision_tool._run()

        if not result["success"]:
            time.sleep(POLL_INTERVAL)
            continue

        curr_occupancy = _get_occupancy_from_vision(result)

        # Check if occupancy changed from baseline
        if curr_occupancy == baseline_occupancy:
            stable_count = 0
            last_candidate_occupancy = None
            last_vision_result = None
            time.sleep(POLL_INTERVAL)
            continue

        # Occupancy changed - check stability
        if last_candidate_occupancy is None:
            last_candidate_occupancy = curr_occupancy
            last_vision_result = result
            stable_count = 1
            print("[Agent] Detected board change, confirming...")
        elif curr_occupancy == last_candidate_occupancy:
            stable_count += 1
            last_vision_result = result
        else:
            # Different from last candidate (hand still moving), reset
            last_candidate_occupancy = curr_occupancy
            last_vision_result = result
            stable_count = 1

        # Check if stable enough
        if stable_count >= STABILITY_THRESHOLD:
            print("[Agent] Move confirmed! Detecting move via tracker...")

            # Use tracker to detect the move
            vision_pieces = last_vision_result.get("piece_positions", {})
            detection = tracker.detect_human_move(
                prev_occupancy=baseline_occupancy,
                curr_occupancy=last_candidate_occupancy,
                vision_pieces=vision_pieces,
            )

            if detection.success:
                # Apply move to tracker
                import chess
                move = chess.Move.from_uci(detection.uci_move)
                tracker.push_move(move)

                # Build new FEN from tracker
                new_fen = tracker.fen

                chess_move: ChessMove = {
                    "from_square": detection.uci_move[:2],
                    "to_square": detection.uci_move[2:4],
                    "piece": detection.piece,
                    "is_capture": detection.is_capture,
                    "captured_piece": detection.captured_piece,
                    "promotion": detection.promotion,
                }

                board_snapshot: BoardSnapshot = {
                    "fen": new_fen,
                    "piece_positions": vision_pieces,
                    "ascii_board": last_vision_result.get("ascii_board", ""),
                    "timestamp": last_vision_result.get("timestamp", time.time()),
                    "is_valid": True,
                    "warnings": [],
                }

                # Validate tracker vs vision (advisory)
                discrepancies = tracker.validate_against_vision(vision_pieces)
                if discrepancies:
                    print(f"[Agent] Post-move validation: {len(discrepancies)} discrepancies (tracker authoritative)")
                    for d in discrepancies:
                        print(f"  {d.square}: {d.type.value} - tracker={d.tracker_piece}, vision={d.vision_piece}")

                # Update game state
                game = state["game"].copy()
                game["previous_fen"] = state["game"]["current_fen"]
                game["current_fen"] = new_fen
                game["last_human_move"] = chess_move
                game["whose_turn"] = "robot"
                game["is_check"] = is_in_check(new_fen)
                game["game_status"] = get_game_status(new_fen)

                # Generate voice response
                agent_mode = state.get("agent_mode", "friend")
                voice_text = generate_human_move_response(
                    move=chess_move,
                    agent_mode=agent_mode,
                    is_check=game["is_check"],
                )

                return {
                    "game": game,
                    "previous_board": state.get("current_board"),
                    "current_board": board_snapshot,
                    "human_move_detected": True,
                    "detected_human_move": chess_move,
                    "waiting_for_human": False,
                    "stable_board_count": 0,
                    "current_phase": "detect_change",
                    "pending_voice_response": voice_text,
                    "board_tracker_state": tracker.to_dict(),
                    "tracker_discrepancies": [
                        {"square": d.square, "type": d.type.value,
                         "tracker_piece": d.tracker_piece, "vision_piece": d.vision_piece}
                        for d in discrepancies
                    ],
                }
            else:
                print(f"[Warning] Move detection failed: {detection.error}")
                if detection.candidates:
                    print(f"  Candidates: {detection.candidates}")
                stable_count = 0
                last_candidate_occupancy = None

        time.sleep(POLL_INTERVAL)

    # Timeout
    return {
        "waiting_for_human": False,
        "human_move_detected": False,
        "error_type": "timeout",
        "error_message": "Timeout waiting for human move",
        "current_phase": "error",
    }


def wait_for_human_manual(state: AgentState) -> dict:
    """
    Wait for human to indicate move is complete (manual trigger mode).

    Uses occupancy-based detection with the BoardTracker.
    """
    print("\n[Agent] Your turn. Resume via /game/resume when you've made your move.")

    user_action = state.get("user_action")
    if user_action != "continue":
        return {
            "paused": True,
            "pause_reason": "waiting_for_human_move",
            "pause_options": ["continue"],
        }

    # Clear pause state after resume
    # (will be merged into the final return dict)

    # Restore tracker
    tracker = _restore_tracker(state)
    if tracker is None:
        return {
            "error_type": "tracker_missing",
            "error_message": "BoardTracker not found in state",
            "current_phase": "error",
        }

    baseline_occupancy = tracker.get_occupancy()

    # Capture current board
    vision_tool = get_vision_tool()
    result = vision_tool._run()

    if not result["success"]:
        return {
            "vision_error": result.get("error"),
            "current_phase": "error",
            "error_type": "vision",
        }

    curr_occupancy = _get_occupancy_from_vision(result)
    vision_pieces = result.get("piece_positions", {})

    # Detect move via tracker
    detection = tracker.detect_human_move(
        prev_occupancy=baseline_occupancy,
        curr_occupancy=curr_occupancy,
        vision_pieces=vision_pieces,
    )

    if not detection.success:
        print(f"[Warning] Could not detect move: {detection.error}")
        if detection.candidates:
            print(f"  Candidates: {detection.candidates}")
        return {
            "human_move_detected": False,
            "current_phase": "wait_human",
        }

    # Apply move to tracker
    import chess
    move = chess.Move.from_uci(detection.uci_move)
    tracker.push_move(move)
    new_fen = tracker.fen

    chess_move: ChessMove = {
        "from_square": detection.uci_move[:2],
        "to_square": detection.uci_move[2:4],
        "piece": detection.piece,
        "is_capture": detection.is_capture,
        "captured_piece": detection.captured_piece,
        "promotion": detection.promotion,
    }

    board_snapshot: BoardSnapshot = {
        "fen": new_fen,
        "piece_positions": vision_pieces,
        "ascii_board": result.get("ascii_board", ""),
        "timestamp": result.get("timestamp", time.time()),
        "is_valid": True,
        "warnings": [],
    }

    game = state["game"].copy()
    game["previous_fen"] = state["game"]["current_fen"]
    game["current_fen"] = new_fen
    game["last_human_move"] = chess_move
    game["whose_turn"] = "robot"
    game["is_check"] = is_in_check(new_fen)
    game["game_status"] = get_game_status(new_fen)

    print(f"[Agent] Detected move: {detection.piece} {detection.uci_move[:2]} -> {detection.uci_move[2:4]}")

    # Generate voice response
    agent_mode = state.get("agent_mode", "friend")
    voice_text = generate_human_move_response(
        move=chess_move,
        agent_mode=agent_mode,
        is_check=game["is_check"],
    )

    return {
        "game": game,
        "previous_board": state.get("current_board"),
        "current_board": board_snapshot,
        "human_move_detected": True,
        "detected_human_move": chess_move,
        "waiting_for_human": False,
        "current_phase": "detect_change",
        "pending_voice_response": voice_text,
        "board_tracker_state": tracker.to_dict(),
        "tracker_discrepancies": [],
    }


def detect_human_move(state: AgentState) -> dict:
    """
    Detect human move from board state change.

    This is called after wait_for_human when using polling mode.
    The move detection is already done in wait_for_human,
    so this just confirms and routes.
    """
    if state.get("human_move_detected"):
        move = state.get("detected_human_move")
        if move:
            print(f"[Agent] Human played: {move['piece']} {move['from_square']} -> {move['to_square']}")

    return {
        "current_phase": "detect_change",
    }
