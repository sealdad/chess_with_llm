# tests/test_game_flow.py
"""
Mock integration test: Scholar's Mate game flow.

Tests the state machine transitions and BoardTracker through a
complete game without requiring actual hardware services.
"""

import chess
import pytest
from rx200_agent.board_tracker import BoardTracker
from rx200_agent.state import create_initial_state, AgentState


def check_game_status(state):
    """Mirror of error_handler.check_game_status (avoids heavy imports)."""
    game = state.get("game", {})
    status = game.get("game_status", "playing")
    if status != "playing":
        return "end"
    if not state.get("should_continue", True):
        return "end"
    return "continue"


def _occupancy(board):
    return {chess.square_name(sq) for sq in board.piece_map().keys()}


def _simulate_human_move(tracker, uci_move):
    """Simulate a human move via occupancy-based detection."""
    prev = tracker.get_occupancy()
    board_copy = tracker.board.copy()
    board_copy.push(chess.Move.from_uci(uci_move))
    curr = _occupancy(board_copy)

    result = tracker.detect_human_move(prev, curr)
    assert result.success, f"Failed to detect {uci_move}: {result.error}"
    tracker.push_uci(result.uci_move)
    return result


def _simulate_robot_move(tracker, uci_move):
    """Robot move: directly push to tracker."""
    tracker.push_uci(uci_move)


class TestScholarsMate:
    """
    Scholar's Mate:
        1. e4   e5
        2. Bc4  Nc6
        3. Qh5  Nf6??
        4. Qxf7#
    """

    def test_full_game(self):
        tracker = BoardTracker()
        state = create_initial_state(robot_color="white")

        # Robot (white) plays e4
        _simulate_robot_move(tracker, "e2e4")
        state["game"]["current_fen"] = tracker.fen
        state["game"]["whose_turn"] = "human"
        assert tracker.turn == "black"

        # Human (black) plays e5
        result = _simulate_human_move(tracker, "e7e5")
        assert result.piece == "black_pawn"
        state["game"]["current_fen"] = tracker.fen
        state["game"]["whose_turn"] = "robot"

        # Robot plays Bc4
        _simulate_robot_move(tracker, "f1c4")
        state["game"]["current_fen"] = tracker.fen
        state["game"]["whose_turn"] = "human"

        # Human plays Nc6
        result = _simulate_human_move(tracker, "b8c6")
        assert result.piece == "black_knight"
        state["game"]["current_fen"] = tracker.fen
        state["game"]["whose_turn"] = "robot"

        # Robot plays Qh5
        _simulate_robot_move(tracker, "d1h5")
        state["game"]["current_fen"] = tracker.fen
        state["game"]["whose_turn"] = "human"

        # Human plays Nf6 (blunder)
        result = _simulate_human_move(tracker, "g8f6")
        assert result.piece == "black_knight"
        state["game"]["current_fen"] = tracker.fen
        state["game"]["whose_turn"] = "robot"

        # Robot plays Qxf7# (checkmate)
        _simulate_robot_move(tracker, "h5f7")
        state["game"]["current_fen"] = tracker.fen

        # Verify checkmate
        board = chess.Board(tracker.fen)
        assert board.is_checkmate()

        # Verify game status routing
        state["game"]["game_status"] = "checkmate"
        assert check_game_status(state) == "end"

    def test_move_count(self):
        """Verify correct number of moves after full game."""
        tracker = BoardTracker()
        moves = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]
        for m in moves:
            tracker.push_uci(m)
        assert len(tracker.board.move_stack) == 7

    def test_captured_piece(self):
        """Verify Qxf7 captures the f7 pawn."""
        tracker = BoardTracker()
        moves = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]
        for m in moves:
            tracker.push_uci(m)
        assert "black_pawn" in tracker._captured_pieces


class TestGameStatusRouting:
    """Test check_game_status routing function."""

    def test_playing_continues(self, initial_state):
        assert check_game_status(initial_state) == "continue"

    def test_checkmate_ends(self, initial_state):
        initial_state["game"]["game_status"] = "checkmate"
        assert check_game_status(initial_state) == "end"

    def test_stalemate_ends(self, initial_state):
        initial_state["game"]["game_status"] = "stalemate"
        assert check_game_status(initial_state) == "end"

    def test_should_continue_false_ends(self, initial_state):
        initial_state["should_continue"] = False
        assert check_game_status(initial_state) == "end"


class TestPauseState:
    """Test the new pause fields in state."""

    def test_initial_state_not_paused(self, initial_state):
        assert initial_state["paused"] is False
        assert initial_state["pause_reason"] is None
        assert initial_state["user_action"] is None

    def test_pause_fields_exist(self, initial_state):
        assert "paused" in initial_state
        assert "pause_reason" in initial_state
        assert "pause_options" in initial_state
        assert "user_action" in initial_state
