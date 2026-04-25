# rx200_agent/graph.py
"""
LangGraph state machine for the chess robot agent.

Defines the agent graph with nodes and edges for the game loop.
"""

from typing import Literal
from langgraph.graph import StateGraph, END

from .state import AgentState, create_initial_state
from .nodes.observe import observe_board, check_board_validity
from .nodes.detect_change import wait_for_human_move, wait_for_human_manual, detect_human_move
from .nodes.think import think_move, think_move_stockfish_only
from .nodes.act import execute_robot_move, verify_robot_move
from .nodes.error_handler import (
    handle_vision_error,
    handle_robot_error,
    handle_timeout,
    handle_game_end,
    check_game_status,
)
from .nodes.voice_announce import voice_announce
from .board_tracker import BoardTracker


def route_after_observe(state: AgentState) -> Literal["valid", "vision_error"]:
    """Route after observation based on success."""
    if state.get("vision_error") or not state.get("current_board"):
        return "vision_error"
    if not state["current_board"].get("is_valid", False):
        return "vision_error"
    return "valid"


def route_by_turn(state: AgentState) -> Literal["human", "robot"]:
    """Route based on whose turn it is."""
    return state["game"]["whose_turn"]


def route_after_human_move(state: AgentState) -> Literal["detected", "waiting", "vision_error", "timeout", "paused"]:
    """Route after human move detection."""
    if state.get("paused", False):
        return "paused"
    error_type = state.get("error_type")
    if error_type == "timeout":
        return "timeout"
    if error_type:
        return "vision_error"
    if state.get("human_move_detected"):
        return "detected"
    return "waiting"


def route_after_act(state: AgentState) -> Literal["success", "error"]:
    """Route after robot action."""
    if state.get("robot_error") or state.get("error_type"):
        return "error"
    return "success"


def route_game_status(state: AgentState) -> Literal["continue", "end"]:
    """Check if game should continue."""
    return check_game_status(state)


def route_check_paused(state: AgentState) -> Literal["paused", "running"]:
    """Route based on whether the agent is paused."""
    if state.get("paused", False):
        return "paused"
    return "running"


def init_game(state: AgentState) -> dict:
    """Initialize game state."""
    robot_color = state.get("game", {}).get("robot_color", "black")
    initial = create_initial_state(robot_color)

    print("\n" + "=" * 50)
    print("  RX200 Chess Robot Agent")
    print("=" * 50)
    print(f"  Robot plays: {robot_color.upper()}")
    print(f"  Human plays: {'white' if robot_color == 'black' else 'black'}")
    print("=" * 50)

    if initial["game"]["whose_turn"] == "human":
        print("\nYour move first! (You are White)")
    else:
        print("\nI'll move first! (I am White)")

    return {**initial, "current_phase": "init"}


def wait_for_resume(state: AgentState) -> dict:
    """
    Pause node — the graph stops here when paused.

    The external API sets user_action and clears paused, then the graph
    re-enters from the appropriate phase via current_phase routing.
    """
    reason = state.get("pause_reason", "unknown")
    print(f"[Agent] PAUSED ({reason}) — waiting for /game/resume ...")
    # Return END-like state; the web API will resume by re-invoking the graph
    return {
        "should_continue": False,
    }


def print_board(state: AgentState) -> dict:
    """Print current board state."""
    board = state.get("current_board")
    if board:
        print("\n" + board.get("ascii_board", ""))
        print(f"FEN: {board.get('fen', '')}")
    return {}


def validate_initial_position(state: AgentState) -> dict:
    """
    Validate the initial board position using vision and the tracker.

    Creates a BoardTracker with the standard starting position,
    captures the board via vision, and compares.
    """
    from .nodes.detect_change import get_vision_tool

    tracker = BoardTracker()

    # Capture board via vision
    vision_tool = get_vision_tool()
    result = vision_tool._run()

    if not result["success"]:
        print("[Agent] Vision failed during initial validation, proceeding with tracker only")
        return {
            "board_tracker_state": tracker.to_dict(),
            "initial_position_validated": False,
            "tracker_discrepancies": [],
            "current_phase": "validate_start",
        }

    vision_pieces = result.get("piece_positions", {})
    is_valid, discrepancies = tracker.validate_initial_position(vision_pieces)

    if is_valid:
        print("[Agent] Initial position validated - all pieces match!")
    else:
        print(f"[Agent] Initial position: {len(discrepancies)} discrepancies (tracker is authoritative)")
        for d in discrepancies:
            print(f"  {d.square}: {d.type.value} - tracker={d.tracker_piece}, vision={d.vision_piece}")

    return {
        "board_tracker_state": tracker.to_dict(),
        "initial_position_validated": is_valid,
        "tracker_discrepancies": [
            {"square": d.square, "type": d.type.value,
             "tracker_piece": d.tracker_piece, "vision_piece": d.vision_piece}
            for d in discrepancies
        ],
        "current_phase": "validate_start",
    }


def create_chess_agent_graph(use_manual_trigger: bool = False, use_llm: bool = True):
    """
    Create the LangGraph state machine for the chess robot.

    Args:
        use_manual_trigger: If True, wait for ENTER key instead of polling
        use_llm: If True, use LLM for move decisions; otherwise just Stockfish

    Returns:
        Compiled LangGraph
    """
    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("init", init_game)
    workflow.add_node("validate_start", validate_initial_position)
    workflow.add_node("observe", observe_board)
    workflow.add_node("print_board", print_board)
    workflow.add_node("handle_vision_error", handle_vision_error)

    # Human turn nodes
    if use_manual_trigger:
        workflow.add_node("wait_human", wait_for_human_manual)
    else:
        workflow.add_node("wait_human", wait_for_human_move)
    workflow.add_node("detect_change", detect_human_move)

    # Robot turn nodes
    if use_llm:
        workflow.add_node("think", think_move)
    else:
        workflow.add_node("think", think_move_stockfish_only)
    workflow.add_node("act", execute_robot_move)
    workflow.add_node("verify", verify_robot_move)

    # Voice announcement node
    workflow.add_node("voice_announce", voice_announce)

    # Error/end nodes
    workflow.add_node("handle_robot_error", handle_robot_error)
    workflow.add_node("handle_timeout", handle_timeout)
    workflow.add_node("handle_game_end", handle_game_end)

    # Pause node
    workflow.add_node("wait_for_resume", wait_for_resume)

    # Set entry point
    workflow.set_entry_point("init")

    # Edges from init -> validate_start -> observe
    workflow.add_edge("init", "validate_start")
    workflow.add_edge("validate_start", "observe")

    # Edges from observe
    workflow.add_conditional_edges(
        "observe",
        route_after_observe,
        {
            "valid": "print_board",
            "vision_error": "handle_vision_error",
        }
    )

    # Vision error handling — may pause
    workflow.add_conditional_edges(
        "handle_vision_error",
        route_check_paused,
        {
            "paused": "wait_for_resume",
            "running": "observe",
        }
    )

    # After printing board, route by whose turn
    workflow.add_conditional_edges(
        "print_board",
        route_by_turn,
        {
            "human": "wait_human",
            "robot": "think",
        }
    )

    # Human turn flow
    workflow.add_conditional_edges(
        "wait_human",
        route_after_human_move,
        {
            "detected": "detect_change",
            "waiting": "wait_human",
            "paused": "wait_for_resume",
            "timeout": "handle_timeout",
            "vision_error": "handle_vision_error",
        }
    )

    # Timeout handling — may pause for user decision
    workflow.add_conditional_edges(
        "handle_timeout",
        route_check_paused,
        {
            "paused": "wait_for_resume",
            "running": "wait_human",
        }
    )

    # After detecting human move, announce and check game status
    workflow.add_conditional_edges(
        "detect_change",
        route_game_status,
        {
            "continue": "voice_announce",  # Announce, then show board
            "end": "handle_game_end",
        }
    )

    # Robot turn flow
    workflow.add_edge("think", "act")

    # After robot action
    workflow.add_conditional_edges(
        "act",
        route_after_act,
        {
            "success": "verify",
            "error": "handle_robot_error",
        }
    )

    # Robot error handling — may pause for user decision
    workflow.add_conditional_edges(
        "handle_robot_error",
        route_check_paused,
        {
            "paused": "wait_for_resume",
            "running": "observe",
        }
    )

    # After verification, announce and check game status
    workflow.add_conditional_edges(
        "verify",
        route_game_status,
        {
            "continue": "voice_announce",  # Announce robot move, then show board
            "end": "handle_game_end",
        }
    )

    # Voice announce routes to print_board
    workflow.add_edge("voice_announce", "print_board")

    # Pause — graph exits; web API re-invokes after user resumes
    workflow.add_edge("wait_for_resume", END)

    # Game end
    workflow.add_edge("handle_game_end", END)

    return workflow.compile()


def create_simple_graph():
    """
    Create a simpler graph for testing without full hardware.

    Uses mock tools and manual trigger.
    """
    return create_chess_agent_graph(
        use_manual_trigger=True,
        use_llm=False,
    )
