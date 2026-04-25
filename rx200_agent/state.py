# rx200_agent/state.py
"""
LangGraph state schema for the chess robot agent.

Defines all state types passed between nodes in the graph.
"""

from typing import TypedDict, Optional, List, Literal, Dict
try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated
from operator import add


# Agent mode type - matches conversation/modes.py AgentMode enum values
AgentModeType = Literal["coach", "friend", "opponent", "mean_friend"]


class ChessMove(TypedDict):
    """Represents a chess move."""
    from_square: str          # e.g., "e2"
    to_square: str            # e.g., "e4"
    piece: str                # e.g., "white_pawn"
    is_capture: bool          # True if capturing opponent piece
    captured_piece: Optional[str]  # e.g., "black_pawn" if capture
    promotion: Optional[str]  # e.g., "queen" for pawn promotion


class BoardSnapshot(TypedDict):
    """Snapshot of board state from vision."""
    fen: str                  # FEN notation
    piece_positions: dict     # Square -> piece mapping
    ascii_board: str          # ASCII representation
    timestamp: float          # Unix timestamp
    is_valid: bool            # Whether board state is valid
    warnings: List[str]       # Any validation warnings


class GameState(TypedDict):
    """Complete game state."""
    # Board state
    current_fen: str
    previous_fen: Optional[str]
    board_history: List[str]  # List of FENs

    # Turn management
    whose_turn: Literal["human", "robot"]
    move_number: int
    robot_color: Literal["white", "black"]

    # Last moves
    last_human_move: Optional[ChessMove]
    last_robot_move: Optional[ChessMove]

    # Game status
    game_status: Literal["playing", "checkmate", "stalemate", "draw", "resigned", "error"]
    is_check: bool


class AgentState(TypedDict):
    """
    Main LangGraph state schema.

    This is the central state object passed between all nodes in the graph.
    """
    # Game state
    game: GameState

    # Vision state
    current_board: Optional[BoardSnapshot]
    previous_board: Optional[BoardSnapshot]
    vision_error: Optional[str]
    consecutive_vision_failures: int

    # Agent reasoning
    messages: Annotated[List[dict], add]  # Chat history for LLM (accumulates)
    llm_reasoning: str
    suggested_move: Optional[str]      # e.g., "e2e4" in UCI format
    stockfish_best_move: Optional[str]
    stockfish_evaluation: Optional[str]

    # Robot state
    robot_busy: bool
    robot_error: Optional[str]
    last_robot_action: Optional[str]

    # Human move detection
    waiting_for_human: bool
    human_move_detected: bool
    detected_human_move: Optional[ChessMove]
    stable_board_count: int  # Consecutive stable frames

    # Board tracker (authoritative state)
    board_tracker_state: Optional[dict]
    initial_position_validated: bool
    tracker_discrepancies: List[dict]

    # Control flow
    current_phase: Literal[
        "init",
        "validate_start",
        "observe",
        "wait_human",
        "detect_change",
        "think",
        "act",
        "verify",
        "error",
        "end"
    ]
    error_type: Optional[str]
    error_message: Optional[str]
    should_continue: bool

    # Voice interaction state
    voice_enabled: bool
    last_voice_input: Optional[str]       # Transcribed user speech
    pending_voice_response: Optional[str]  # Text to speak to user
    voice_context: List[dict]              # Voice conversation history

    # Agent mode and conversation
    agent_mode: AgentModeType             # Current personality mode
    conversation_history: List[Dict]       # LLM conversation history

    # Pause/resume (replaces blocking input() calls)
    paused: bool                          # True when waiting for user action
    pause_reason: Optional[str]           # Why the agent is paused
    pause_options: Optional[List[str]]    # Available actions (e.g. ["retry","skip","abort"])
    user_action: Optional[str]            # Action chosen by user via /game/resume


def create_initial_state(robot_color: str = "black") -> AgentState:
    """Create initial agent state for a new game."""
    # Determine who moves first
    # White always moves first in chess
    human_plays_white = (robot_color == "black")

    return {
        "game": {
            "current_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "previous_fen": None,
            "board_history": [],
            "whose_turn": "human" if human_plays_white else "robot",
            "move_number": 1,
            "robot_color": robot_color,
            "last_human_move": None,
            "last_robot_move": None,
            "game_status": "playing",
            "is_check": False,
        },
        "current_board": None,
        "previous_board": None,
        "vision_error": None,
        "consecutive_vision_failures": 0,
        "messages": [],
        "llm_reasoning": "",
        "suggested_move": None,
        "stockfish_best_move": None,
        "stockfish_evaluation": None,
        "robot_busy": False,
        "robot_error": None,
        "last_robot_action": None,
        "waiting_for_human": False,
        "human_move_detected": False,
        "detected_human_move": None,
        "stable_board_count": 0,
        "board_tracker_state": None,
        "initial_position_validated": False,
        "tracker_discrepancies": [],
        "current_phase": "init",
        "error_type": None,
        "error_message": None,
        "should_continue": True,
        "voice_enabled": False,
        "last_voice_input": None,
        "pending_voice_response": None,
        "voice_context": [],
        "agent_mode": "friend",  # Default to friendly mode
        "conversation_history": [],
        "paused": False,
        "pause_reason": None,
        "pause_options": None,
        "user_action": None,
    }
