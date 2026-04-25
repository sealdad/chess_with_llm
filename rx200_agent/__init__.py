# rx200_agent package
"""
RX200 Chess Robot Agent - LangGraph-based AI agent for playing chess.

Main components:
    - state: State schema for the agent
    - graph: LangGraph state machine
    - tools: Vision, Stockfish, and Robot tools
    - nodes: Graph node implementations
"""

from .state import AgentState, GameState, BoardSnapshot, ChessMove

try:
    from .graph import create_chess_agent_graph
except ImportError:
    # langgraph may not be installed in test / lightweight environments
    create_chess_agent_graph = None

__version__ = "0.1.0"
