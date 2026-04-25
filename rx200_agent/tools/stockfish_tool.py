# rx200_agent/tools/stockfish_tool.py
"""
Stockfish chess engine tool for move analysis.
"""

from typing import Optional, Any
from langchain.tools import BaseTool
from pydantic import Field

from ..config import STOCKFISH_PATH, STOCKFISH_DEPTH, STOCKFISH_SKILL_LEVEL


class StockfishTool(BaseTool):
    """
    Tool for querying Stockfish chess engine.

    Provides best move recommendations and position evaluation.
    """

    name: str = "query_stockfish"
    description: str = """
    Query Stockfish chess engine for the best move in a given position.
    Input: FEN string of the current chess position.
    Returns: Best move in UCI format (e.g., "e2e4"), evaluation score, and top alternative moves.
    Use this to get strong move recommendations.
    """

    stockfish_path: str = STOCKFISH_PATH
    depth: int = STOCKFISH_DEPTH
    skill_level: int = STOCKFISH_SKILL_LEVEL

    _engine: Optional[Any] = None
    _initialized: bool = False

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        stockfish_path: str = STOCKFISH_PATH,
        depth: int = STOCKFISH_DEPTH,
        skill_level: int = STOCKFISH_SKILL_LEVEL,
        lazy_init: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.skill_level = skill_level
        if not lazy_init:
            self._initialize()

    def _initialize(self):
        """Initialize Stockfish engine."""
        if self._initialized:
            return

        try:
            from stockfish import Stockfish
            self._engine = Stockfish(self.stockfish_path)
            self._engine.set_depth(self.depth)
            self._engine.set_skill_level(self.skill_level)
            self._initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Stockfish: {e}")

    def _run(self, fen: str) -> dict:
        """
        Query Stockfish for the best move.

        Args:
            fen: FEN string of the position to analyze

        Returns:
            dict with:
                - success: bool
                - best_move: UCI move string (e.g., "e2e4")
                - evaluation: position evaluation
                - top_moves: list of top 3 moves with evaluations
                - error: error message (if failed)
        """
        self._initialize()

        try:
            # Set the position
            self._engine.set_fen_position(fen)

            # Get best move
            best_move = self._engine.get_best_move()

            if best_move is None:
                return {
                    "success": False,
                    "error": "No legal moves available (game over?)",
                }

            # Get evaluation
            evaluation = self._engine.get_evaluation()

            # Get top moves
            top_moves = self._engine.get_top_moves(3)

            # Format evaluation string
            if evaluation["type"] == "cp":
                eval_str = f"{evaluation['value'] / 100:.2f}"  # Convert centipawns to pawns
            else:  # mate
                eval_str = f"M{evaluation['value']}"

            return {
                "success": True,
                "best_move": best_move,
                "evaluation": eval_str,
                "evaluation_raw": evaluation,
                "top_moves": top_moves,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def _arun(self, fen: str) -> dict:
        """Async version - just calls sync version."""
        return self._run(fen)


class MockStockfishTool(BaseTool):
    """
    Mock Stockfish tool for testing without the engine.
    """

    name: str = "query_stockfish"
    description: str = "Mock Stockfish tool for testing."

    def _run(self, fen: str) -> dict:
        """Return a mock best move."""
        # Simple mock - just return a common opening move
        if "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" in fen:
            # Starting position - suggest e4 or d4
            return {
                "success": True,
                "best_move": "e2e4",
                "evaluation": "0.30",
                "evaluation_raw": {"type": "cp", "value": 30},
                "top_moves": [
                    {"Move": "e2e4", "Centipawn": 30},
                    {"Move": "d2d4", "Centipawn": 25},
                    {"Move": "c2c4", "Centipawn": 20},
                ],
            }

        # Default response
        return {
            "success": True,
            "best_move": "e2e4",
            "evaluation": "0.00",
            "evaluation_raw": {"type": "cp", "value": 0},
            "top_moves": [],
        }

    async def _arun(self, fen: str) -> dict:
        return self._run(fen)
