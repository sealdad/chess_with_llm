# rx200_agent/tools/vision_tool.py
"""
Vision tool wrapping the ChessVisionPipeline.
"""

import time
from typing import Optional, Any
from langchain.tools import BaseTool
from pydantic import Field

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chess_vision.vision_pipeline import ChessVisionPipeline
from chess_vision.camera import RealSenseCamera


class VisionTool(BaseTool):
    """
    Tool for capturing and analyzing the chess board.

    Wraps the existing ChessVisionPipeline and RealSenseCamera
    to provide board state information to the agent.
    """

    name: str = "get_board_state"
    description: str = """
    Capture an image of the chess board and analyze the current position.
    Returns FEN notation, piece positions, and ASCII board visualization.
    Use this tool to observe the current state of the chess game.
    No input required - captures from the connected camera.
    """

    camera: Optional[Any] = Field(default=None, exclude=True)
    pipeline: Optional[Any] = Field(default=None, exclude=True)
    _initialized: bool = False

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, lazy_init: bool = True, **kwargs):
        """
        Initialize the vision tool.

        Args:
            lazy_init: If True, defer camera/pipeline init until first use
        """
        super().__init__(**kwargs)
        if not lazy_init:
            self._initialize()

    def _initialize(self):
        """Initialize camera and vision pipeline."""
        if self._initialized:
            return

        try:
            self.camera = RealSenseCamera()
            self.pipeline = ChessVisionPipeline()
            self._initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vision system: {e}")

    def _run(self, *args, **kwargs) -> dict:
        """
        Capture image and analyze board state.

        Returns:
            dict with:
                - success: bool
                - fen: FEN string (if success)
                - ascii_board: ASCII representation
                - piece_positions: dict of square -> piece
                - is_valid: whether position is valid
                - warnings: list of warnings
                - timestamp: capture timestamp
                - error: error message (if failed)
        """
        self._initialize()

        try:
            # Capture frame from camera
            success, _, color_img, depth_img = self.camera.get_frame()

            if not success or color_img is None:
                return {
                    "success": False,
                    "error": "Failed to capture camera frame",
                }

            # Analyze with vision pipeline
            result = self.pipeline.analyze_image(color_img)

            if result is None:
                return {
                    "success": False,
                    "error": "Board not detected in image",
                }

            return {
                "success": True,
                "fen": result.get_fen(),
                "ascii_board": result.get_ascii_board(),
                "piece_positions": result.board_state,
                "is_valid": result.analysis["is_valid"],
                "warnings": result.analysis.get("warnings", []),
                "timestamp": time.time(),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def _arun(self, *args, **kwargs) -> dict:
        """Async version - just calls sync version."""
        return self._run(*args, **kwargs)

    def release(self):
        """Release camera resources."""
        if self.camera:
            self.camera.release()
            self._initialized = False


class MockVisionTool(BaseTool):
    """
    Mock vision tool for testing without hardware.
    """

    name: str = "get_board_state"
    description: str = "Mock vision tool for testing."

    current_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def set_fen(self, fen: str):
        """Set the FEN that will be returned."""
        self.current_fen = fen

    def _run(self, *args, **kwargs) -> dict:
        """Return mock board state."""
        return {
            "success": True,
            "fen": self.current_fen,
            "ascii_board": self._fen_to_ascii(self.current_fen),
            "piece_positions": {},
            "is_valid": True,
            "warnings": [],
            "timestamp": time.time(),
        }

    async def _arun(self, *args, **kwargs) -> dict:
        return self._run(*args, **kwargs)

    def _fen_to_ascii(self, fen: str) -> str:
        """Convert FEN to simple ASCII board."""
        rows = fen.split()[0].split("/")
        lines = ["    a b c d e f g h", "  +-----------------+"]

        for i, row in enumerate(rows):
            rank = 8 - i
            expanded = ""
            for c in row:
                if c.isdigit():
                    expanded += "." * int(c)
                else:
                    expanded += c
            line = f"  {rank}| {' '.join(expanded)} |{rank}"
            lines.append(line)

        lines.append("  +-----------------+")
        lines.append("    a b c d e f g h")
        return "\n".join(lines)
