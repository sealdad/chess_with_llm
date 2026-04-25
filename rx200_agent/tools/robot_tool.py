# rx200_agent/tools/robot_tool.py
"""
Robot control tool for the RX-200 arm via ROS1.
"""

import time
from typing import Optional, Any, Tuple
from langchain.tools import BaseTool
from pydantic import Field
import numpy as np

from ..config import (
    HAND_EYE_CALIBRATION,
    BOARD_ORIGIN,
    BOARD_SQUARE_SIZE,
    APPROACH_HEIGHT,
    GRASP_HEIGHT,
    GRAVEYARD_WHITE,
    GRAVEYARD_BLACK,
)
from ..utils.coord_transform import (
    square_to_robot_xyz,
    get_piece_grasp_height,
    load_calibration,
)
from ..utils.move_parser import (
    parse_uci_move,
    is_capture_move,
    get_captured_piece,
    is_castling_move,
    get_castling_rook_move,
)


class RobotTool(BaseTool):
    """
    Tool for controlling the RX-200 robot arm to move chess pieces.

    Uses ROS1 and the Interbotix SDK for motion control.
    """

    name: str = "move_piece"
    description: str = """
    Move a chess piece on the board using the robot arm.
    Input: A JSON object with:
        - uci_move: The move in UCI format (e.g., "e2e4")
        - fen: Current board FEN (to determine if capture)

    The robot will:
    1. If capture: first remove the opponent's piece to the graveyard
    2. Pick up the piece from the source square
    3. Place it on the destination square
    4. For castling: also move the rook

    Returns success status and any error messages.
    """

    _bot: Optional[Any] = None
    _initialized: bool = False
    calibration: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, lazy_init: bool = True, **kwargs):
        super().__init__(**kwargs)
        if not lazy_init:
            self._initialize()

    def _initialize(self):
        """Initialize robot and load calibration."""
        if self._initialized:
            return

        try:
            # Import ROS/Interbotix modules
            from interbotix_xs_modules.arm import InterbotixManipulatorXS

            self._bot = InterbotixManipulatorXS("rx200", "arm", "gripper")

            # Load calibration
            try:
                self.calibration = load_calibration(str(HAND_EYE_CALIBRATION))
            except FileNotFoundError:
                print("[Warning] Hand-eye calibration not found, using defaults")
                self.calibration = None

            self._initialized = True

        except Exception as e:
            raise RuntimeError(f"Failed to initialize robot: {e}")

    def _run(self, uci_move: str, fen: str) -> dict:
        """
        Execute a chess move with the robot.

        Args:
            uci_move: Move in UCI format (e.g., "e2e4")
            fen: Current board FEN

        Returns:
            dict with success status and details
        """
        self._initialize()

        try:
            from_sq, to_sq, promotion = parse_uci_move(uci_move)

            # Check if this is a capture
            is_capture = is_capture_move(fen, uci_move)
            captured_piece = get_captured_piece(fen, uci_move) if is_capture else None

            # Handle capture - remove opponent piece first
            if is_capture and captured_piece:
                graveyard = GRAVEYARD_WHITE if "white" in captured_piece else GRAVEYARD_BLACK
                self._remove_piece_to_graveyard(to_sq, graveyard)

            # Move the piece
            self._move_piece(from_sq, to_sq)

            # Handle castling - also move the rook
            if is_castling_move(fen, uci_move):
                rook_move = get_castling_rook_move(fen, uci_move)
                if rook_move:
                    rook_from, rook_to = rook_move
                    self._move_piece(rook_from, rook_to)

            # Handle promotion - for now, just log it
            # (physical piece swap would need human intervention)
            if promotion:
                print(f"[Note] Pawn promoted to {promotion} - physical swap may be needed")

            # Return to home position
            self._go_home()

            return {
                "success": True,
                "move": uci_move,
                "from_square": from_sq,
                "to_square": to_sq,
                "was_capture": is_capture,
                "captured_piece": captured_piece,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "move": uci_move,
            }

    async def _arun(self, uci_move: str, fen: str) -> dict:
        """Async version - just calls sync version."""
        return self._run(uci_move, fen)

    def _move_piece(self, from_square: str, to_square: str):
        """Pick up piece from one square and place on another."""
        # Get coordinates
        from_xyz = square_to_robot_xyz(from_square)
        to_xyz = square_to_robot_xyz(to_square)

        # Pick up piece
        self._pick_piece(from_xyz, square=from_square)

        # Place piece
        self._place_piece(to_xyz, square=to_square)

    def _pick_piece(self, xyz: np.ndarray, square: Optional[str] = None):
        """Pick up a piece at the given coordinates."""
        # Add 20mm offset for rows 1-3
        grasp_z_offset = GRASP_HEIGHT
        if square and len(square) >= 2 and square[1] in "123":
            grasp_z_offset += 0.020
            print(f"[RobotTool._pick_piece] +20mm row offset for {square}")

        # Move to approach position above piece
        self._bot.arm.set_ee_pose_components(
            x=xyz[0], y=xyz[1], z=xyz[2] + APPROACH_HEIGHT,
            moving_time=1.0
        )
        time.sleep(0.5)

        # Open gripper
        self._bot.gripper.open()
        time.sleep(0.3)

        # Lower to grasp height
        self._bot.arm.set_ee_pose_components(
            x=xyz[0], y=xyz[1], z=xyz[2] + grasp_z_offset,
            moving_time=0.8
        )
        time.sleep(0.3)

        # Close gripper to grasp piece
        self._bot.gripper.close()
        time.sleep(0.3)

        # Lift piece
        self._bot.arm.set_ee_pose_components(
            x=xyz[0], y=xyz[1], z=xyz[2] + APPROACH_HEIGHT,
            moving_time=0.8
        )
        time.sleep(0.3)

    def _place_piece(self, xyz: np.ndarray, square: Optional[str] = None):
        """Place the held piece at the given coordinates."""
        # Add 20mm offset for rows 1-3
        grasp_z_offset = GRASP_HEIGHT
        if square and len(square) >= 2 and square[1] in "123":
            grasp_z_offset += 0.020
            print(f"[RobotTool._place_piece] +20mm row offset for {square}")

        # Move to approach position above destination
        self._bot.arm.set_ee_pose_components(
            x=xyz[0], y=xyz[1], z=xyz[2] + APPROACH_HEIGHT,
            moving_time=1.0
        )
        time.sleep(0.5)

        # Lower to place height
        self._bot.arm.set_ee_pose_components(
            x=xyz[0], y=xyz[1], z=xyz[2] + grasp_z_offset,
            moving_time=0.8
        )
        time.sleep(0.3)

        # Open gripper to release piece
        self._bot.gripper.open()
        time.sleep(0.3)

        # Lift away
        self._bot.arm.set_ee_pose_components(
            x=xyz[0], y=xyz[1], z=xyz[2] + APPROACH_HEIGHT,
            moving_time=0.8
        )
        time.sleep(0.3)

    def _remove_piece_to_graveyard(self, square: str, graveyard_pos: list):
        """Remove a captured piece to the graveyard area."""
        # Get piece position
        piece_xyz = square_to_robot_xyz(square)

        # Pick up the piece
        self._pick_piece(piece_xyz, square=square)

        # Place in graveyard (no square-based offset for graveyard)
        graveyard_xyz = np.array(graveyard_pos)
        self._place_piece(graveyard_xyz)

    def _go_home(self):
        """Return robot to home/sleep position."""
        self._bot.arm.go_to_home_pose()
        time.sleep(0.5)


class MockRobotTool(BaseTool):
    """
    Mock robot tool for testing without hardware.
    """

    name: str = "move_piece"
    description: str = "Mock robot tool for testing."

    def _run(self, uci_move: str, fen: str) -> dict:
        """Simulate moving a piece."""
        from_sq, to_sq, promotion = parse_uci_move(uci_move)

        print(f"[MockRobot] Moving piece from {from_sq} to {to_sq}")

        is_capture = is_capture_move(fen, uci_move)
        captured = get_captured_piece(fen, uci_move) if is_capture else None

        if is_capture:
            print(f"[MockRobot] Capturing {captured}")

        return {
            "success": True,
            "move": uci_move,
            "from_square": from_sq,
            "to_square": to_sq,
            "was_capture": is_capture,
            "captured_piece": captured,
        }

    async def _arun(self, uci_move: str, fen: str) -> dict:
        return self._run(uci_move, fen)
