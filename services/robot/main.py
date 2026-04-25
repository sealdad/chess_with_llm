#!/usr/bin/env python3
"""
Robot Service API — Reference implementation for Interbotix RX-200

Monolithic FastAPI service controlling an Interbotix RX-200 5-DOF arm via
ROS 1 Noetic and the Interbotix SDK. Handles chess piece manipulation,
calibration, and board teaching.

Hardware-specific parameters (joint names, gripper range, servo settling
times) are collected in the configuration section near the top of this file.
To adapt for a different robot arm, implement the same REST API endpoints
with your robot's SDK.
"""

import time
import math
import logging
import traceback
import threading
from typing import Optional, List, Tuple, Dict
from contextlib import asynccontextmanager

import httpx
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path

# Import board tracker directly (bypass rx200_agent/__init__.py which needs Python 3.9+)
import sys
import importlib.util
_bt_paths = [
    Path(__file__).parent / "rx200_agent" / "board_tracker.py",           # /app/rx200_agent/ in container
    Path(__file__).parent.parent.parent / "rx200_agent" / "board_tracker.py",  # project root locally
]
_bt_path = next((p for p in _bt_paths if p.exists()), None)
if _bt_path is None:
    raise ImportError("Cannot find rx200_agent/board_tracker.py")
_spec = importlib.util.spec_from_file_location("board_tracker", _bt_path)
_bt_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bt_mod)
BoardTracker = _bt_mod.BoardTracker


# Configuration
HAND_EYE_CALIBRATION = Path(__file__).parent.parent.parent / "hand_eye_calibration.yaml"
WAYPOINTS_FILE = Path("/app/waypoints.yaml")
CALIBRATION_FILE = Path("/app/camera_robot_calibration.yaml")
CAPTURE_ZONE_FILE = Path("/app/capture_zone_position.yaml")
PROMOTION_QUEEN_FILE = Path("/app/promotion_queen_position.yaml")
SQUARE_POSITIONS_FILE = Path("/app/square_positions.yaml")
BOARD_SURFACE_Z_FILE = Path("/app/board_surface_z.yaml")
GESTURES_FILE = Path("/app/gestures.yaml")
BOARD_SQUARE_SIZE = 0.05  # 5cm squares
BOARD_ORIGIN = [0.30, -0.15, 0.02]  # XYZ of a1 corner
APPROACH_HEIGHT = 0.15
GRASP_HEIGHT = 0.02
# Speed multiplier: 1.0 = normal, 0.5 = 2x speed
SPEED_FACTOR = 0.5
GRAVEYARD_WHITE = [0.10, 0.25, 0.02]
GRAVEYARD_BLACK = [0.10, -0.25, 0.02]
VISION_SERVICE_URL = "http://localhost:8001"

log = logging.getLogger("robot_service")


# Global robot instance
robot = None
robot_connected = False


# Capture zone position (XYZ+pitch — where captured pieces are placed)
capture_zone_position = None  # Dict with x, y, z, pitch

# Promotion queen position (XYZ+pitch — where a spare queen sits for pickup)
promotion_queen_position = None  # Dict with x, y, z, pitch

# Calibration state
calibration_points_camera = []  # List of [x, y, z] in camera frame
calibration_points_robot = []   # List of [x, y, z] in robot frame
calibration_points_tag_ids = [] # List of tag IDs for display
calibration_transform = None    # 4x4 transformation matrix (camera to robot)

# Board surface Z (taught by touching the board)
board_surface_z: Optional[float] = None

# Gesture recording/playback
_gestures = {}  # type: Dict[str, dict]  # {name: {frames: [...], fps: 20, ...}}
_gesture_recording = False
_gesture_record_thread = None  # type: Optional[threading.Thread]
_gesture_record_frames = []  # type: List[list]
_gesture_playing = False


class MoveRequest(BaseModel):
    """Request to move a piece."""
    uci_move: str = Field(..., description="Move in UCI format (e.g., 'e2e4')")
    fen: str = Field(..., description="Current board FEN")


class SquareRequest(BaseModel):
    """Request for a single square operation."""
    square: str = Field(..., description="Square name (e.g., 'e4')")
    grasp_height: Optional[float] = Field(None, description="Measured piece height in meters (uses default if None)")


class MoveToXYZRequest(BaseModel):
    """Request to move to XYZ position."""
    x: float = Field(..., description="X position in meters")
    y: float = Field(..., description="Y position in meters")
    z: float = Field(..., description="Z position in meters")
    pitch: Optional[float] = Field(1.5708, description="Pitch angle in radians (default: +90° pointing down)")
    roll: Optional[float] = Field(0.0, description="Roll angle in radians (default: 0)")
    moving_time: Optional[float] = Field(1.5, description="Movement time in seconds")
    auto_orientation: Optional[bool] = Field(False, description="If True, let IK choose best orientation (ignores pitch/roll)")
    pitch_tolerance: Optional[float] = Field(0.0, description="Pitch tolerance in radians. If >0, will search nearby pitch values if exact fails")


class MoveResponse(BaseModel):
    """Response for move operations."""
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None
    move: Optional[str] = None
    from_square: Optional[str] = None
    to_square: Optional[str] = None
    was_capture: Optional[bool] = None
    debug: Optional[dict] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    robot_connected: bool
    timestamp: float


class StatusResponse(BaseModel):
    """Robot status response."""
    service: str
    version: str
    robot_connected: bool
    robot_type: str
    last_action: Optional[str] = None
    last_action_time: Optional[float] = None


class JointJogRequest(BaseModel):
    """Request to jog a single joint."""
    joint: str = Field(..., description="Joint name: waist, shoulder, elbow, wrist_angle, wrist_rotate")
    step: float = Field(..., description="Step in radians (positive or negative)")


class CartesianJogRequest(BaseModel):
    """Request to jog in cartesian space."""
    axis: str = Field(..., description="Axis: x, y, or z")
    step: float = Field(..., description="Step in meters (positive or negative)")


class ContinuousJogRequest(BaseModel):
    """Request to start/stop continuous jogging."""
    joint: str = Field(..., description="Joint name or 'stop' to stop all")
    direction: int = Field(..., description="1 for positive, -1 for negative, 0 to stop")
    velocity: float = Field(0.5, description="Velocity in rad/s")


class GripperRequest(BaseModel):
    """Request to set gripper position."""
    position: float = Field(..., description="Gripper position 0.0 (closed) to 1.0 (open)")


class JointPositionsResponse(BaseModel):
    """Response with joint positions."""
    success: bool
    joints: Optional[dict] = None
    ee_pose: Optional[dict] = None
    error: Optional[str] = None


class WaypointData(BaseModel):
    """Waypoint data structure."""
    name: str
    joints: dict  # {waist, shoulder, elbow, wrist_angle, wrist_rotate}
    gripper: float  # 0-1
    tag: Optional[str] = None  # e.g. "work_position", "vision_position"
    timestamp: float


class WaypointListResponse(BaseModel):
    """Response with list of waypoints (names + tags)."""
    success: bool
    waypoints: List[dict]  # [{name, tag}, ...]


class WaypointResponse(BaseModel):
    """Response for waypoint operations."""
    success: bool
    waypoint: Optional[WaypointData] = None
    error: Optional[str] = None


class WaypointSaveRequest(BaseModel):
    """Request to save a waypoint."""
    name: str = Field(..., description="Name for the waypoint")
    tag: Optional[str] = Field(None, description="Optional tag, e.g. 'work_position', 'vision_position'")


class HoverSquareRequest(BaseModel):
    """Request to hover above a square (for teaching demonstrations)."""
    square: str = Field(..., description="Square name (e.g., 'e4')")


class ConnectionResponse(BaseModel):
    """Response for connection operations."""
    success: bool
    connected: bool
    mock: bool = False
    message: Optional[str] = None
    error: Optional[str] = None


class ConnectRequest(BaseModel):
    """Request to connect to robot."""
    mock: bool = Field(False, description="Connect in mock mode (no real robot)")


class CalibrationPointPair(BaseModel):
    """A pair of corresponding points for calibration."""
    camera: List[float] = Field(..., description="[x, y, z] in camera frame (meters)")
    robot: List[float] = Field(..., description="[x, y, z] in robot frame (meters)")
    tag_id: Optional[int] = None


class CalibrationAddPointRequest(BaseModel):
    """Request to add a calibration point pair."""
    camera_point: List[float] = Field(..., description="[x, y, z] from camera")
    tag_id: Optional[int] = None


class CalibrationStatus(BaseModel):
    """Calibration status response."""
    success: bool
    num_points: int = 0
    points: List[CalibrationPointPair] = []
    is_calibrated: bool = False
    transform: Optional[List[List[float]]] = None  # 4x4 matrix
    error: Optional[str] = None
    message: Optional[str] = None


class ManualPickRequest(BaseModel):
    """Request to manually pick up a piece from a square."""
    square: str = Field(..., description="Chess square (e.g., 'e2')")
    piece_type: Optional[str] = Field(None, description="Piece type for height lookup (e.g., 'pawn')")
    skip_return_to_work: bool = Field(False, description="Skip L1 work return (when place follows immediately)")


class PromotionPickupRequest(BaseModel):
    """Request to pick up a spare piece from the promotion position."""
    piece_type: Optional[str] = Field("queen", description="Piece type for height lookup")


class ManualPlaceRequest(BaseModel):
    """Request to manually place a piece on a square."""
    square: str = Field(..., description="Chess square (e.g., 'e4')")
    piece_type: Optional[str] = Field(None, description="Piece type for height lookup (e.g., 'pawn')")


PIECE_HEIGHTS = {
    "pawn": 0.035, "rook": 0.040, "knight": 0.060,
    "bishop": 0.055, "queen": 0.070, "king": 0.080,
}


class TransformPointRequest(BaseModel):
    """Request to transform a point from camera to robot frame."""
    point: List[float] = Field(..., description="[x, y, z] in camera frame")


class TransformPointResponse(BaseModel):
    """Response with transformed point."""
    success: bool
    camera_point: List[float] = []
    robot_point: List[float] = []
    error: Optional[str] = None


# ============================================================
# Hardware Configuration — Interbotix RX-200 Reference Values
# Adapt these constants for a different robot arm.
# ============================================================
JOINT_NAMES = ["waist", "shoulder", "elbow", "wrist_angle", "wrist_rotate"]
DEFAULT_JOG_STEP_RAD = 0.05  # ~3 degrees
DEFAULT_JOG_STEP_M = 0.01  # 1cm
GRIPPER_CLOSED_M = 0.015     # Gripper fully closed position (metres)
GRIPPER_OPEN_M = 0.037       # Gripper fully open position (metres)
GRIPPER_65_OPEN_M = 0.037    # ~75 % open — used for piece pick/place (widened +3mm)


# Track last action
last_action: Optional[str] = None
last_action_time: Optional[float] = None

# Game state (occupancy-based tracking)
_board_tracker: Optional[BoardTracker] = None
_game_move_history: List[dict] = []
_game_white_side: Optional[str] = None  # "bottom" or "top"


def square_to_xyz(square: str) -> np.ndarray:
    """Convert chess square to robot XYZ coordinates."""
    file_char = square[0].lower()
    rank_char = square[1]

    file_idx = ord(file_char) - ord('a')
    rank_idx = int(rank_char) - 1

    x = BOARD_ORIGIN[0] + (file_idx + 0.5) * BOARD_SQUARE_SIZE
    y = BOARD_ORIGIN[1] + (rank_idx + 0.5) * BOARD_SQUARE_SIZE
    z = BOARD_ORIGIN[2]

    return np.array([x, y, z])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize service (robot connection is manual)."""
    print("[Robot Service] Starting... (use /connect to connect to robot)")

    yield

    # Cleanup
    if robot:
        try:
            robot.arm.go_to_sleep_pose()
        except:
            pass
    print("[Robot Service] Shutdown complete")


def connect_robot(mock: bool = False) -> Tuple[bool, str]:
    """Connect to the RX-200 robot."""
    global robot, robot_connected

    if robot_connected:
        return True, "Already connected" + (" (mock)" if robot is None else "")

    # If we have an existing robot object (soft disconnected), reuse it
    if robot is not None and not mock:
        robot_connected = True
        print("[Robot Service] Reconnected to existing robot instance")
        # Move to home pose on reconnect
        try:
            robot.arm.go_to_home_pose()
        except Exception as e:
            print(f"[Robot Service] Warning: Could not go to home pose: {e}")
        return True, "Reconnected to RX-200"

    if mock:
        robot = None
        robot_connected = True
        print("[Robot Service] Connected in MOCK mode")
        return True, "Connected in MOCK mode (no real robot)"

    try:
        from interbotix_xs_modules.arm import InterbotixManipulatorXS
        robot = InterbotixManipulatorXS("rx200", "arm", "gripper")
        robot_connected = True
        print("[Robot Service] Robot connected")
        return True, "Connected to RX-200"
    except ImportError as e:
        print(f"[Robot Service] Interbotix SDK not available: {e}")
        return False, "Interbotix SDK not installed. Use mock mode for testing."
    except Exception as e:
        print(f"[Robot Service] Robot connection failed: {e}")
        robot = None
        robot_connected = False
        return False, str(e)


def disconnect_robot() -> Tuple[bool, str]:
    """Soft disconnect from the robot (keeps ROS node alive for reconnection)."""
    global robot, robot_connected

    if not robot_connected:
        return True, "Already disconnected"

    # For mock mode, just mark as disconnected
    if robot is None:
        robot_connected = False
        return True, "Disconnected from mock mode"

    try:
        # Move to sleep pose before disconnecting
        robot.arm.go_to_sleep_pose()
        time.sleep(1 * SPEED_FACTOR)
    except Exception as e:
        print(f"[Robot Service] Error during sleep pose: {e}")

    # Soft disconnect: keep robot object alive but mark as disconnected
    # This allows reconnection without restarting the container
    robot_connected = False
    print("[Robot Service] Robot soft-disconnected (ROS node kept alive)")
    return True, "Disconnected from RX-200 (ready to reconnect)"


app = FastAPI(
    title="Chess Robot Arm API",
    description="REST API for robot arm chess piece manipulation. Reference implementation for Interbotix RX-200.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_collision_warning(start_xyz: np.ndarray, end_xyz: np.ndarray):
    """
    Call vision service to check for obstacles along the path.
    Warning-only: logs but does not block movement.
    """
    if calibration_transform is None:
        return

    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.post(
                f"{VISION_SERVICE_URL}/depth/check_collision",
                json={
                    "start_xyz": start_xyz.tolist(),
                    "end_xyz": end_xyz.tolist(),
                    "calibration_transform": calibration_transform.tolist(),
                },
            )
            data = resp.json()
            if data.get("success") and not data.get("clear", True):
                obstacles = data.get("obstacles", [])
                log.warning(
                    f"[Collision Warning] Path has {len(obstacles)} obstacle(s), "
                    f"min clearance: {data.get('min_clearance', '?')}m"
                )
    except Exception as e:
        log.debug(f"[Collision Check] Skipped: {e}")


def find_reachable_pitch(x, y, z):
    """
    Choose pitch based on distance from robot base.
    Tries multiple pitches from steep to shallow, returns first that passes IK.
    """
    import math

    # Distance from robot base (assuming base at origin)
    distance = math.sqrt(x**2 + y**2)

    # Try pitches from steep (straight down) to shallow (tilted forward)
    # Close positions can usually go straight down; far ones need more tilt
    if distance <= 0.20:
        # Very close — try steep angles first, then shallower
        pitches = [90, 80, 70, 60]
    elif distance <= 0.30:
        # Medium — try 80° down to 45°
        pitches = [80, 70, 60, 55, 45]
    else:
        # Far — try tilted angles
        pitches = [55, 45, 60, 70]

    print(f"[find_reachable_pitch] d={distance:.3f}m, z={z:.3f}m, trying {pitches}")

    if robot is None:
        return math.radians(pitches[0])

    for deg in pitches:
        pitch = math.radians(deg)
        result = robot.arm.set_ee_pose_components(
            x=x, y=y, z=z, pitch=pitch,
            blocking=False, execute=False  # Just check IK, don't move
        )
        # Result could be (joints, success) tuple or just bool
        if isinstance(result, tuple):
            success = result[1] if len(result) > 1 else bool(result[0] is not None)
        else:
            success = bool(result)
        if success:
            print(f"[find_reachable_pitch] OK: {deg}° for ({x:.3f}, {y:.3f}, {z:.3f})")
            return pitch
        else:
            print(f"[find_reachable_pitch] FAIL: {deg}° for ({x:.3f}, {y:.3f}, {z:.3f})")

    # Fallback
    fallback = math.radians(pitches[0])
    print(f"[find_reachable_pitch] WARNING: No valid pitch! Defaulting to {pitches[0]}°")
    return fallback


def pick_piece(xyz: np.ndarray, grasp_height: Optional[float] = None, square: Optional[str] = None):
    """
    Pick up a piece at coordinates.

    Args:
        xyz: target position [x, y, z]
        grasp_height: optional measured piece height in meters.
                      When provided, grasp Z = height * 0.5.
                      When None, uses default GRASP_HEIGHT.
        square: optional square name (e.g. "e2") for row-based offset.
    """
    global last_action, last_action_time

    effective_grasp_z = grasp_height * 0.5 if grasp_height is not None else GRASP_HEIGHT

    # Add 20mm offset for rows 1-3
    if square and len(square) >= 2 and square[1] in "123":
        effective_grasp_z += 0.020
        print(f"[pick_piece] +20mm row offset for {square} (row {square[1]})")

    if robot is None:
        print(f"[Mock] Pick piece at {xyz} (grasp_z={effective_grasp_z:.3f})")
        return

    print(f"===== PICK: {square or '?'} xyz=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}) grasp_z={effective_grasp_z:.3f} =====")

    # Find reachable pitch at approach height
    approach_z = xyz[2] + APPROACH_HEIGHT
    pitch_approach = find_reachable_pitch(xyz[0], xyz[1], approach_z)

    # Find reachable pitch at grasp height (may differ from approach)
    grasp_z = xyz[2] + effective_grasp_z
    pitch_grasp = find_reachable_pitch(xyz[0], xyz[1], grasp_z)

    # 1. Move above piece
    mt = 2.0
    at = 0.5
    print(f"[pick] Step 1: move above ({xyz[0]:.3f},{xyz[1]:.3f},{approach_z:.3f}) pitch={math.degrees(pitch_approach):.0f}°")
    robot.arm.set_ee_pose_components(
        x=xyz[0], y=xyz[1], z=approach_z,
        pitch=pitch_approach, moving_time=mt, accel_time=at
    )
    time.sleep(mt + 0.3)

    # 2. Open gripper
    robot.gripper.open()
    time.sleep(0.3)

    # 3. Lower to grasp
    mt = 1.5
    at = 0.4
    print(f"[pick] Step 3: lower to grasp ({xyz[0]:.3f},{xyz[1]:.3f},{grasp_z:.3f}) pitch={math.degrees(pitch_grasp):.0f}°")
    robot.arm.set_ee_pose_components(
        x=xyz[0], y=xyz[1], z=grasp_z,
        pitch=pitch_grasp, moving_time=mt, accel_time=at
    )
    time.sleep(mt + 0.2)

    # 4. Close gripper
    robot.gripper.close()
    time.sleep(0.5)

    # 5. Lift
    mt = 1.5
    at = 0.4
    print(f"[pick] Step 5: lift to ({xyz[0]:.3f},{xyz[1]:.3f},{approach_z:.3f})")
    robot.arm.set_ee_pose_components(
        x=xyz[0], y=xyz[1], z=approach_z,
        pitch=pitch_approach, moving_time=mt, accel_time=at
    )
    time.sleep(mt + 0.2)

    last_action = "pick"
    last_action_time = time.time()


def place_piece(xyz: np.ndarray, grasp_height: Optional[float] = None, square: Optional[str] = None):
    """
    Place piece at coordinates.

    Args:
        xyz: target position [x, y, z]
        grasp_height: optional measured piece height in meters.
                      When provided, place Z = height * 0.5.
                      When None, uses default GRASP_HEIGHT.
        square: optional square name (e.g. "e2") for row-based offset.
    """
    global last_action, last_action_time

    effective_grasp_z = grasp_height * 0.5 if grasp_height is not None else GRASP_HEIGHT

    # Add 20mm offset for rows 1-3
    if square and len(square) >= 2 and square[1] in "123":
        effective_grasp_z += 0.020
        print(f"[place_piece] +20mm row offset for {square} (row {square[1]})")

    if robot is None:
        print(f"[Mock] Place piece at {xyz} (grasp_z={effective_grasp_z:.3f})")
        return

    print(f"===== PLACE: {square or '?'} xyz=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}) grasp_z={effective_grasp_z:.3f} =====")

    approach_z = xyz[2] + APPROACH_HEIGHT
    grasp_z = xyz[2] + effective_grasp_z
    pitch_approach = find_reachable_pitch(xyz[0], xyz[1], approach_z)
    pitch_grasp = find_reachable_pitch(xyz[0], xyz[1], grasp_z)

    # 1. Move above destination
    mt = 2.0
    at = 0.5
    print(f"[place] Step 1: move above ({xyz[0]:.3f},{xyz[1]:.3f},{approach_z:.3f}) pitch={math.degrees(pitch_approach):.0f}°")
    robot.arm.set_ee_pose_components(
        x=xyz[0], y=xyz[1], z=approach_z,
        pitch=pitch_approach, moving_time=mt, accel_time=at
    )
    time.sleep(mt + 0.3)

    # 2. Lower
    mt = 1.5
    at = 0.4
    print(f"[place] Step 2: lower to ({xyz[0]:.3f},{xyz[1]:.3f},{grasp_z:.3f}) pitch={math.degrees(pitch_grasp):.0f}°")
    robot.arm.set_ee_pose_components(
        x=xyz[0], y=xyz[1], z=grasp_z,
        pitch=pitch_grasp, moving_time=mt, accel_time=at
    )
    time.sleep(mt + 0.2)

    # 3. Release
    robot.gripper.open()
    time.sleep(0.3)

    # 4. Lift
    mt = 1.5
    at = 0.4
    print(f"[place] Step 4: lift to ({xyz[0]:.3f},{xyz[1]:.3f},{approach_z:.3f})")
    robot.arm.set_ee_pose_components(
        x=xyz[0], y=xyz[1], z=approach_z,
        pitch=pitch_approach, moving_time=mt, accel_time=at
    )
    time.sleep(mt + 0.2)

    last_action = "place"
    last_action_time = time.time()


def _drop_at_capture_zone():
    """Move to capture zone waypoint (joints) and open gripper to drop piece."""
    global last_action, last_action_time
    cz_joints = get_capture_zone_joints()
    if cz_joints is None:
        print("[capture] WARNING: No capture_zone waypoint, using hardcoded graveyard")
        place_piece(np.array(GRAVEYARD_BLACK))
        return

    if robot is None:
        print("[Mock] Drop at capture zone")
        return

    print("[capture] Moving to capture zone waypoint (joints)")
    robot.arm.set_trajectory_time(moving_time=2.0 * SPEED_FACTOR, accel_time=0.5 * SPEED_FACTOR)
    robot.arm.set_joint_positions(cz_joints)
    time.sleep(1.5 * SPEED_FACTOR)

    # Open gripper to release piece
    robot.gripper.open()
    time.sleep(0.5 * SPEED_FACTOR)

    last_action = "drop_capture_zone"
    last_action_time = time.time()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check."""
    return HealthResponse(
        status="healthy",
        robot_connected=robot_connected,
        timestamp=time.time(),
    )


@app.post("/connect", response_model=ConnectionResponse)
async def connect(request: ConnectRequest = ConnectRequest()):
    """Connect to the RX-200 robot."""
    success, message = connect_robot(mock=request.mock)
    is_mock = robot_connected and robot is None
    return ConnectionResponse(
        success=success,
        connected=robot_connected,
        mock=is_mock,
        message=message if success else None,
        error=message if not success else None,
    )


@app.post("/disconnect", response_model=ConnectionResponse)
async def disconnect():
    """Disconnect from the robot (moves to sleep pose first)."""
    success, message = disconnect_robot()
    return ConnectionResponse(
        success=success,
        connected=robot_connected,
        message=message if success else None,
        error=message if not success else None,
    )


@app.get("/connection", response_model=ConnectionResponse)
async def get_connection_status():
    """Get current connection status."""
    is_mock = robot_connected and robot is None
    status = "Disconnected"
    if robot_connected:
        status = "Connected (mock)" if is_mock else "Connected"
    return ConnectionResponse(
        success=True,
        connected=robot_connected,
        mock=is_mock,
        message=status,
    )


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get robot status."""
    return StatusResponse(
        service="robot",
        version="1.0.0",
        robot_connected=robot_connected,
        robot_type="RX-200",
        last_action=last_action,
        last_action_time=last_action_time,
    )


@app.post("/move", response_model=MoveResponse)
async def execute_move(request: MoveRequest):
    """
    Execute a chess move.

    Handles:
    - Regular moves
    - Captures (removes opponent piece first)
    - Castling (moves rook too)
    """
    global last_action, last_action_time

    try:
        import chess

        uci_move = request.uci_move
        fen = request.fen

        # Parse move
        from_sq = uci_move[0:2]
        to_sq = uci_move[2:4]
        promotion = uci_move[4] if len(uci_move) > 4 else None

        # Check if capture
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci_move)

        is_capture = board.piece_at(move.to_square) is not None or board.is_en_passant(move)
        is_castling = board.is_castling(move)

        # Handle capture - remove piece first
        if is_capture:
            if board.is_en_passant(move):
                # En passant - captured pawn is on different square
                ep_file = chess.square_file(move.to_square)
                ep_rank = chess.square_rank(move.from_square)
                captured_square = chess.square_name(chess.square(ep_file, ep_rank))
            else:
                captured_square = to_sq

            # Remove captured piece: pick up, move to capture zone, drop
            pick_piece(square_to_xyz(captured_square), square=captured_square)
            _drop_at_capture_zone()

        # Check for obstacles in path (warning only)
        from_xyz = square_to_xyz(from_sq)
        to_xyz = square_to_xyz(to_sq)
        check_collision_warning(from_xyz, to_xyz)

        # Move the piece
        pick_piece(from_xyz, square=from_sq)
        place_piece(to_xyz, square=to_sq)

        # Handle castling - move rook too
        if is_castling:
            from_file = chess.square_file(move.from_square)
            to_file = chess.square_file(move.to_square)
            rank = chess.square_rank(move.from_square)

            if to_file > from_file:  # Kingside
                rook_from = chess.square_name(chess.square(7, rank))
                rook_to = chess.square_name(chess.square(5, rank))
            else:  # Queenside
                rook_from = chess.square_name(chess.square(0, rank))
                rook_to = chess.square_name(chess.square(3, rank))

            pick_piece(square_to_xyz(rook_from), square=rook_from)
            place_piece(square_to_xyz(rook_to), square=rook_to)

        # Go home
        if robot:
            robot.arm.go_to_home_pose()

        last_action = f"move:{uci_move}"
        last_action_time = time.time()

        return MoveResponse(
            success=True,
            message=f"Move {uci_move} executed",
            move=uci_move,
            from_square=from_sq,
            to_square=to_sq,
            was_capture=is_capture,
        )

    except Exception as e:
        return MoveResponse(
            success=False,
            error=str(e),
            move=request.uci_move,
        )


@app.post("/pick", response_model=MoveResponse)
async def pick_from_square(request: SquareRequest):
    """Pick up piece from a square."""
    try:
        xyz = square_to_xyz(request.square)
        pick_piece(xyz, grasp_height=request.grasp_height, square=request.square)
        return MoveResponse(
            success=True,
            message=f"Picked piece from {request.square}",
            from_square=request.square,
        )
    except Exception as e:
        return MoveResponse(success=False, error=str(e))


@app.post("/place", response_model=MoveResponse)
async def place_on_square(request: SquareRequest):
    """Place piece on a square."""
    try:
        xyz = square_to_xyz(request.square)
        place_piece(xyz, grasp_height=request.grasp_height, square=request.square)
        return MoveResponse(
            success=True,
            message=f"Placed piece on {request.square}",
            to_square=request.square,
        )
    except Exception as e:
        return MoveResponse(success=False, error=str(e))


@app.post("/arm/home", response_model=MoveResponse)
async def go_home():
    """Move robot to home position."""
    global last_action, last_action_time

    try:
        if robot:
            # Set safe speed for home movement (2 seconds)
            robot.arm.set_trajectory_time(moving_time=2.0 * SPEED_FACTOR, accel_time=0.5 * SPEED_FACTOR)
            robot.arm.go_to_home_pose()
            time.sleep(2.5 * SPEED_FACTOR)
        last_action = "home"
        last_action_time = time.time()
        return MoveResponse(success=True, message="Moved to home position")
    except Exception as e:
        return MoveResponse(success=False, error=str(e))


@app.post("/gripper/open", response_model=MoveResponse)
async def open_gripper():
    """Open the gripper."""
    try:
        if robot:
            robot.gripper.open()
        return MoveResponse(success=True, message="Gripper opened")
    except Exception as e:
        return MoveResponse(success=False, error=str(e))


@app.post("/gripper/close", response_model=MoveResponse)
async def close_gripper():
    """Close the gripper."""
    try:
        if robot:
            robot.gripper.close()
        return MoveResponse(success=True, message="Gripper closed")
    except Exception as e:
        return MoveResponse(success=False, error=str(e))


@app.post("/arm/sleep", response_model=MoveResponse)
async def go_sleep():
    """Move robot to sleep position."""
    global last_action, last_action_time

    try:
        if robot:
            # Set safe speed for sleep movement (2 seconds)
            robot.arm.set_trajectory_time(moving_time=2.0 * SPEED_FACTOR, accel_time=0.5 * SPEED_FACTOR)
            robot.arm.go_to_sleep_pose()
            time.sleep(2.5 * SPEED_FACTOR)
        last_action = "sleep"
        last_action_time = time.time()
        return MoveResponse(success=True, message="Moved to sleep position")
    except Exception as e:
        return MoveResponse(success=False, error=str(e))


@app.post("/arm/work", response_model=MoveResponse)
async def go_work():
    """Move robot to saved work position (from waypoints with tag 'work_position')."""
    global last_action, last_action_time

    wp = get_work_position()
    if wp is None:
        return MoveResponse(success=False, error="No work position saved. Save a waypoint with tag 'work_position' first.")

    try:
        if robot:
            robot.arm.set_trajectory_time(moving_time=2.0 * SPEED_FACTOR, accel_time=0.5 * SPEED_FACTOR)
            robot.arm.set_joint_positions(wp)
            time.sleep(2.5 * SPEED_FACTOR)
        last_action = "work"
        last_action_time = time.time()
        return MoveResponse(success=True, message="Moved to work position")
    except Exception as e:
        return MoveResponse(success=False, error=str(e))


@app.post("/arm/work/set", response_model=MoveResponse)
async def set_work_position():
    """Save current position as work position (waypoint with tag 'work_position')."""
    try:
        if robot is None:
            return MoveResponse(success=False, error="Robot not connected")

        # Capture current joint positions
        robot.arm.capture_joint_positions()
        joint_cmds = robot.arm.get_joint_commands()
        joints = joint_cmds.tolist() if hasattr(joint_cmds, 'tolist') else list(joint_cmds)
        joints_dict = {n: float(j) for n, j in zip(JOINT_NAMES, joints)}

        # Save as waypoint with tag
        waypoint = WaypointData(
            name="work",
            joints=joints_dict,
            gripper=0.5,
            tag="work_position",
            timestamp=time.time()
        )
        waypoints = load_waypoints()
        # Clear tag from any other waypoint
        for wn, wd in waypoints.items():
            if wd.get('tag') == 'work_position':
                wd['tag'] = None
        waypoints["work"] = waypoint.model_dump()
        save_waypoints(waypoints)

        print(f"[Work Position] Saved as waypoint 'work': {joints}")
        return MoveResponse(success=True, message=f"Work position saved: {[round(j, 3) for j in joints]}")
    except Exception as e:
        return MoveResponse(success=False, error=str(e))


@app.post("/arm/vision", response_model=MoveResponse)
async def go_vision():
    """Move robot to saved vision position (from waypoints with tag 'vision_position')."""
    global last_action, last_action_time

    vp = get_vision_position()
    if vp is None:
        return MoveResponse(success=False, error="No vision position saved. Save a waypoint with tag 'vision_position' first.")

    try:
        if robot:
            robot.arm.set_trajectory_time(moving_time=2.0 * SPEED_FACTOR, accel_time=0.5 * SPEED_FACTOR)
            robot.arm.set_joint_positions(vp)
            time.sleep(2.5 * SPEED_FACTOR)
        last_action = "vision"
        last_action_time = time.time()
        return MoveResponse(success=True, message="Moved to vision position")
    except Exception as e:
        return MoveResponse(success=False, error=str(e))


@app.post("/arm/hover_square", response_model=MoveResponse)
async def hover_square(request: HoverSquareRequest):
    """Move robot to hover above a square (for teach mode demonstrations)."""
    try:
        square = request.square.lower()
        if len(square) != 2 or square[0] not in "abcdefgh" or square[1] not in "12345678":
            return MoveResponse(success=False, error=f"Invalid square: {request.square}")

        # Resolve XY position from taught squares or grid calculation
        if square in _square_positions:
            sp = _square_positions[square]
            xyz = np.array([sp['x'], sp['y'], sp['z']])
        else:
            xyz = square_to_xyz(square)

        hover_z = (board_surface_z if board_surface_z is not None else xyz[2]) + 0.100

        # Always compute pitch for the HOVER height (not board level)
        # Taught pitch is for grasping at board level — different Z needs different pitch
        pitch = find_reachable_pitch(xyz[0], xyz[1], hover_z)

        print(f"[Hover] {square}: x={xyz[0]:.3f} y={xyz[1]:.3f} z={hover_z:.3f} pitch={math.degrees(pitch):.0f}°")

        if robot is None:
            print(f"[Mock] Hover above {square} at z={hover_z:.4f}")
            return MoveResponse(success=True, message=f"[Mock] Hovering above {square}")

        # Deliberate, visible movement for teaching demo (not speed-adjusted)
        result = robot.arm.set_ee_pose_components(
            x=xyz[0], y=xyz[1], z=hover_z,
            pitch=pitch, moving_time=1.5
        )
        print(f"[Hover] {square} IK result: {result}")
        time.sleep(1.5)

        return MoveResponse(success=True, message=f"Hovering above {square}")
    except Exception as e:
        return MoveResponse(success=False, error=str(e))


@app.post("/arm/vision/set", response_model=MoveResponse)
async def set_vision_position():
    """Save current position as vision position (waypoint with tag 'vision_position')."""
    try:
        if robot is None:
            return MoveResponse(success=False, error="Robot not connected")

        # Capture current joint positions
        robot.arm.capture_joint_positions()
        joint_cmds = robot.arm.get_joint_commands()
        joints = joint_cmds.tolist() if hasattr(joint_cmds, 'tolist') else list(joint_cmds)
        joints_dict = {n: float(j) for n, j in zip(JOINT_NAMES, joints)}

        # Save as waypoint with tag
        waypoint = WaypointData(
            name="vision",
            joints=joints_dict,
            gripper=0.5,
            tag="vision_position",
            timestamp=time.time()
        )
        waypoints = load_waypoints()
        for wn, wd in waypoints.items():
            if wd.get('tag') == 'vision_position':
                wd['tag'] = None
        waypoints["vision"] = waypoint.model_dump()
        save_waypoints(waypoints)

        print(f"[Vision Position] Saved as waypoint 'vision': {joints}")
        return MoveResponse(success=True, message=f"Vision position saved: {[round(j, 3) for j in joints]}")
    except Exception as e:
        return MoveResponse(success=False, error=str(e))


# ── Capture Zone Position ───────────────────────────────────────────

def load_capture_zone_position():
    """Load capture zone position from YAML file."""
    global capture_zone_position
    if CAPTURE_ZONE_FILE.exists():
        try:
            with open(CAPTURE_ZONE_FILE, 'r') as f:
                data = yaml.safe_load(f)
                if data and 'x' in data:
                    capture_zone_position = data
                    print(f"[Capture Zone] Loaded: ({data['x']:.4f}, {data['y']:.4f}, {data['z']:.4f})")
        except Exception as e:
            print(f"[Capture Zone] Failed to load: {e}")


# Load on startup
load_capture_zone_position()


@app.get("/capture_zone")
async def get_capture_zone():
    """Return the current capture zone position."""
    return {"success": True, "capture_zone": capture_zone_position}


@app.post("/capture_zone/set", response_model=MoveResponse)
async def set_capture_zone():
    """Save current joint positions as capture zone waypoint."""
    try:
        if robot is None:
            return MoveResponse(success=False, error="Robot not connected")

        robot.arm.capture_joint_positions()
        joint_cmds = robot.arm.get_joint_commands()
        joints = joint_cmds.tolist() if hasattr(joint_cmds, 'tolist') else list(joint_cmds)
        joints_dict = {n: float(j) for n, j in zip(JOINT_NAMES, joints)}

        # Save as waypoint with capture_zone tag
        waypoint = WaypointData(
            name="capture_zone",
            joints=joints_dict,
            tag="capture_zone",
            timestamp=time.time(),
        )

        waypoints = load_waypoints()
        # Clear any existing capture_zone tag
        for wn, wd in waypoints.items():
            if wd.get('tag') == 'capture_zone':
                wd['tag'] = None
        waypoints["capture_zone"] = waypoint.model_dump()
        save_waypoints(waypoints)

        print(f"[Capture Zone] Saved as waypoint: {joints}")
        return MoveResponse(
            success=True,
            message=f"Capture zone saved: ({x:.4f}, {y:.4f}, {z:.4f})",
        )
    except Exception as e:
        return MoveResponse(success=False, error=str(e))


@app.post("/arm/capture_zone", response_model=MoveResponse)
async def go_capture_zone():
    """Move robot to saved capture zone position (joint-based waypoint)."""
    global last_action, last_action_time

    cz_joints = get_capture_zone_joints()
    if cz_joints is None:
        return MoveResponse(success=False, error="No capture_zone waypoint. Save a waypoint with tag 'capture_zone'.")

    try:
        if robot:
            robot.arm.set_joint_positions(cz_joints, moving_time=2.0, accel_time=0.5)
            time.sleep(2.0)
        last_action = "capture_zone"
        last_action_time = time.time()
        return MoveResponse(success=True, message="Moved to capture zone")
    except Exception as e:
        return MoveResponse(success=False, error=str(e))


# ── Promotion Queen Position ──────────────────────────────────────────

def load_promotion_queen_position():
    """Load promotion queen position from YAML file."""
    global promotion_queen_position
    if PROMOTION_QUEEN_FILE.exists():
        try:
            with open(PROMOTION_QUEEN_FILE, 'r') as f:
                data = yaml.safe_load(f)
                if data and 'x' in data:
                    promotion_queen_position = data
                    print(f"[Promotion Queen] Loaded: ({data['x']:.4f}, {data['y']:.4f}, {data['z']:.4f})")
        except Exception as e:
            print(f"[Promotion Queen] Failed to load: {e}")


# Load on startup
load_promotion_queen_position()


@app.get("/promotion_queen")
async def get_promotion_queen():
    """Return the current promotion queen position."""
    return {"success": True, "promotion_queen": promotion_queen_position}


@app.post("/promotion_queen/set", response_model=MoveResponse)
async def set_promotion_queen():
    """Save current end-effector position as promotion queen position (where a spare queen sits)."""
    global promotion_queen_position

    try:
        if robot is None:
            return MoveResponse(success=False, error="Robot not connected")

        # Read current EE pose
        robot.arm.capture_joint_positions()
        ee_pose = robot.arm.get_ee_pose()
        x = float(ee_pose[0, 3])
        y = float(ee_pose[1, 3])
        z = float(ee_pose[2, 3])

        # Get current pitch from joint positions
        joint_cmds = robot.arm.get_joint_commands()
        joints = joint_cmds.tolist() if hasattr(joint_cmds, 'tolist') else list(joint_cmds)
        pitch = joints[3] if len(joints) > 3 else 1.5708

        promotion_queen_position = {'x': x, 'y': y, 'z': z, 'pitch': pitch}

        # Save to file
        with open(PROMOTION_QUEEN_FILE, 'w') as f:
            yaml.dump(promotion_queen_position, f, default_flow_style=False)

        print(f"[Promotion Queen] Saved: ({x:.4f}, {y:.4f}, {z:.4f}, pitch={pitch:.4f})")
        return MoveResponse(
            success=True,
            message=f"Promotion queen position saved: ({x:.4f}, {y:.4f}, {z:.4f})",
        )
    except Exception as e:
        return MoveResponse(success=False, error=str(e))


@app.post("/arm/promotion_queen", response_model=MoveResponse)
async def go_promotion_queen():
    """Move robot to saved promotion queen position."""
    global last_action, last_action_time, promotion_queen_position

    if promotion_queen_position is None:
        load_promotion_queen_position()

    if promotion_queen_position is None:
        return MoveResponse(success=False, error="No promotion queen position saved. Set it first.")

    try:
        if robot:
            pq = promotion_queen_position
            robot.arm.set_trajectory_time(moving_time=2.0 * SPEED_FACTOR, accel_time=0.5 * SPEED_FACTOR)
            robot.arm.set_ee_pose_components(
                x=pq['x'], y=pq['y'], z=pq['z'],
                pitch=pq.get('pitch', 1.5708),
                moving_time=2.0 * SPEED_FACTOR,
            )
        last_action = "promotion_queen"
        last_action_time = time.time()
        return MoveResponse(success=True, message="Moved to promotion queen position")
    except Exception as e:
        return MoveResponse(success=False, error=str(e))


@app.post("/pickup_from_promotion_queen", response_model=MoveResponse)
async def pickup_from_promotion_queen(request: PromotionPickupRequest):
    """
    Pick up a spare piece from the taught promotion_queen_position.

    Sequence:
      1. L1 work + gripper open
      2. Move above promotion queen pos (approach height)
      3. Lower to grasp height based on piece type
      4. Gripper close
      5. Lift back to approach height
      6. L1 work (piece in gripper, ready for manual_place)
    """
    global last_action, last_action_time

    try:
        # --- Validate promotion queen position ---
        if promotion_queen_position is None:
            load_promotion_queen_position()
        if promotion_queen_position is None:
            return MoveResponse(success=False, error="No promotion queen position saved. Set it first.")

        pq = promotion_queen_position
        pq_x = pq['x']
        pq_y = pq['y']
        pq_z = pq['z']
        pq_pitch = pq.get('pitch', 1.5708)

        # --- Piece type ---
        piece_type = (request.piece_type or "queen").lower()
        piece_h = PIECE_HEIGHTS.get(piece_type, PIECE_HEIGHTS["queen"])
        print(f"[promotion_pickup] piece={piece_type}, height={piece_h*1000:.0f}mm")

        # --- Heights ---
        approach_z = pq_z + 0.100
        grasp_z = pq_z + piece_h * 0.50
        lift_z = pq_z + 0.120

        print(f"[promotion_pickup] pos=({pq_x:.4f}, {pq_y:.4f}, {pq_z:.4f})")
        print(f"[promotion_pickup] approach_z={approach_z:.4f}, grasp_z={grasp_z:.4f}")

        # --- Mock mode ---
        if robot is None:
            print(f"[Mock] pickup_from_promotion_queen {piece_type}")
            last_action = "promotion_pickup"
            last_action_time = time.time()
            return MoveResponse(success=True, message=f"[Mock] Picked up {piece_type} from promotion pos")

        # --- Work position ---
        wp = get_work_position()
        if wp is None:
            return MoveResponse(success=False, error="No work position saved")

        # --- Helper: move with pitch tolerance ---
        def move_with_pitch_tolerance(x, y, z, preferred_pitch, moving_time=1.5):
            import math
            PITCH_TOL = math.radians(45)
            pitches_to_try = [preferred_pitch]
            step = math.radians(5)
            for offset in range(1, int(PITCH_TOL / step) + 1):
                pitches_to_try.append(preferred_pitch + offset * step)
                pitches_to_try.append(preferred_pitch - offset * step)
            for pitch in pitches_to_try:
                test_result = robot.arm.set_ee_pose_components(
                    x=x, y=y, z=z, pitch=pitch,
                    blocking=False, execute=False,
                )
                if isinstance(test_result, tuple):
                    test_ok = test_result[1]
                else:
                    test_ok = bool(test_result)
                if test_ok:
                    if abs(pitch - preferred_pitch) > 0.01:
                        print(f"  [pitch search] Using {math.degrees(pitch):.1f}")
                    robot.arm.set_ee_pose_components(
                        x=x, y=y, z=z, pitch=pitch,
                        moving_time=moving_time, blocking=True, execute=True,
                    )
                    return True
            robot.arm.set_ee_pose_components(
                x=x, y=y, z=z, pitch=preferred_pitch,
                moving_time=moving_time,
            )
            return False

        # Step 1: L1 work + gripper open
        print(f"[promotion_pickup] Step 1/6: L1 work + gripper open")
        robot.arm.set_trajectory_time(moving_time=2.0 * SPEED_FACTOR, accel_time=0.5 * SPEED_FACTOR)
        robot.arm.set_joint_positions(wp)
        time.sleep(0.3 * SPEED_FACTOR)
        robot.gripper.open(delay=0.5)
        time.sleep(0.3 * SPEED_FACTOR)

        # Step 2: Move above promotion queen pos
        print(f"[promotion_pickup] Step 2/6: Above promotion pos z={approach_z:.4f}")
        move_with_pitch_tolerance(pq_x, pq_y, approach_z, pq_pitch, 1.5 * SPEED_FACTOR)
        time.sleep(0.3 * SPEED_FACTOR)

        # Step 3: Lower to grasp
        print(f"[promotion_pickup] Step 3/6: Grasp z={grasp_z:.4f}")
        move_with_pitch_tolerance(pq_x, pq_y, grasp_z, pq_pitch, 1.2 * SPEED_FACTOR)
        time.sleep(0.3 * SPEED_FACTOR)

        # Step 4: Close gripper
        print(f"[promotion_pickup] Step 4/6: Gripper close")
        robot.gripper.close(delay=0.8)
        time.sleep(0.5 * SPEED_FACTOR)

        # Step 5: Lift
        print(f"[promotion_pickup] Step 5/6: Lift z={lift_z:.4f}")
        move_with_pitch_tolerance(pq_x, pq_y, lift_z, pq_pitch, 1.2 * SPEED_FACTOR)
        time.sleep(0.3 * SPEED_FACTOR)

        # Step 6: L1 work
        print(f"[promotion_pickup] Step 6/6: L1 work (piece in gripper)")
        robot.arm.set_joint_positions(wp)
        time.sleep(0.3 * SPEED_FACTOR)

        last_action = "promotion_pickup"
        last_action_time = time.time()
        return MoveResponse(success=True, message=f"Picked up {piece_type} from promotion pos")

    except Exception as e:
        # Try to recover to work position
        try:
            recovery_wp = get_work_position()
            if robot and recovery_wp:
                robot.arm.set_joint_positions(recovery_wp)
        except Exception:
            pass
        return MoveResponse(success=False, error=str(e))


@app.post("/arm/move_to_xyz", response_model=MoveResponse)
async def move_to_xyz(request: MoveToXYZRequest):
    """
    Move robot end-effector to XYZ position with gripper pointing down.

    Default pitch is -90° (pointing down), which is ideal for:
    - Picking up objects from table
    - Calibration (touching tags)
    - Any top-down approach
    """
    global last_action, last_action_time

    try:
        if robot is None:
            return MoveResponse(success=False, error="Robot not connected")

        # Set movement speed
        robot.arm.set_trajectory_time(
            moving_time=request.moving_time,
            accel_time=request.moving_time * 0.3
        )


        if request.auto_orientation:
            print(f"[move_to_xyz] Moving to x={request.x:.4f}, y={request.y:.4f}, z={request.z:.4f} (auto orientation)")
            # Let IK choose best orientation - don't constrain pitch/roll
            result = robot.arm.set_ee_pose_components(
                x=request.x,
                y=request.y,
                z=request.z,
                blocking=True,
                execute=True,
            )
            used_pitch = None
        elif request.pitch_tolerance > 0:
            # Try exact pitch first, then search within tolerance
            print(f"[move_to_xyz] Moving to x={request.x:.4f}, y={request.y:.4f}, z={request.z:.4f}, pitch={request.pitch} (tolerance={math.degrees(request.pitch_tolerance):.1f}°)")

            # Build list of pitches to try: exact, then +/-1°, +/-2°, etc.
            pitches_to_try = [request.pitch]
            step = math.radians(5)  # 1 degree steps
            for offset in range(1, int(request.pitch_tolerance / step) + 1):
                pitches_to_try.append(request.pitch + offset * step)
                pitches_to_try.append(request.pitch - offset * step)

            result = None
            used_pitch = None
            for pitch in pitches_to_try:
                test_result = robot.arm.set_ee_pose_components(
                    x=request.x,
                    y=request.y,
                    z=request.z,
                    roll=request.roll,
                    pitch=pitch,
                    blocking=False,  # Don't execute yet, just check IK
                    execute=False,
                )
                # Check if IK succeeded
                if isinstance(test_result, tuple):
                    test_success = test_result[1]
                else:
                    test_success = test_result

                if test_success:
                    print(f"[move_to_xyz] Found valid pitch: {math.degrees(pitch):.1f}° (offset: {math.degrees(pitch - request.pitch):.1f}°)")
                    used_pitch = pitch
                    # Now actually execute the move
                    result = robot.arm.set_ee_pose_components(
                        x=request.x,
                        y=request.y,
                        z=request.z,
                        roll=request.roll,
                        pitch=pitch,
                        blocking=True,
                        execute=True,
                    )
                    break

            if used_pitch is None:
                print(f"[move_to_xyz] No valid pitch found within tolerance")
                result = (None, False)
        else:
            print(f"[move_to_xyz] Moving to x={request.x:.4f}, y={request.y:.4f}, z={request.z:.4f}, pitch={request.pitch}, roll={request.roll}")
            # Move to position with specified orientation
            # pitch = +1.5708 rad = +90° = gripper pointing down
            result = robot.arm.set_ee_pose_components(
                x=request.x,
                y=request.y,
                z=request.z,
                roll=request.roll,
                pitch=request.pitch,
                blocking=True,
                execute=True,
            )
            used_pitch = request.pitch

        # Handle return value - could be tuple (joints, success) or just bool
        if isinstance(result, tuple):
            success = result[1]
            print(f"[move_to_xyz] IK result: joints={result[0]}, success={success}")
        else:
            success = result
            print(f"[move_to_xyz] Result: {success}")

        if success:
            last_action = f"move_xyz ({request.x:.3f}, {request.y:.3f}, {request.z:.3f})"
            last_action_time = time.time()
            if used_pitch is not None:
                pitch_deg = math.degrees(used_pitch)
                return MoveResponse(
                    success=True,
                    message=f"Moved to ({request.x:.3f}, {request.y:.3f}, {request.z:.3f}) m, pitch={pitch_deg:.1f}°"
                )
            else:
                return MoveResponse(
                    success=True,
                    message=f"Moved to ({request.x:.3f}, {request.y:.3f}, {request.z:.3f}) m (auto orientation)"
                )
        else:
            return MoveResponse(
                success=False,
                error="IK solution not found - position may be out of reach"
            )

    except Exception as e:
        return MoveResponse(success=False, error=str(e))


@app.get("/arm/positions", response_model=JointPositionsResponse)
async def get_positions():
    """Get current joint positions and end-effector pose."""
    try:
        if robot is None:
            return JointPositionsResponse(
                success=True,
                joints={name: 0.0 for name in JOINT_NAMES},
                ee_pose={"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            )

        # Sync with actual encoder positions and get joint commands
        robot.arm.capture_joint_positions()
        joint_positions = robot.arm.get_joint_commands()
        joints = {name: float(pos) for name, pos in zip(JOINT_NAMES, joint_positions)}

        # Get EE pose (4x4 transformation matrix)
        ee_pose_matrix = robot.arm.get_ee_pose()
        x, y, z = ee_pose_matrix[0, 3], ee_pose_matrix[1, 3], ee_pose_matrix[2, 3]

        # Extract orientation
        R = ee_pose_matrix[:3, :3]
        roll, pitch = rotation_matrix_to_euler(R)

        return JointPositionsResponse(
            success=True,
            joints=joints,
            ee_pose={"x": float(x), "y": float(y), "z": float(z), "roll": float(roll), "pitch": float(pitch)},
        )
    except Exception as e:
        return JointPositionsResponse(success=False, error=str(e))


@app.post("/arm/jog/joint", response_model=MoveResponse)
async def jog_joint(request: JointJogRequest):
    """Jog a single joint by a step amount."""
    global last_action, last_action_time

    print(f"[Joint Jog] Request: joint={request.joint}, step={request.step}")
    print(f"[Joint Jog] robot is None: {robot is None}, robot_connected: {robot_connected}")

    try:
        joint = request.joint.lower()
        if joint not in JOINT_NAMES:
            return MoveResponse(
                success=False,
                error=f"Invalid joint '{joint}'. Valid: {JOINT_NAMES}"
            )

        if robot is None:
            return MoveResponse(
                success=True,
                message=f"[Mock] Jogged {joint} by {request.step:.3f} rad"
            )

        # Sync internal commands with actual encoder positions first
        robot.arm.capture_joint_positions()

        # Get current commanded positions (now synced with actual)
        current_positions = robot.arm.get_joint_commands()
        joint_idx = JOINT_NAMES.index(joint)
        current_pos = current_positions[joint_idx]
        new_position = current_pos + request.step
        print(f"[Joint Jog] Joint: {joint}, Index: {joint_idx}")
        print(f"[Joint Jog] Current: {current_pos:.4f}, Step: {request.step:.4f}, New: {new_position:.4f}")

        # Move joint (0.5s motion, blocking so each click completes)
        success = robot.arm.set_single_joint_position(joint, new_position, moving_time=0.5, blocking=True)
        print(f"[Joint Jog] set_single_joint_position returned: {success}")

        last_action = f"jog:{joint}"
        last_action_time = time.time()

        return MoveResponse(
            success=True,
            message=f"Jogged {joint} by {request.step:.3f} rad to {new_position:.3f}"
        )
    except Exception as e:
        print(f"[Joint Jog] Error: {e}")
        import traceback
        traceback.print_exc()
        return MoveResponse(success=False, error=str(e))


def rotation_matrix_to_euler(R):
    """Extract roll, pitch from rotation matrix (for 5-DOF arm, yaw is ignored)."""
    # Extract pitch (rotation around Y axis)
    pitch = math.atan2(-R[2, 0], math.sqrt(R[0, 0]**2 + R[1, 0]**2))
    # Extract roll (rotation around X axis)
    if abs(math.cos(pitch)) > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
    else:
        roll = 0
    return roll, pitch


@app.post("/arm/jog/cartesian", response_model=MoveResponse)
async def jog_cartesian(request: CartesianJogRequest):
    """Jog end-effector in cartesian space while preserving orientation."""
    global last_action, last_action_time

    print(f"[Cartesian Jog] Request: axis={request.axis}, step={request.step}")
    print(f"[Cartesian Jog] robot is None: {robot is None}, robot_connected: {robot_connected}")

    try:
        axis = request.axis.lower()
        if axis not in ["x", "y", "z"]:
            return MoveResponse(
                success=False,
                error=f"Invalid axis '{axis}'. Valid: x, y, z"
            )

        if robot is None:
            return MoveResponse(
                success=True,
                message=f"[Mock] Jogged {axis} by {request.step*100:.1f} cm"
            )

        # Get current EE pose (4x4 transformation matrix)
        ee_pose = robot.arm.get_ee_pose()
        old_x, old_y, old_z = ee_pose[0, 3], ee_pose[1, 3], ee_pose[2, 3]

        # Extract current orientation (roll, pitch) from rotation matrix
        R = ee_pose[:3, :3]
        roll, pitch = rotation_matrix_to_euler(R)

        print(f"[Cartesian Jog] Current EE pose: x={old_x:.4f}, y={old_y:.4f}, z={old_z:.4f}")
        print(f"[Cartesian Jog] Current orientation: roll={roll:.4f}, pitch={pitch:.4f}")

        # Calculate new position
        x, y, z = old_x, old_y, old_z
        if axis == "x":
            x += request.step
        elif axis == "y":
            y += request.step
        elif axis == "z":
            z += request.step

        print(f"[Cartesian Jog] Target EE pose: x={x:.4f}, y={y:.4f}, z={z:.4f}")
        print(f"[Cartesian Jog] Delta: dx={x-old_x:.4f}, dy={y-old_y:.4f}, dz={z-old_z:.4f}")

        # Move to new position while preserving orientation (0.5s motion, blocking)
        result = robot.arm.set_ee_pose_components(
            x=x, y=y, z=z,
            roll=roll, pitch=pitch,
            moving_time=0.5,
            blocking=True
        )
        print(f"[Cartesian Jog] set_ee_pose_components returned: {result}")

        # Verify final position
        final_pose = robot.arm.get_ee_pose()
        final_x, final_y, final_z = final_pose[0, 3], final_pose[1, 3], final_pose[2, 3]
        print(f"[Cartesian Jog] Final EE pose: x={final_x:.4f}, y={final_y:.4f}, z={final_z:.4f}")

        last_action = f"jog:{axis}"
        last_action_time = time.time()

        return MoveResponse(
            success=True,
            message=f"Jogged {axis} by {request.step*100:.1f} cm"
        )
    except Exception as e:
        print(f"[Cartesian Jog] Error: {e}")
        import traceback
        traceback.print_exc()
        return MoveResponse(success=False, error=str(e))


@app.post("/gripper/set", response_model=MoveResponse)
async def set_gripper(request: GripperRequest):
    """Set gripper to specific position (0=closed, 1=open)."""
    try:
        pos = max(0.0, min(1.0, request.position))

        if robot is None:
            return MoveResponse(
                success=True,
                message=f"[Mock] Gripper set to {pos:.2f}"
            )

        # Get current gripper position first
        joint_states = robot.dxl.robot_get_joint_states()
        gripper_names = joint_states.name
        gripper_positions = joint_states.position
        current_pos = None
        for i, name in enumerate(gripper_names):
            if 'gripper' in name or 'finger' in name:
                current_pos = gripper_positions[i]
                print(f"[Gripper] Joint '{name}' current position: {current_pos}")

        gripper_pos = GRIPPER_CLOSED_M + pos * (GRIPPER_OPEN_M - GRIPPER_CLOSED_M)
        print(f"[Gripper] Request pos={pos}, calculated gripper_pos={gripper_pos}")

        # Switch to position mode, set position, switch back to PWM
        robot.dxl.robot_set_operating_modes('single', 'gripper', 'position')
        time.sleep(0.3 * SPEED_FACTOR)
        robot.dxl.robot_write_joint_command('gripper', gripper_pos)
        time.sleep(0.8 * SPEED_FACTOR)

        # Check position after move
        joint_states = robot.dxl.robot_get_joint_states()
        for i, name in enumerate(joint_states.name):
            if 'gripper' in name or 'finger' in name:
                print(f"[Gripper] After move, '{name}' position: {joint_states.position[i]}")

        robot.dxl.robot_set_operating_modes('single', 'gripper', 'pwm')
        time.sleep(0.3 * SPEED_FACTOR)

        return MoveResponse(
            success=True,
            message=f"Gripper set to {pos:.2f} (target={gripper_pos:.4f})"
        )
    except Exception as e:
        return MoveResponse(success=False, error=str(e))


# Waypoint helper functions
def load_waypoints() -> dict:
    """Load waypoints from file."""
    if WAYPOINTS_FILE.exists():
        try:
            with open(WAYPOINTS_FILE, 'r') as f:
                data = yaml.safe_load(f)
                return data if data else {}
        except Exception as e:
            print(f"[Robot Service] Error loading waypoints: {e}")
            return {}
    return {}


def save_waypoints(waypoints: dict):
    """Save waypoints to file."""
    try:
        with open(WAYPOINTS_FILE, 'w') as f:
            yaml.dump(waypoints, f, default_flow_style=False)
    except Exception as e:
        print(f"[Robot Service] Error saving waypoints: {e}")
        raise


def get_waypoint_by_tag(tag: str) -> Optional[dict]:
    """Find the first waypoint with the given tag. Returns waypoint dict or None."""
    waypoints = load_waypoints()
    for name, wp in waypoints.items():
        if wp.get('tag') == tag:
            return wp
    return None


def get_work_position() -> Optional[list]:
    """Get joint positions for the work_position waypoint."""
    wp = get_waypoint_by_tag('work_position')
    if wp and 'joints' in wp:
        joints = wp['joints']
        if isinstance(joints, dict):
            return [joints[n] for n in JOINT_NAMES]
        return joints
    return None


def get_vision_position() -> Optional[list]:
    """Get joint positions for the vision_position waypoint."""
    wp = get_waypoint_by_tag('vision_position')
    if wp and 'joints' in wp:
        joints = wp['joints']
        if isinstance(joints, dict):
            return [joints[n] for n in JOINT_NAMES]
        return joints
    return None


def get_capture_zone_joints() -> Optional[list]:
    """Get joint positions for the capture_zone waypoint."""
    wp = get_waypoint_by_tag('capture_zone')
    if wp and 'joints' in wp:
        joints = wp['joints']
        if isinstance(joints, dict):
            return [joints[n] for n in JOINT_NAMES]
        return joints
    return None


@app.get("/waypoints", response_model=WaypointListResponse)
async def list_waypoints():
    """List all saved waypoints with names and tags."""
    waypoints = load_waypoints()
    wp_list = [{"name": name, "tag": wp.get("tag")} for name, wp in waypoints.items()]
    return WaypointListResponse(
        success=True,
        waypoints=wp_list
    )


@app.post("/waypoints/save", response_model=WaypointResponse)
async def save_waypoint(request: WaypointSaveRequest):
    """Save current position as a named waypoint."""
    try:
        name = request.name.strip()
        if not name:
            return WaypointResponse(success=False, error="Waypoint name cannot be empty")

        # Get current joint positions (actual physical positions from servo feedback)
        if robot is None:
            joints = {n: 0.0 for n in JOINT_NAMES}
            gripper = 0.5
        else:
            joint_states = robot.dxl.robot_get_joint_states()
            # joint_states.name includes all joints; filter to arm joints only
            all_names = list(joint_states.name)
            all_positions = list(joint_states.position)
            joints = {}
            for jn in JOINT_NAMES:
                if jn in all_names:
                    joints[jn] = float(all_positions[all_names.index(jn)])
                else:
                    joints[jn] = 0.0
            print(f"[Waypoint] Saving '{name}' (from servo feedback): {joints}")
            # Get gripper position (estimate based on last command, or default)
            gripper = 0.5  # Default, as gripper position isn't easily readable

        # If tag provided, clear it from any other waypoint first (unique tag)
        tag = getattr(request, 'tag', None)

        # Create waypoint data
        waypoint = WaypointData(
            name=name,
            joints=joints,
            gripper=gripper,
            tag=tag,
            timestamp=time.time()
        )

        # Load existing, clear tag from others if needed, add new, save
        waypoints = load_waypoints()
        if tag:
            for wn, wd in waypoints.items():
                if wd.get('tag') == tag and wn != name:
                    wd['tag'] = None
        waypoints[name] = waypoint.model_dump()
        save_waypoints(waypoints)

        return WaypointResponse(success=True, waypoint=waypoint)
    except Exception as e:
        return WaypointResponse(success=False, error=str(e))


@app.post("/waypoints/load/{name}", response_model=WaypointResponse)
async def goto_waypoint(name: str):
    """Move robot to a saved waypoint."""
    global last_action, last_action_time

    try:
        waypoints = load_waypoints()

        if name not in waypoints:
            return WaypointResponse(success=False, error=f"Waypoint '{name}' not found")

        wp_data = waypoints[name]
        waypoint = WaypointData(**wp_data)

        if robot is None:
            print(f"[Mock] Moving to waypoint '{name}': {waypoint.joints}")
        else:
            # Move to joint positions
            joint_positions = [waypoint.joints[n] for n in JOINT_NAMES]
            robot.arm.set_joint_positions(joint_positions, moving_time=2.0 * SPEED_FACTOR)
            time.sleep(2.5 * SPEED_FACTOR)

            # Set gripper
            gripper_pos = GRIPPER_CLOSED_M + waypoint.gripper * (GRIPPER_OPEN_M - GRIPPER_CLOSED_M)
            robot.gripper.set_pressure(1.0)
            robot.gripper.go_to_position(gripper_pos)

        last_action = f"waypoint:{name}"
        last_action_time = time.time()

        return WaypointResponse(success=True, waypoint=waypoint)
    except Exception as e:
        return WaypointResponse(success=False, error=str(e))


@app.delete("/waypoints/{name}", response_model=MoveResponse)
async def delete_waypoint(name: str):
    """Delete a saved waypoint."""
    try:
        waypoints = load_waypoints()

        if name not in waypoints:
            return MoveResponse(success=False, error=f"Waypoint '{name}' not found")

        del waypoints[name]
        save_waypoints(waypoints)

        return MoveResponse(success=True, message=f"Waypoint '{name}' deleted")
    except Exception as e:
        return MoveResponse(success=False, error=str(e))


class RebootRequest(BaseModel):
    """Request to reboot motors."""
    name: str = Field("all", description="Motor group or joint name (all, arm, or joint name)")
    enable: bool = Field(True, description="Enable torque after reboot")
    smart_reboot: bool = Field(True, description="Only reboot motors in error state")


@app.post("/motors/reboot", response_model=MoveResponse)
async def reboot_motors(request: RebootRequest = RebootRequest()):
    """
    Reboot motors to clear error states (like overload/overheat).
    Motor ID 4 (wrist_angle) commonly enters error state.
    WARNING: Robot may collapse - ensure it's in a safe position!
    """
    if robot is None:
        return MoveResponse(success=True, message="[Mock] Motors rebooted")

    try:
        import rospy
        from interbotix_xs_msgs.srv import Reboot

        # Wait for service
        service_name = '/rx200/reboot_motors'
        rospy.wait_for_service(service_name, timeout=5.0)
        reboot_srv = rospy.ServiceProxy(service_name, Reboot)

        # Call reboot service
        # cmd_type: 'group' for group, 'single' for single joint
        if request.name in ['all', 'arm', 'gripper']:
            cmd_type = 'group'
        else:
            cmd_type = 'single'

        result = reboot_srv(
            cmd_type=cmd_type,
            name=request.name,
            enable=request.enable,
            smart_reboot=request.smart_reboot
        )

        print(f"[Motors] Rebooted {request.name}, enable={request.enable}, smart={request.smart_reboot}")
        return MoveResponse(
            success=True,
            message=f"Rebooted motors: {request.name} (torque {'enabled' if request.enable else 'disabled'})"
        )
    except rospy.ROSException as e:
        print(f"[Motors] Service timeout: {e}")
        return MoveResponse(success=False, error=f"Service not available: {e}")
    except Exception as e:
        print(f"[Motors] Reboot error: {e}")
        return MoveResponse(success=False, error=str(e))


class TorqueRequest(BaseModel):
    """Request to enable/disable motor torque."""
    name: str = Field("all", description="Motor group or joint name")
    enable: bool = Field(True, description="Enable or disable torque")


@app.post("/motors/torque", response_model=MoveResponse)
async def set_torque(request: TorqueRequest):
    """
    Enable or disable motor torque.
    Use this to re-enable torque after a motor enters error state.
    WARNING: Disabling torque will cause robot to collapse!
    """
    if robot is None:
        return MoveResponse(success=True, message=f"[Mock] Torque {'enabled' if request.enable else 'disabled'}")

    try:
        import rospy
        from interbotix_xs_msgs.srv import TorqueEnable

        service_name = '/rx200/torque_enable'
        rospy.wait_for_service(service_name, timeout=5.0)
        torque_srv = rospy.ServiceProxy(service_name, TorqueEnable)

        if request.name in ['all', 'arm', 'gripper']:
            cmd_type = 'group'
        else:
            cmd_type = 'single'

        result = torque_srv(
            cmd_type=cmd_type,
            name=request.name,
            enable=request.enable
        )

        action = 'enabled' if request.enable else 'disabled'
        print(f"[Motors] Torque {action} for {request.name}")
        return MoveResponse(success=True, message=f"Torque {action} for {request.name}")
    except rospy.ROSException as e:
        print(f"[Motors] Service timeout: {e}")
        return MoveResponse(success=False, error=f"Service not available: {e}")
    except Exception as e:
        print(f"[Motors] Torque error: {e}")
        return MoveResponse(success=False, error=str(e))


@app.post("/motors/reboot/wrist", response_model=MoveResponse)
async def reboot_wrist_angle():
    """
    Quick shortcut to reboot wrist_angle motor (ID 4) which commonly loses torque.
    """
    return await reboot_motors(RebootRequest(name="wrist_angle", enable=True, smart_reboot=True))


# ==================== CALIBRATION ENDPOINTS ====================

def compute_rigid_transform(points_camera: np.ndarray, points_robot: np.ndarray) -> np.ndarray:
    """
    Compute rigid transformation (rotation + translation) from camera to robot frame.
    Uses SVD method for least-squares solution.

    Args:
        points_camera: Nx3 array of points in camera frame
        points_robot: Nx3 array of corresponding points in robot frame

    Returns:
        4x4 transformation matrix T where P_robot = T @ P_camera
    """
    assert points_camera.shape == points_robot.shape
    assert points_camera.shape[0] >= 3, "Need at least 3 points"

    # Compute centroids
    centroid_camera = np.mean(points_camera, axis=0)
    centroid_robot = np.mean(points_robot, axis=0)

    # Center the points
    camera_centered = points_camera - centroid_camera
    robot_centered = points_robot - centroid_robot

    # Compute covariance matrix
    H = camera_centered.T @ robot_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = centroid_robot - R @ centroid_camera

    # Build 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def load_calibration() -> Optional[np.ndarray]:
    """Load calibration from file."""
    global calibration_transform
    if CALIBRATION_FILE.exists():
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                data = yaml.safe_load(f)
                if data and 'transform' in data:
                    calibration_transform = np.array(data['transform'])
                    print(f"[Calibration] Loaded from {CALIBRATION_FILE}")
                    return calibration_transform
        except Exception as e:
            print(f"[Calibration] Error loading: {e}")
    return None


def save_calibration(transform: np.ndarray, points_camera: list, points_robot: list):
    """Save calibration to file."""
    try:
        data = {
            'transform': transform.tolist(),
            'points_camera': points_camera,
            'points_robot': points_robot,
            'timestamp': time.time(),
        }
        with open(CALIBRATION_FILE, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"[Calibration] Saved to {CALIBRATION_FILE}")
    except Exception as e:
        print(f"[Calibration] Error saving: {e}")
        raise


@app.get("/calibration/status", response_model=CalibrationStatus)
async def get_calibration_status():
    """Get current calibration status."""
    global calibration_transform

    # Try to load if not loaded
    if calibration_transform is None:
        load_calibration()

    points = []
    for i in range(len(calibration_points_camera)):
        tag_id = calibration_points_tag_ids[i] if i < len(calibration_points_tag_ids) else None
        points.append(CalibrationPointPair(
            camera=calibration_points_camera[i],
            robot=calibration_points_robot[i],
            tag_id=tag_id
        ))

    return CalibrationStatus(
        success=True,
        num_points=len(calibration_points_camera),
        points=points,
        is_calibrated=calibration_transform is not None,
        transform=calibration_transform.tolist() if calibration_transform is not None else None,
    )


@app.post("/calibration/clear", response_model=CalibrationStatus)
async def clear_calibration_points():
    """Clear all recorded calibration points."""
    global calibration_points_camera, calibration_points_robot, calibration_points_tag_ids

    calibration_points_camera = []
    calibration_points_robot = []
    calibration_points_tag_ids = []

    return CalibrationStatus(
        success=True,
        num_points=0,
        points=[],
        is_calibrated=calibration_transform is not None,
        message="Calibration points cleared"
    )


@app.post("/calibration/add_point", response_model=CalibrationStatus)
async def add_calibration_point(request: CalibrationAddPointRequest):
    """
    Add a calibration point pair.
    Records current robot EE position paired with the provided camera point.
    """
    global calibration_points_camera, calibration_points_robot, calibration_points_tag_ids

    try:
        # Get current robot EE position
        if robot is None:
            # Mock mode - use dummy position
            robot_point = [0.2, 0.0, 0.1]
            print(f"[Calibration] Mock mode - using dummy robot position")
        else:
            ee_pose = robot.arm.get_ee_pose()
            robot_point = [float(ee_pose[0, 3]), float(ee_pose[1, 3]), float(ee_pose[2, 3])]

        # Store the point pair with tag_id
        calibration_points_camera.append(request.camera_point)
        calibration_points_robot.append(robot_point)
        calibration_points_tag_ids.append(request.tag_id)

        print(f"[Calibration] Added point {len(calibration_points_camera)} (Tag {request.tag_id}): "
              f"camera={request.camera_point}, robot={robot_point}")

        # Build response with all tag_ids
        points = []
        for i in range(len(calibration_points_camera)):
            points.append(CalibrationPointPair(
                camera=calibration_points_camera[i],
                robot=calibration_points_robot[i],
                tag_id=calibration_points_tag_ids[i]
            ))

        return CalibrationStatus(
            success=True,
            num_points=len(calibration_points_camera),
            points=points,
            is_calibrated=calibration_transform is not None,
            message=f"Point {len(calibration_points_camera)} added"
        )

    except Exception as e:
        return CalibrationStatus(
            success=False,
            error=str(e)
        )


@app.post("/calibration/compute", response_model=CalibrationStatus)
async def compute_calibration():
    """
    Compute calibration transformation from recorded point pairs.
    Requires at least 3 point pairs.
    """
    global calibration_transform

    if len(calibration_points_camera) < 3:
        return CalibrationStatus(
            success=False,
            num_points=len(calibration_points_camera),
            error=f"Need at least 3 points, have {len(calibration_points_camera)}"
        )

    try:
        # Convert to numpy arrays
        pts_camera = np.array(calibration_points_camera)
        pts_robot = np.array(calibration_points_robot)

        # Compute transformation
        calibration_transform = compute_rigid_transform(pts_camera, pts_robot)

        # Compute reprojection error
        errors = []
        for i in range(len(calibration_points_camera)):
            pt_cam = np.array(calibration_points_camera[i] + [1.0])  # Homogeneous
            pt_robot_pred = calibration_transform @ pt_cam
            pt_robot_actual = np.array(calibration_points_robot[i])
            error = np.linalg.norm(pt_robot_pred[:3] - pt_robot_actual)
            errors.append(error)

        mean_error = np.mean(errors)
        max_error = np.max(errors)

        print(f"[Calibration] Computed transformation:")
        print(f"  Mean error: {mean_error*1000:.2f} mm")
        print(f"  Max error: {max_error*1000:.2f} mm")

        # Save calibration
        save_calibration(calibration_transform, calibration_points_camera, calibration_points_robot)

        # Build response
        points = [CalibrationPointPair(camera=calibration_points_camera[i], robot=calibration_points_robot[i])
                  for i in range(len(calibration_points_camera))]

        return CalibrationStatus(
            success=True,
            num_points=len(calibration_points_camera),
            points=points,
            is_calibrated=True,
            transform=calibration_transform.tolist(),
            message=f"Calibration computed. Mean error: {mean_error*1000:.2f}mm, Max error: {max_error*1000:.2f}mm"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return CalibrationStatus(
            success=False,
            num_points=len(calibration_points_camera),
            error=str(e)
        )


@app.post("/calibration/transform", response_model=TransformPointResponse)
async def transform_point(request: TransformPointRequest):
    """
    Transform a point from camera frame to robot frame using calibration.
    """
    global calibration_transform

    if calibration_transform is None:
        load_calibration()

    if calibration_transform is None:
        return TransformPointResponse(
            success=False,
            error="Not calibrated. Run calibration first."
        )

    try:
        # Convert to homogeneous coordinates
        pt_cam = np.array(request.point + [1.0])

        # Transform
        pt_robot = calibration_transform @ pt_cam

        return TransformPointResponse(
            success=True,
            camera_point=request.point,
            robot_point=pt_robot[:3].tolist()
        )

    except Exception as e:
        return TransformPointResponse(
            success=False,
            error=str(e)
        )


@app.post("/calibration/load", response_model=CalibrationStatus)
async def load_calibration_file():
    """Load calibration from file."""
    global calibration_transform, calibration_points_camera, calibration_points_robot

    try:
        if CALIBRATION_FILE.exists():
            with open(CALIBRATION_FILE, 'r') as f:
                data = yaml.safe_load(f)
                if data:
                    calibration_transform = np.array(data['transform'])
                    calibration_points_camera = data.get('points_camera', [])
                    calibration_points_robot = data.get('points_robot', [])

                    points = [CalibrationPointPair(camera=calibration_points_camera[i], robot=calibration_points_robot[i])
                              for i in range(len(calibration_points_camera))]

                    return CalibrationStatus(
                        success=True,
                        num_points=len(calibration_points_camera),
                        points=points,
                        is_calibrated=True,
                        transform=calibration_transform.tolist(),
                        message="Calibration loaded successfully"
                    )

        return CalibrationStatus(
            success=False,
            error="No calibration file found"
        )

    except Exception as e:
        return CalibrationStatus(
            success=False,
            error=str(e)
        )


# ==================== SQUARE POSITIONS (MANUAL TEACHING) ====================

# In-memory cache of square positions
_square_positions: dict = {}  # {"e2": {"x": ..., "y": ..., "z": ..., "pitch": ...}, ...}


class SquarePositionData(BaseModel):
    """Grasp position for a single chess square."""
    x: float = Field(..., description="Robot X in meters")
    y: float = Field(..., description="Robot Y in meters")
    z: float = Field(..., description="Robot Z in meters")
    pitch: Optional[float] = Field(None, description="Pitch in radians (None = auto)")


class SquarePositionSaveRequest(BaseModel):
    """Request to save the current EE position as a square's grasp position."""
    square: str = Field(..., description="Chess square, e.g. 'e2'")


class SquarePositionManualRequest(BaseModel):
    """Request to manually set a square's grasp position."""
    square: str = Field(..., description="Chess square, e.g. 'e2'")
    x: float
    y: float
    z: float
    pitch: Optional[float] = None


class SquarePositionsResponse(BaseModel):
    """Response for square positions queries."""
    success: bool
    count: int = 0
    squares: dict = {}  # {square: {x, y, z, pitch}}
    message: Optional[str] = None
    error: Optional[str] = None


class SquarePositionResponse(BaseModel):
    """Response for a single square position."""
    success: bool
    square: Optional[str] = None
    position: Optional[SquarePositionData] = None
    message: Optional[str] = None
    error: Optional[str] = None


def _validate_square(square: str) -> str:
    """Validate and normalize a chess square name. Returns lowercase."""
    s = square.strip().lower()
    if len(s) != 2 or s[0] not in "abcdefgh" or s[1] not in "12345678":
        raise ValueError(f"Invalid square: '{square}'")
    return s


def load_square_positions() -> dict:
    """Load square positions from YAML file into cache."""
    global _square_positions
    if SQUARE_POSITIONS_FILE.exists():
        try:
            with open(SQUARE_POSITIONS_FILE, 'r') as f:
                data = yaml.safe_load(f)
                _square_positions = data if data else {}
                print(f"[SquarePositions] Loaded {len(_square_positions)} squares")
        except Exception as e:
            print(f"[SquarePositions] Error loading: {e}")
            _square_positions = {}
    return _square_positions


def save_square_positions():
    """Persist current cache to YAML file."""
    try:
        with open(SQUARE_POSITIONS_FILE, 'w') as f:
            yaml.dump(_square_positions, f, default_flow_style=False)
        print(f"[SquarePositions] Saved {len(_square_positions)} squares to {SQUARE_POSITIONS_FILE}")
    except Exception as e:
        print(f"[SquarePositions] Error saving: {e}")
        raise


# Load on startup
load_square_positions()


@app.get("/square_positions", response_model=SquarePositionsResponse)
async def list_square_positions():
    """List all saved square grasp positions."""
    return SquarePositionsResponse(
        success=True,
        count=len(_square_positions),
        squares=_square_positions,
    )


@app.get("/square_positions/{square}", response_model=SquarePositionResponse)
async def get_square_position(square: str):
    """Get the saved grasp position for a single square."""
    try:
        sq = _validate_square(square)
    except ValueError as e:
        return SquarePositionResponse(success=False, error=str(e))

    if sq not in _square_positions:
        return SquarePositionResponse(success=False, square=sq, error=f"No position saved for {sq}")

    pos = _square_positions[sq]
    return SquarePositionResponse(
        success=True,
        square=sq,
        position=SquarePositionData(**pos),
    )


@app.post("/square_positions/record", response_model=SquarePositionResponse)
async def record_square_position(request: SquarePositionSaveRequest):
    """
    Save the robot's current EE position as the grasp position for a square.
    Jog the arm to the correct grasp point first, then call this endpoint.
    """
    try:
        sq = _validate_square(request.square)
    except ValueError as e:
        return SquarePositionResponse(success=False, error=str(e))

    if robot is None:
        return SquarePositionResponse(success=False, error="Robot not connected (or mock mode)")

    try:
        ee_pose = robot.arm.get_ee_pose()
        x, y, z = float(ee_pose[0, 3]), float(ee_pose[1, 3]), float(ee_pose[2, 3])

        R = ee_pose[:3, :3]
        _, pitch = rotation_matrix_to_euler(R)

        pos = {"x": round(x, 5), "y": round(y, 5), "z": round(z, 5), "pitch": round(float(pitch), 5)}
        _square_positions[sq] = pos
        save_square_positions()

        print(f"[SquarePositions] Recorded {sq}: {pos}")
        return SquarePositionResponse(
            success=True,
            square=sq,
            position=SquarePositionData(**pos),
            message=f"Recorded {sq} at ({x:.4f}, {y:.4f}, {z:.4f})",
        )
    except Exception as e:
        return SquarePositionResponse(success=False, error=str(e))


@app.post("/square_positions/set", response_model=SquarePositionResponse)
async def set_square_position(request: SquarePositionManualRequest):
    """Manually set the grasp position for a square (without moving the robot)."""
    try:
        sq = _validate_square(request.square)
    except ValueError as e:
        return SquarePositionResponse(success=False, error=str(e))

    pos = {
        "x": round(request.x, 5),
        "y": round(request.y, 5),
        "z": round(request.z, 5),
        "pitch": round(request.pitch, 5) if request.pitch is not None else None,
    }
    _square_positions[sq] = pos
    save_square_positions()

    return SquarePositionResponse(
        success=True,
        square=sq,
        position=SquarePositionData(**pos),
        message=f"Set {sq} to ({request.x:.4f}, {request.y:.4f}, {request.z:.4f})",
    )


@app.delete("/square_positions/{square}", response_model=SquarePositionResponse)
async def delete_square_position(square: str):
    """Delete a saved square position."""
    try:
        sq = _validate_square(square)
    except ValueError as e:
        return SquarePositionResponse(success=False, error=str(e))

    if sq not in _square_positions:
        return SquarePositionResponse(success=False, square=sq, error=f"No position saved for {sq}")

    del _square_positions[sq]
    save_square_positions()
    return SquarePositionResponse(success=True, square=sq, message=f"Deleted position for {sq}")


@app.post("/square_positions/clear", response_model=SquarePositionsResponse)
async def clear_square_positions():
    """Delete all saved square positions."""
    global _square_positions
    _square_positions = {}
    save_square_positions()
    return SquarePositionsResponse(success=True, count=0, squares={}, message="All square positions cleared")


@app.post("/square_positions/interpolate")
async def interpolate_square_positions():
    """
    Compute all 64 square positions from 4 taught corners via bilinear interpolation.

    Required corners: a1, a8, h1, h8 (must be in _square_positions).
    Overwrites all 64 square positions and saves.
    """
    global _square_positions

    required = ["a1", "a8", "h1", "h8"]
    missing = [c for c in required if c not in _square_positions]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing corners: {missing}. Teach a1, a8, h1, h8 first.",
        )

    c_a1 = _square_positions["a1"]
    c_a8 = _square_positions["a8"]
    c_h1 = _square_positions["h1"]
    c_h8 = _square_positions["h8"]

    files = "abcdefgh"
    ranks = "12345678"

    for fi, f in enumerate(files):
        u = fi / 7.0  # 0.0 at a-file, 1.0 at h-file
        for ri, r in enumerate(ranks):
            v = ri / 7.0  # 0.0 at rank 1, 1.0 at rank 8
            sq = f + r

            # Bilinear interpolation for each coordinate
            pos = {}
            for key in ["x", "y", "z"]:
                val_a1 = c_a1.get(key, 0)
                val_a8 = c_a8.get(key, 0)
                val_h1 = c_h1.get(key, 0)
                val_h8 = c_h8.get(key, 0)

                # Bilinear: lerp along file (u), then along rank (v)
                bottom = val_a1 * (1 - u) + val_h1 * u  # rank 1 edge
                top = val_a8 * (1 - u) + val_h8 * u      # rank 8 edge
                pos[key] = bottom * (1 - v) + top * v

            # Pitch: average of all 4 corners (usually constant)
            pitches = [c.get("pitch", 90) for c in [c_a1, c_a8, c_h1, c_h8]]
            pos["pitch"] = sum(pitches) / len(pitches)

            _square_positions[sq] = pos

    save_square_positions()
    print(f"[SquarePositions] Interpolated 64 squares from 4 corners")

    return {
        "success": True,
        "count": len(_square_positions),
        "message": "Interpolated 64 squares from corners a1, a8, h1, h8",
        "corners": {c: _square_positions[c] for c in required},
    }


# ==================== BOARD SURFACE Z ====================

def load_board_surface_z():
    """Load board_surface_z from YAML file."""
    global board_surface_z
    if BOARD_SURFACE_Z_FILE.exists():
        try:
            with open(BOARD_SURFACE_Z_FILE, 'r') as f:
                data = yaml.safe_load(f)
                board_surface_z = data.get("board_surface_z") if data else None
                print(f"[BoardSurfaceZ] Loaded: {board_surface_z}")
        except Exception as e:
            print(f"[BoardSurfaceZ] Error loading: {e}")
            board_surface_z = None
    return board_surface_z


def save_board_surface_z():
    """Persist board_surface_z to YAML file."""
    try:
        with open(BOARD_SURFACE_Z_FILE, 'w') as f:
            yaml.dump({"board_surface_z": board_surface_z}, f, default_flow_style=False)
        print(f"[BoardSurfaceZ] Saved: {board_surface_z}")
    except Exception as e:
        print(f"[BoardSurfaceZ] Error saving: {e}")
        raise


# Load on startup
load_board_surface_z()


@app.get("/board_surface_z")
async def get_board_surface_z():
    """Return the current board surface Z value."""
    return {"success": True, "board_surface_z": board_surface_z}


@app.post("/board_surface_z/record")
async def record_board_surface_z():
    """Record the current EE Z position as the board surface Z."""
    global board_surface_z
    if robot is None:
        return {"success": False, "error": "Robot not connected"}
    try:
        ee_pose = robot.arm.get_ee_pose()
        z = float(ee_pose[2, 3])
        board_surface_z = round(z, 5)
        save_board_surface_z()
        return {"success": True, "board_surface_z": board_surface_z,
                "message": f"Board surface Z recorded: {board_surface_z:.4f}m"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/board_surface_z/set")
async def set_board_surface_z(request: dict):
    """Manually set the board surface Z value."""
    global board_surface_z
    value = request.get("value")
    if value is None:
        return {"success": False, "error": "Missing 'value' field"}
    try:
        board_surface_z = round(float(value), 5)
        save_board_surface_z()
        return {"success": True, "board_surface_z": board_surface_z,
                "message": f"Board surface Z set to {board_surface_z:.4f}m"}
    except (TypeError, ValueError) as e:
        return {"success": False, "error": f"Invalid value: {e}"}




@app.post("/manual_pick", response_model=MoveResponse)
async def manual_pick(request: ManualPickRequest):
    """
    Pick up a piece:

    Approach:  L1 work -> above target -> grasp -> lift -> L1 work

    Position lookup: taught square positions > vision+calibration > hardcoded grid
    """
    global last_action, last_action_time, calibration_transform

    try:
        square = request.square.lower()
        if len(square) != 2 or square[0] not in "abcdefgh" or square[1] not in "12345678":
            return MoveResponse(success=False, error=f"Invalid square: {request.square}")

        # Piece type (for fallback height lookup)
        piece_type = (request.piece_type or "pawn").lower()
        for color in ("white_", "black_"):
            if piece_type.startswith(color):
                piece_type = piece_type[len(color):]
                break

        # Piece height from lookup table
        piece_h = PIECE_HEIGHTS.get(piece_type, PIECE_HEIGHTS["pawn"])
        print(f"[manual_pick] {square}: {piece_type} height={piece_h*1000:.0f}mm")

        # --- Resolve target XYZ (priority: taught > vision+calib > hardcoded) ---
        source = None
        if square in _square_positions:
            sp = _square_positions[square]
            xyz = np.array([sp['x'], sp['y'], sp['z']])
            source = "taught"
            print(f"[manual_pick] {square}: taught ({sp['x']:.4f}, {sp['y']:.4f}, {sp['z']:.4f})")
        else:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"http://localhost:8001/square_position/{square}", timeout=5.0)
                    pos_data = resp.json()
                if pos_data.get("success"):
                    if calibration_transform is None:
                        load_calibration()
                    if calibration_transform is not None:
                        camera_xyz = pos_data["camera_xyz"]
                        pt_robot = calibration_transform @ np.array(camera_xyz + [1.0])
                        xyz = pt_robot[:3]
                        source = "vision+calib"
                        print(f"[manual_pick] {square}: vision+calib ({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f})")
            except Exception as e:
                print(f"[manual_pick] vision lookup failed: {e}")

            if source is None:
                xyz = square_to_xyz(square)
                source = "hardcoded"
                print(f"[manual_pick] {square}: hardcoded fallback ({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f})")

        # --- Load L1 work position ---
        wp = get_work_position()
        if wp is None:
            return MoveResponse(success=False, error="No work position saved")

        # --- Pitch for taught position ---
        taught_pitch = None
        if square in _square_positions and _square_positions[square].get('pitch'):
            taught_pitch = _square_positions[square]['pitch']
        if taught_pitch is None:
            taught_pitch = find_reachable_pitch(xyz[0], xyz[1], xyz[2])

        if robot is None:
            approach_z_mock = (board_surface_z if board_surface_z is not None else xyz[2]) + 0.100
            suffix = " (skip L1 return)" if request.skip_return_to_work else ""
            print(f"[Mock] manual_pick {square} ({piece_type}, {source}): L1 -> above({approach_z_mock:.3f}) -> taught -> pick -> taught -> above -> L1{suffix}")
            last_action = f"manual_pick:{square}"
            last_action_time = time.time()
            return MoveResponse(success=True, message=f"[Mock] Picked {piece_type} from {square} ({source}){suffix}")

        # ===== CALCULATE HEIGHTS =====
        board_z = board_surface_z if board_surface_z is not None else xyz[2]
        board_z_source = "taught" if board_surface_z is not None else "square_z"
        approach_z = board_z + 0.150
        row = int(square[1])
        if piece_type == "knight":
            grasp_ratio = 0.35 if row <= 3 else 0.25
        else:
            grasp_ratio = 0.75
        grasp_z = board_z + piece_h * grasp_ratio
        lift_z = board_z + 0.150

        # Add Z offset for close rows
        row_offset = 0.0
        if row == 1:
            row_offset = 0.030
        elif row <= 3:
            row_offset = 0.020
        grasp_z += row_offset

        # Debug summary
        print(f"")
        print(f"  ===== PICK DEBUG: {square} ({piece_type}) =====")
        print(f"  taught XYZ:      ({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}) [{source}]")
        print(f"  board_surface_z: {board_surface_z if board_surface_z is not None else 'NOT SET'} ({board_z_source})")
        print(f"  board_z (used):  {board_z:.4f}")
        print(f"  piece_height:    {piece_h*1000:.0f}mm")
        print(f"  grasp_ratio:     {grasp_ratio*100:.0f}%")
        print(f"  row_offset:      {row_offset*1000:.0f}mm (row {row})")
        print(f"  approach_z:      {approach_z:.4f}")
        print(f"  grasp_z:         {grasp_z:.4f}")
        print(f"  lift_z:          {lift_z:.4f}")
        print(f"  pitch:           {taught_pitch}")
        print(f"  ==========================================")
        print(f"")

        # ===== HELPER: move with pitch tolerance (90° first, then search to 45°) =====
        def move_with_pitch_tolerance(x, y, z, preferred_pitch, moving_time=1.5):
            """Try preferred pitch, then search in 1° steps down to 45°."""
            import math
            PITCH_TOL = math.radians(45)
            pitches_to_try = [preferred_pitch]
            step = math.radians(5)
            for offset in range(1, int(PITCH_TOL / step) + 1):
                pitches_to_try.append(preferred_pitch + offset * step)
                pitches_to_try.append(preferred_pitch - offset * step)

            for pitch in pitches_to_try:
                test_result = robot.arm.set_ee_pose_components(
                    x=x, y=y, z=z, pitch=pitch,
                    blocking=False, execute=False,
                )
                if isinstance(test_result, tuple):
                    test_ok = test_result[1]
                else:
                    test_ok = bool(test_result)
                if test_ok:
                    if abs(pitch - preferred_pitch) > 0.01:
                        print(f"  [pitch search] Using {math.degrees(pitch):.1f}° (offset {math.degrees(pitch - preferred_pitch):.1f}°)")
                    robot.arm.set_ee_pose_components(
                        x=x, y=y, z=z, pitch=pitch,
                        moving_time=moving_time, blocking=True, execute=True,
                    )
                    return True
            print(f"  [pitch search] No valid pitch found for ({x:.4f}, {y:.4f}, {z:.4f})")
            robot.arm.set_ee_pose_components(
                x=x, y=y, z=z, pitch=preferred_pitch,
                moving_time=moving_time,
            )
            return False

        # ===== APPROACH =====

        # Step 1: L1 work + gripper open 65%
        print(f"[manual_pick] Step 1/7: L1 work + gripper open 65%")
        robot.arm.set_trajectory_time(moving_time=2.0 * SPEED_FACTOR, accel_time=0.5 * SPEED_FACTOR)
        robot.arm.set_joint_positions(wp)
        time.sleep(1.0 * SPEED_FACTOR)
        robot.gripper.open()
        time.sleep(0.5 * SPEED_FACTOR)
        robot.dxl.robot_set_operating_modes('single', 'gripper', 'position')
        time.sleep(0.3 * SPEED_FACTOR)
        robot.dxl.robot_write_joint_command('gripper', GRIPPER_65_OPEN_M)
        time.sleep(0.8 * SPEED_FACTOR)
        robot.dxl.robot_set_operating_modes('single', 'gripper', 'pwm')
        time.sleep(0.8 * SPEED_FACTOR)

        # Step 2: Above target square at board_surface_z + 150mm
        print(f"[manual_pick] Step 2/6: Above {square} (z={approach_z:.4f})")
        move_with_pitch_tolerance(xyz[0], xyz[1], approach_z, taught_pitch, moving_time=1.5 * SPEED_FACTOR)
        time.sleep(1.0 * SPEED_FACTOR)

        # ===== PICK UP =====

        # Step 3: Down to grasp Z (3/4 piece height)
        print(f"[manual_pick] Step 3/6: Down to grasp z={grasp_z:.4f}")
        move_with_pitch_tolerance(xyz[0], xyz[1], grasp_z, taught_pitch, moving_time=1.0 * SPEED_FACTOR)
        time.sleep(0.5 * SPEED_FACTOR)

        # Step 4: Gripper close
        print(f"[manual_pick] Step 4/6: Gripper close")
        robot.gripper.close()
        time.sleep(0.8 * SPEED_FACTOR)

        # Step 5: Lift to board_surface_z + 150mm
        print(f"[manual_pick] Step 5/6: Lift to z={lift_z:.4f}")
        move_with_pitch_tolerance(xyz[0], xyz[1], lift_z, taught_pitch, moving_time=1.0 * SPEED_FACTOR)
        time.sleep(0.5 * SPEED_FACTOR)

        # ===== RETURN =====

        if request.skip_return_to_work:
            print(f"[manual_pick] Step 6/6: SKIPPED (skip_return_to_work=True, piece in gripper)")
        else:
            # Step 6: L1 work
            print(f"[manual_pick] Step 6/6: L1 work")
            robot.arm.set_trajectory_time(moving_time=2.0 * SPEED_FACTOR, accel_time=0.5 * SPEED_FACTOR)
            robot.arm.set_joint_positions(wp)
            time.sleep(1.5 * SPEED_FACTOR)

        last_action = f"manual_pick:{square}"
        last_action_time = time.time()

        return MoveResponse(
            success=True,
            message=f"Picked {piece_type} from {square}",
            from_square=square,
            debug={
                "taught_xyz": [round(xyz[0], 4), round(xyz[1], 4), round(xyz[2], 4)],
                "source": source,
                "board_surface_z": board_surface_z,
                "board_z_used": round(board_z, 4),
                "piece_type": piece_type,
                "piece_height_mm": round(piece_h * 1000),
                "approach_z": round(approach_z, 4),
                "grasp_z": round(grasp_z, 4),
                "lift_z": round(lift_z, 4),
                "pitch": taught_pitch,
            },
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return MoveResponse(success=False, error=str(e))


@app.post("/manual_place", response_model=MoveResponse)
async def manual_place(request: ManualPlaceRequest):
    """
    Place a held piece onto a square.

    5-step sequence:
      1. Above target square (board_z + 150mm)
      2. Down to grasp Z (board_z + piece_height * 3/4)
      3. Gripper open
      4. Lift (board_z + 150mm)
      5. L1 work
    """
    global last_action, last_action_time

    try:
        square = request.square.lower()
        if len(square) != 2 or square[0] not in "abcdefgh" or square[1] not in "12345678":
            return MoveResponse(success=False, error=f"Invalid square: {request.square}")

        # Piece type
        piece_type = (request.piece_type or "pawn").lower()
        for color in ("white_", "black_"):
            if piece_type.startswith(color):
                piece_type = piece_type[len(color):]
                break

        piece_h = PIECE_HEIGHTS.get(piece_type, PIECE_HEIGHTS["pawn"])
        print(f"[manual_place] {square}: {piece_type} height={piece_h*1000:.0f}mm")

        # --- Resolve target XYZ ---
        source = None
        if square in _square_positions:
            sp = _square_positions[square]
            xyz = np.array([sp['x'], sp['y'], sp['z']])
            source = "taught"
        else:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"http://localhost:8001/square_position/{square}", timeout=5.0)
                    pos_data = resp.json()
                if pos_data.get("success"):
                    if calibration_transform is None:
                        load_calibration()
                    if calibration_transform is not None:
                        camera_xyz = pos_data["camera_xyz"]
                        pt_robot = calibration_transform @ np.array(camera_xyz + [1.0])
                        xyz = pt_robot[:3]
                        source = "vision+calib"
            except Exception as e:
                print(f"[manual_place] vision lookup failed: {e}")

            if source is None:
                xyz = square_to_xyz(square)
                source = "hardcoded"

        # --- Load L1 work position ---
        wp = get_work_position()
        if wp is None:
            return MoveResponse(success=False, error="No work position saved")

        # --- Pitch ---
        taught_pitch = None
        if square in _square_positions and _square_positions[square].get('pitch'):
            taught_pitch = _square_positions[square]['pitch']
        if taught_pitch is None:
            taught_pitch = find_reachable_pitch(xyz[0], xyz[1], xyz[2])

        if robot is None:
            print(f"[Mock] manual_place {square} ({piece_type}, {source})")
            last_action = f"manual_place:{square}"
            last_action_time = time.time()
            return MoveResponse(success=True, message=f"[Mock] Placed {piece_type} on {square} ({source})")

        # ===== CALCULATE HEIGHTS =====
        board_z = board_surface_z if board_surface_z is not None else xyz[2]
        approach_z = board_z + 0.150
        row = int(square[1])
        if piece_type == "knight":
            grasp_ratio = 0.35 if row <= 3 else 0.25
        else:
            grasp_ratio = 0.75
        grasp_z = board_z + piece_h * grasp_ratio
        lift_z = board_z + 0.150

        # Add Z offset for close rows
        row_offset = 0.0
        if row == 1:
            row_offset = 0.030
        elif row <= 3:
            row_offset = 0.020
        grasp_z += row_offset

        # Debug summary
        print(f"")
        print(f"  ===== PLACE DEBUG: {square} ({piece_type}) =====")
        print(f"  taught XYZ:      ({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}) [{source}]")
        print(f"  board_z (used):  {board_z:.4f}")
        print(f"  piece_height:    {piece_h*1000:.0f}mm")
        print(f"  grasp_ratio:     {grasp_ratio*100:.0f}%")
        print(f"  row_offset:      {row_offset*1000:.0f}mm (row {row})")
        print(f"  approach_z:      {approach_z:.4f}")
        print(f"  grasp_z:         {grasp_z:.4f}")
        print(f"  lift_z:          {lift_z:.4f}")
        print(f"  pitch:           {taught_pitch}")
        print(f"  ==========================================")
        print(f"")

        # ===== HELPER: move with pitch tolerance =====
        def move_with_pitch_tolerance(x, y, z, preferred_pitch, moving_time=1.5):
            import math
            PITCH_TOL = math.radians(45)
            pitches_to_try = [preferred_pitch]
            step = math.radians(5)
            for offset in range(1, int(PITCH_TOL / step) + 1):
                pitches_to_try.append(preferred_pitch + offset * step)
                pitches_to_try.append(preferred_pitch - offset * step)
            for pitch in pitches_to_try:
                test_result = robot.arm.set_ee_pose_components(
                    x=x, y=y, z=z, pitch=pitch,
                    blocking=False, execute=False,
                )
                if isinstance(test_result, tuple):
                    test_ok = test_result[1]
                else:
                    test_ok = bool(test_result)
                if test_ok:
                    if abs(pitch - preferred_pitch) > 0.01:
                        print(f"  [pitch search] Using {math.degrees(pitch):.1f}° (offset {math.degrees(pitch - preferred_pitch):.1f}°)")
                    robot.arm.set_ee_pose_components(
                        x=x, y=y, z=z, pitch=pitch,
                        moving_time=moving_time, blocking=True, execute=True,
                    )
                    return True
            print(f"  [pitch search] No valid pitch found for ({x:.4f}, {y:.4f}, {z:.4f})")
            robot.arm.set_ee_pose_components(
                x=x, y=y, z=z, pitch=preferred_pitch,
                moving_time=moving_time,
            )
            return False

        # Step 1: Above target square (board_z + 150mm)
        print(f"[manual_place] Step 1/5: Above {square} (z={approach_z:.4f})")
        move_with_pitch_tolerance(xyz[0], xyz[1], approach_z, taught_pitch, moving_time=1.5 * SPEED_FACTOR)
        time.sleep(1.0 * SPEED_FACTOR)

        # Step 2: Down to grasp Z (3/4 piece height)
        print(f"[manual_place] Step 2/5: Down to grasp z={grasp_z:.4f}")
        move_with_pitch_tolerance(xyz[0], xyz[1], grasp_z, taught_pitch, moving_time=0.8 * SPEED_FACTOR)
        time.sleep(0.5 * SPEED_FACTOR)

        # Step 3: Gripper open 65%
        print(f"[manual_place] Step 3/5: Gripper open 65%")
        robot.dxl.robot_set_operating_modes('single', 'gripper', 'position')
        time.sleep(0.3 * SPEED_FACTOR)
        robot.dxl.robot_write_joint_command('gripper', GRIPPER_65_OPEN_M)
        time.sleep(0.8 * SPEED_FACTOR)
        robot.dxl.robot_set_operating_modes('single', 'gripper', 'pwm')
        time.sleep(0.8 * SPEED_FACTOR)

        # Step 4: Lift (board_z + 150mm)
        print(f"[manual_place] Step 4/5: Lift to z={lift_z:.4f}")
        move_with_pitch_tolerance(xyz[0], xyz[1], lift_z, taught_pitch, moving_time=1.0 * SPEED_FACTOR)
        time.sleep(0.5 * SPEED_FACTOR)

        # Step 5: L1 work
        print(f"[manual_place] Step 5/5: L1 work")
        robot.arm.set_trajectory_time(moving_time=2.0 * SPEED_FACTOR, accel_time=0.5 * SPEED_FACTOR)
        robot.arm.set_joint_positions(wp)
        time.sleep(1.5 * SPEED_FACTOR)

        last_action = f"manual_place:{square}"
        last_action_time = time.time()

        return MoveResponse(
            success=True,
            message=f"Placed {piece_type} on {square}",
            to_square=square,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return MoveResponse(success=False, error=str(e))


@app.post("/capture", response_model=MoveResponse)
async def capture_piece(request: ManualPickRequest):
    """
    Capture an enemy piece: pick it up from a square and place it at the capture zone.

    Uses the same smart routing as manual_pick for the pick phase,
    then moves to the taught capture_zone_position to drop the piece.

    Sequence:
      1. L1 work + gripper open
      2. Above square -> down to grasp -> gripper close -> lift
      3. Move to capture zone (above -> down -> gripper open -> lift)
      4. L1 work
    """
    global last_action, last_action_time, calibration_transform

    try:
        # --- Validate capture zone ---
        cz_joints = get_capture_zone_joints()
        if cz_joints is None:
            return MoveResponse(success=False, error="No capture_zone waypoint. Save a waypoint with tag 'capture_zone'.")

        # --- Validate square ---
        square = request.square.lower()
        if len(square) != 2 or square[0] not in "abcdefgh" or square[1] not in "12345678":
            return MoveResponse(success=False, error=f"Invalid square: {request.square}")

        # --- Piece type ---
        piece_type = (request.piece_type or "pawn").lower()
        for color in ("white_", "black_"):
            if piece_type.startswith(color):
                piece_type = piece_type[len(color):]
                break
        piece_h = PIECE_HEIGHTS.get(piece_type, PIECE_HEIGHTS["pawn"])
        print(f"[capture] {square}: {piece_type} height={piece_h*1000:.0f}mm -> capture zone")

        # --- Resolve square XYZ (taught > vision+calib > hardcoded) ---
        source = None
        if square in _square_positions:
            sp = _square_positions[square]
            xyz = np.array([sp['x'], sp['y'], sp['z']])
            source = "taught"
        else:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"http://localhost:8001/square_position/{square}", timeout=5.0)
                    pos_data = resp.json()
                if pos_data.get("success"):
                    if calibration_transform is None:
                        load_calibration()
                    if calibration_transform is not None:
                        camera_xyz = pos_data["camera_xyz"]
                        pt_robot = calibration_transform @ np.array(camera_xyz + [1.0])
                        xyz = pt_robot[:3]
                        source = "vision+calib"
            except Exception as e:
                print(f"[capture] vision lookup failed: {e}")
            if source is None:
                xyz = square_to_xyz(square)
                source = "hardcoded"

        print(f"[capture] {square}: {source} ({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f})")
        print(f"[capture] capture zone: joint-based waypoint")

        # --- Work position ---
        wp = get_work_position()
        if wp is None:
            return MoveResponse(success=False, error="No work position saved")

        # --- Pitch for square ---
        taught_pitch = None
        if square in _square_positions and _square_positions[square].get('pitch'):
            taught_pitch = _square_positions[square]['pitch']
        if taught_pitch is None:
            taught_pitch = find_reachable_pitch(xyz[0], xyz[1], xyz[2])

        # --- Mock mode ---
        if robot is None:
            print(f"[Mock] capture {square} ({piece_type}, {source}) -> capture zone")
            last_action = f"capture:{square}"
            last_action_time = time.time()
            return MoveResponse(success=True, message=f"[Mock] Captured {piece_type} from {square} to capture zone")

        # ===== CALCULATE PICK HEIGHTS =====
        board_z = board_surface_z if board_surface_z is not None else xyz[2]
        approach_z = board_z + 0.150
        row = int(square[1])
        if piece_type == "knight":
            grasp_ratio = 0.35 if row <= 3 else 0.25
        else:
            grasp_ratio = 0.75
        grasp_z = board_z + piece_h * grasp_ratio
        lift_z = board_z + 0.150

        # Row offset
        row_offset = 0.0
        if row == 1:
            row_offset = 0.030
        elif row <= 3:
            row_offset = 0.020
        grasp_z += row_offset

        print(f"  ===== CAPTURE DEBUG: {square} -> capture zone (joints) =====")
        print(f"  pick: grasp_z={grasp_z:.4f}, approach_z={approach_z:.4f}")
        print(f"  =====================================================")

        # ===== HELPER: move with pitch tolerance =====
        def move_with_pitch_tolerance(x, y, z, preferred_pitch, moving_time=1.5):
            import math
            PITCH_TOL = math.radians(45)
            pitches_to_try = [preferred_pitch]
            step = math.radians(5)
            for offset in range(1, int(PITCH_TOL / step) + 1):
                pitches_to_try.append(preferred_pitch + offset * step)
                pitches_to_try.append(preferred_pitch - offset * step)
            for pitch in pitches_to_try:
                test_result = robot.arm.set_ee_pose_components(
                    x=x, y=y, z=z, pitch=pitch,
                    blocking=False, execute=False,
                )
                if isinstance(test_result, tuple):
                    test_ok = test_result[1]
                else:
                    test_ok = bool(test_result)
                if test_ok:
                    if abs(pitch - preferred_pitch) > 0.01:
                        print(f"  [pitch search] Using {math.degrees(pitch):.1f}°")
                    robot.arm.set_ee_pose_components(
                        x=x, y=y, z=z, pitch=pitch,
                        moving_time=moving_time, blocking=True, execute=True,
                    )
                    return True
            robot.arm.set_ee_pose_components(
                x=x, y=y, z=z, pitch=preferred_pitch,
                moving_time=moving_time,
            )
            return False

        # ===== PHASE 1: PICK UP from square =====

        # Step 1: L1 work + gripper open
        print(f"[capture] Step 1: L1 work + gripper open")
        robot.arm.set_trajectory_time(moving_time=2.0 * SPEED_FACTOR, accel_time=0.5 * SPEED_FACTOR)
        robot.arm.set_joint_positions(wp)
        time.sleep(1.0 * SPEED_FACTOR)
        robot.gripper.open()
        time.sleep(0.5 * SPEED_FACTOR)
        robot.dxl.robot_set_operating_modes('single', 'gripper', 'position')
        time.sleep(0.3 * SPEED_FACTOR)
        robot.dxl.robot_write_joint_command('gripper', GRIPPER_65_OPEN_M)
        time.sleep(0.8 * SPEED_FACTOR)
        robot.dxl.robot_set_operating_modes('single', 'gripper', 'pwm')
        time.sleep(0.5 * SPEED_FACTOR)

        # Step 2: Above square
        print(f"[capture] Step 2: Above {square} (z={approach_z:.4f})")
        move_with_pitch_tolerance(xyz[0], xyz[1], approach_z, taught_pitch, moving_time=1.5 * SPEED_FACTOR)
        time.sleep(1.0 * SPEED_FACTOR)

        # Step 3: Down to grasp
        print(f"[capture] Step 3: Down to grasp z={grasp_z:.4f}")
        move_with_pitch_tolerance(xyz[0], xyz[1], grasp_z, taught_pitch, moving_time=1.0 * SPEED_FACTOR)
        time.sleep(0.5 * SPEED_FACTOR)

        # Step 4: Gripper close
        print(f"[capture] Step 4: Gripper close")
        robot.gripper.close()
        time.sleep(0.8 * SPEED_FACTOR)

        # Step 5: Lift
        print(f"[capture] Step 5: Lift to z={lift_z:.4f}")
        move_with_pitch_tolerance(xyz[0], xyz[1], lift_z, taught_pitch, moving_time=1.0 * SPEED_FACTOR)
        time.sleep(0.5 * SPEED_FACTOR)

        # ===== PHASE 2: DROP at capture zone (joint-based) =====
        print(f"[capture] Step 6: Move to capture zone + drop")
        _drop_at_capture_zone()

        # Return to work
        print(f"[capture] Step 7: Return to work")
        robot.arm.set_trajectory_time(moving_time=2.0 * SPEED_FACTOR, accel_time=0.5 * SPEED_FACTOR)
        robot.arm.set_joint_positions(wp)
        time.sleep(1.5 * SPEED_FACTOR)

        last_action = f"capture:{square}"
        last_action_time = time.time()

        return MoveResponse(
            success=True,
            message=f"Captured {piece_type} from {square} to capture zone",
            from_square=square,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return MoveResponse(success=False, error=str(e))


# ==================== GESTURE RECORDING & PLAYBACK ====================

# Pre-defined emotional gestures (joint positions for RX-200 5-DOF arm)
# Each frame is [waist, shoulder, elbow, wrist_angle, wrist_rotate]
# fps=10 for smooth playback, each frame = 100ms
_BUILTIN_GESTURES = {
    "nod": {
        "description": "Nodding — agreement/acknowledgment",
        "fps": 10,
        "frames": [
            [0.0, -0.5, 0.5, -0.8, 0.0],   # neutral-ish
            [0.0, -0.3, 0.3, -0.6, 0.0],    # head down
            [0.0, -0.5, 0.5, -0.8, 0.0],    # back up
            [0.0, -0.3, 0.3, -0.6, 0.0],    # head down
            [0.0, -0.5, 0.5, -0.8, 0.0],    # back up
        ],
        "builtin": True,
    },
    "shake": {
        "description": "Head shake — disagreement",
        "fps": 10,
        "frames": [
            [0.0, -0.5, 0.5, -0.8, 0.0],    # neutral
            [0.3, -0.5, 0.5, -0.8, 0.0],     # turn right
            [-0.3, -0.5, 0.5, -0.8, 0.0],    # turn left
            [0.3, -0.5, 0.5, -0.8, 0.0],     # turn right
            [-0.3, -0.5, 0.5, -0.8, 0.0],    # turn left
            [0.0, -0.5, 0.5, -0.8, 0.0],     # back center
        ],
        "builtin": True,
    },
    "think": {
        "description": "Thinking — pause and tilt",
        "fps": 6,
        "frames": [
            [0.0, -0.5, 0.5, -0.8, 0.0],     # neutral
            [0.2, -0.6, 0.6, -0.9, 0.3],      # tilt and rotate
            [0.2, -0.6, 0.6, -0.9, 0.3],      # hold
            [0.2, -0.6, 0.6, -0.9, 0.3],      # hold
            [0.0, -0.5, 0.5, -0.8, 0.0],      # back
        ],
        "builtin": True,
    },
    "celebrate": {
        "description": "Celebration — arm raise and wave",
        "fps": 8,
        "frames": [
            [0.0, -0.5, 0.5, -0.8, 0.0],      # neutral
            [0.0, -1.2, 1.0, -0.3, 0.0],       # arm up
            [0.2, -1.2, 1.0, -0.3, 0.5],       # tilt + rotate
            [-0.2, -1.2, 1.0, -0.3, -0.5],     # other side
            [0.2, -1.2, 1.0, -0.3, 0.5],       # tilt + rotate
            [-0.2, -1.2, 1.0, -0.3, -0.5],     # other side
            [0.0, -0.5, 0.5, -0.8, 0.0],       # back to neutral
        ],
        "builtin": True,
    },
    "wave": {
        "description": "Friendly wave",
        "fps": 8,
        "frames": [
            [0.0, -0.5, 0.5, -0.8, 0.0],     # neutral
            [0.3, -1.0, 0.8, -0.4, 0.0],      # arm up, turned
            [0.3, -1.0, 0.8, -0.4, 0.5],      # wrist rotate
            [0.3, -1.0, 0.8, -0.4, -0.5],     # wrist other way
            [0.3, -1.0, 0.8, -0.4, 0.5],      # wrist rotate
            [0.3, -1.0, 0.8, -0.4, -0.5],     # wrist other way
            [0.0, -0.5, 0.5, -0.8, 0.0],      # back to neutral
        ],
        "builtin": True,
    },
    "sad": {
        "description": "Sad — droop down slowly",
        "fps": 5,
        "frames": [
            [0.0, -0.5, 0.5, -0.8, 0.0],      # neutral
            [0.0, -0.3, 0.3, -0.5, 0.0],       # drooping
            [0.0, -0.2, 0.2, -0.3, 0.0],       # lower
            [0.0, -0.2, 0.2, -0.3, 0.0],       # hold
            [0.0, -0.2, 0.2, -0.3, 0.0],       # hold
            [0.0, -0.5, 0.5, -0.8, 0.0],       # recover
        ],
        "builtin": True,
    },
}


def load_gestures():
    """Load user-recorded gestures from YAML file."""
    global _gestures
    # Start with builtins
    _gestures = dict(_BUILTIN_GESTURES)
    # Override/add from file
    if GESTURES_FILE.exists():
        try:
            with open(GESTURES_FILE, 'r') as f:
                data = yaml.safe_load(f)
                if data:
                    for name, gesture in data.items():
                        gesture["builtin"] = False
                        _gestures[name] = gesture
                    print(f"[Gestures] Loaded {len(data)} custom gestures from file")
        except Exception as e:
            print(f"[Gestures] Error loading: {e}")
    print(f"[Gestures] {len(_gestures)} total gestures available "
          f"({len(_BUILTIN_GESTURES)} builtin)")


def save_gestures():
    """Save only user-recorded gestures (not builtins) to YAML file."""
    try:
        custom = {k: v for k, v in _gestures.items() if not v.get("builtin", False)}
        with open(GESTURES_FILE, 'w') as f:
            yaml.dump(custom, f, default_flow_style=False)
        print(f"[Gestures] Saved {len(custom)} custom gestures")
    except Exception as e:
        print(f"[Gestures] Error saving: {e}")
        raise


# Load on startup
load_gestures()


def _gesture_record_worker(fps):
    """Background thread that samples joint positions at given fps."""
    global _gesture_record_frames, _gesture_recording
    interval = 1.0 / fps
    print(f"[Gestures] Recording at {fps} Hz...")
    while _gesture_recording:
        t0 = time.time()
        try:
            if robot is not None:
                robot.arm.capture_joint_positions()
                joints = robot.arm.get_joint_commands()
                frame = joints.tolist() if hasattr(joints, 'tolist') else list(joints)
                _gesture_record_frames.append(frame)
        except Exception as e:
            print(f"[Gestures] Record sample error: {e}")
        elapsed = time.time() - t0
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
    print(f"[Gestures] Recording stopped, {len(_gesture_record_frames)} frames captured")


def _simplify_gesture_frames(frames, min_delta=0.02):
    """Downsample frames by removing near-duplicates.

    Keeps a frame only when at least one joint moved more than *min_delta*
    radians since the last kept frame.  First and last frames are always kept.
    """
    if len(frames) <= 2:
        return list(frames)
    simplified = [frames[0]]
    for f in frames[1:-1]:
        prev = simplified[-1]
        max_diff = max(abs(a - b) for a, b in zip(f, prev))
        if max_diff >= min_delta:
            simplified.append(f)
    simplified.append(frames[-1])
    return simplified


def _play_gesture_blocking(gesture, speed=1.0):
    """Play a gesture (blocking). Call from thread or sync context.

    Uses robot_write_trajectory() to publish the entire gesture as a single
    JointTrajectory ROS message — the motor controller smoothly interpolates
    between all waypoints rather than snapping frame-by-frame.
    """
    global _gesture_playing
    raw_frames = gesture.get("frames", [])
    fps = gesture.get("fps", 10)
    if not raw_frames:
        return

    # Simplify: drop near-duplicate frames for smoother playback
    frames = _simplify_gesture_frames(raw_frames)
    original_duration = len(raw_frames) / fps
    # Distribute the original duration evenly across simplified frames
    interval = (original_duration / max(len(frames), 1)) / speed
    total_duration = interval * len(frames)

    _gesture_playing = True

    try:
        if robot is not None:
            # Build trajectory: list of {timestamp: [joint_positions]}
            raw_traj = []
            for i, frame in enumerate(frames):
                t = i * interval
                raw_traj.append({t: frame})

            # First, move to the starting frame smoothly
            robot.arm.set_joint_positions(
                frames[0], moving_time=0.5, accel_time=0.2, blocking=True
            )

            # Publish entire trajectory at once — ROS controller interpolates
            robot.arm.core.robot_write_trajectory(
                cmd_type="group", name="arm",
                type="position", raw_traj=raw_traj
            )

            print(f"[Gestures] Trajectory published: {len(frames)} waypoints, "
                  f"{total_duration:.1f}s (simplified from {len(raw_frames)} frames)")

            # Wait for trajectory to complete (can be interrupted)
            t0 = time.time()
            while time.time() - t0 < total_duration:
                if not _gesture_playing:
                    break
                time.sleep(0.1)
        else:
            # Mock mode: just wait the duration
            time.sleep(total_duration)
    finally:
        _gesture_playing = False


class GestureRecordStartRequest(BaseModel):
    """Request to start gesture recording."""
    name: str = Field(..., description="Name for the gesture")
    fps: int = Field(10, description="Recording sample rate (Hz). 10-20 recommended")
    description: str = Field("", description="Optional description of the gesture")


class GesturePlayRequest(BaseModel):
    """Request to play a gesture."""
    name: str = Field(..., description="Gesture name to play")
    speed: float = Field(1.0, description="Playback speed multiplier (1.0=normal, 2.0=double)")
    max_duration: Optional[float] = Field(None, description="Max duration in seconds — speed auto-adjusted to fit")
    return_to: Optional[str] = Field(None, description="Position to return to after: 'work', 'vision', or None")


@app.get("/arm/gestures")
async def list_gestures():
    """List all available gestures (builtin + custom)."""
    result = {}
    for name, g in _gestures.items():
        result[name] = {
            "description": g.get("description", ""),
            "builtin": g.get("builtin", False),
            "num_frames": len(g.get("frames", [])),
            "fps": g.get("fps", 10),
            "duration_s": round(len(g.get("frames", [])) / max(g.get("fps", 10), 1), 2),
        }
    return {"success": True, "gestures": result, "recording": _gesture_recording}


@app.post("/arm/gesture/record/start", response_model=MoveResponse)
async def gesture_record_start(request: GestureRecordStartRequest):
    """
    Start recording a gesture. Disables torque so you can move the arm by hand.
    Move the arm through the desired gesture, then call /arm/gesture/record/stop.
    """
    global _gesture_recording, _gesture_record_thread, _gesture_record_frames

    if _gesture_recording:
        return MoveResponse(success=False, error="Already recording a gesture")

    if _gesture_playing:
        return MoveResponse(success=False, error="A gesture is currently playing")

    fps = max(1, min(request.fps, 50))  # clamp 1-50

    if robot is None:
        # Mock mode
        _gesture_recording = True
        _gesture_record_frames = []
        return MoveResponse(success=True, message=f"[Mock] Recording gesture '{request.name}' at {fps}Hz")

    # Disable torque so user can move arm by hand
    try:
        import rospy
        from interbotix_xs_msgs.srv import TorqueEnable
        service_name = '/rx200/torque_enable'
        rospy.wait_for_service(service_name, timeout=5.0)
        torque_srv = rospy.ServiceProxy(service_name, TorqueEnable)
        torque_srv(cmd_type='group', name='arm', enable=False)
        print("[Gestures] Arm torque disabled for recording")
    except Exception as e:
        return MoveResponse(success=False, error=f"Failed to disable torque: {e}")

    # Start recording thread
    _gesture_record_frames = []
    _gesture_recording = True
    _gesture_record_thread = threading.Thread(
        target=_gesture_record_worker, args=(fps,), daemon=True
    )
    _gesture_record_thread.start()

    # Store metadata for stop handler
    _gesture_record_thread._gesture_name = request.name
    _gesture_record_thread._gesture_fps = fps
    _gesture_record_thread._gesture_desc = request.description

    return MoveResponse(
        success=True,
        message=f"Recording gesture '{request.name}' at {fps}Hz. "
                "Move the arm by hand, then call /arm/gesture/record/stop"
    )


@app.post("/arm/gesture/record/stop", response_model=MoveResponse)
async def gesture_record_stop():
    """
    Stop recording and save the gesture. Re-enables torque.
    """
    global _gesture_recording, _gesture_record_thread

    if not _gesture_recording:
        return MoveResponse(success=False, error="Not currently recording")

    # Stop recording
    _gesture_recording = False
    if _gesture_record_thread is not None:
        _gesture_record_thread.join(timeout=2.0)

    # Get metadata
    name = getattr(_gesture_record_thread, '_gesture_name', 'unnamed')
    fps = getattr(_gesture_record_thread, '_gesture_fps', 10)
    desc = getattr(_gesture_record_thread, '_gesture_desc', '')

    frames = list(_gesture_record_frames)
    _gesture_record_thread = None

    # Re-enable torque
    if robot is not None:
        try:
            import rospy
            from interbotix_xs_msgs.srv import TorqueEnable
            service_name = '/rx200/torque_enable'
            rospy.wait_for_service(service_name, timeout=5.0)
            torque_srv = rospy.ServiceProxy(service_name, TorqueEnable)
            torque_srv(cmd_type='group', name='arm', enable=True)
            print("[Gestures] Arm torque re-enabled")
        except Exception as e:
            print(f"[Gestures] Warning: failed to re-enable torque: {e}")

    if len(frames) < 2:
        return MoveResponse(
            success=False,
            error=f"Too few frames recorded ({len(frames)}). Move the arm during recording."
        )

    # Save gesture
    gesture = {
        "description": desc,
        "fps": fps,
        "frames": frames,
        "builtin": False,
    }
    _gestures[name] = gesture
    save_gestures()

    duration = len(frames) / fps
    return MoveResponse(
        success=True,
        message=f"Gesture '{name}' saved: {len(frames)} frames, {duration:.1f}s at {fps}Hz"
    )


@app.post("/arm/gesture/play", response_model=MoveResponse)
async def gesture_play(request: GesturePlayRequest):
    """
    Play back a recorded or builtin gesture.
    """
    global _gesture_playing

    if request.name not in _gestures:
        available = list(_gestures.keys())
        return MoveResponse(
            success=False,
            error=f"Gesture '{request.name}' not found. Available: {available}"
        )

    if _gesture_recording:
        return MoveResponse(success=False, error="Cannot play while recording")

    if _gesture_playing:
        return MoveResponse(success=False, error="A gesture is already playing")

    gesture = _gestures[request.name]

    # Auto-adjust speed if max_duration is set
    speed = request.speed
    if request.max_duration and request.max_duration > 0:
        natural_duration = len(gesture.get("frames", [])) / max(gesture.get("fps", 10), 1) / speed
        if natural_duration > request.max_duration:
            speed = natural_duration / request.max_duration * speed
            print(f"[Gestures] Capping '{request.name}' to {request.max_duration}s (speed {speed:.1f}x)")

    if robot is None:
        duration = len(gesture.get("frames", [])) / max(gesture.get("fps", 10), 1) / speed
        return MoveResponse(
            success=True,
            message=f"[Mock] Playing gesture '{request.name}' ({duration:.1f}s)"
        )

    # Play in current thread (endpoint is async, but robot calls are blocking anyway)
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, _play_gesture_blocking, gesture, speed
    )

    # Return to position if requested
    if request.return_to == "work":
        _wp = get_work_position()
        if _wp is not None:
            robot.arm.set_joint_positions(_wp, moving_time=1.5 * SPEED_FACTOR)
            time.sleep(1.5 * SPEED_FACTOR)
    elif request.return_to == "vision":
        _vp = get_vision_position()
        if _vp is not None:
            robot.arm.set_joint_positions(_vp, moving_time=1.5 * SPEED_FACTOR)
            time.sleep(1.5 * SPEED_FACTOR)

    duration = len(gesture.get("frames", [])) / max(gesture.get("fps", 10), 1) / request.speed
    return MoveResponse(
        success=True,
        message=f"Played gesture '{request.name}' ({duration:.1f}s)"
    )


@app.post("/arm/gesture/stop", response_model=MoveResponse)
async def gesture_stop():
    """Stop a currently playing gesture."""
    global _gesture_playing
    if not _gesture_playing:
        return MoveResponse(success=True, message="No gesture playing")
    _gesture_playing = False
    return MoveResponse(success=True, message="Gesture playback stopped")


@app.delete("/arm/gesture/{name}", response_model=MoveResponse)
async def gesture_delete(name: str):
    """Delete a custom gesture. Cannot delete builtin gestures."""
    if name not in _gestures:
        return MoveResponse(success=False, error=f"Gesture '{name}' not found")
    if _gestures[name].get("builtin", False):
        return MoveResponse(success=False, error=f"Cannot delete builtin gesture '{name}'")
    del _gestures[name]
    save_gestures()
    return MoveResponse(success=True, message=f"Gesture '{name}' deleted")


# ==================== GAME STATE (OCCUPANCY-BASED) ====================


def _move_to_vision_pos():
    """Move robot to vision position so it doesn't block the camera."""
    vp = get_vision_position()
    if robot is None or vp is None:
        return
    try:
        print("[Robot] Moving to vision position for camera clearance...")
        robot.arm.set_trajectory_time(moving_time=2.0 * SPEED_FACTOR, accel_time=0.5 * SPEED_FACTOR)
        robot.arm.set_joint_positions(vp)
        time.sleep(2.0 * SPEED_FACTOR)
    except Exception as e:
        print(f"[Robot] Vision pos move failed: {e}")


def _move_to_work_pos():
    """Move robot back to work position after vision capture."""
    wp = get_work_position()
    if robot is None or wp is None:
        return
    try:
        print("[Robot] Moving back to work position...")
        robot.arm.set_trajectory_time(moving_time=2.0 * SPEED_FACTOR, accel_time=0.5 * SPEED_FACTOR)
        robot.arm.set_joint_positions(wp)
        time.sleep(2.0 * SPEED_FACTOR)
    except Exception as e:
        print(f"[Robot] Work pos move failed: {e}")


@app.post("/game/init")
async def game_init():
    """
    Initialize a new game using occupancy-based detection.

    1. Moves robot to vision position (clear camera view)
    2. Calls vision /capture/occupancy_init to get occupied squares + white side hint
    3. Creates BoardTracker with standard starting FEN
    4. Validates occupied count (~32 expected)
    5. Moves robot back to work position
    """
    global _board_tracker, _game_move_history, _game_white_side

    try:
        # Move robot out of camera view
        _move_to_vision_pos()

        # Call vision service for occupancy + orientation
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{VISION_SERVICE_URL}/capture/occupancy_init", timeout=15.0)
            vision_data = resp.json()

        if not vision_data.get("success"):
            return {
                "success": False,
                "error": f"Vision failed: {vision_data.get('error', 'Unknown')}",
            }

        occupied_squares = vision_data["occupied_squares"]
        white_side = vision_data.get("white_side", "bottom")
        total_count = vision_data.get("total_count", 0)

        # Create tracker with standard starting position
        _board_tracker = BoardTracker()
        _game_move_history = []
        _game_white_side = white_side

        # Get piece map and FEN from tracker
        piece_map = _board_tracker.get_piece_map()
        fen = _board_tracker.fen

        print(f"[game/init] Initialized game: {total_count} pieces detected, white_side={white_side}")

        # Move robot back to work position
        _move_to_work_pos()

        return {
            "success": True,
            "fen": fen,
            "piece_map": piece_map,
            "white_side": white_side,
            "occupied_count": total_count,
            "expected_count": 32,
            "turn": _board_tracker.turn,
        }

    except Exception as e:
        traceback.print_exc()
        _move_to_work_pos()
        return {"success": False, "error": str(e)}


@app.post("/game/detect_move")
async def game_detect_move():
    """
    Detect the human's move using occupancy diff.

    1. Calls vision /capture/occupancy for current occupied squares
    2. Gets previous occupancy from BoardTracker
    3. Calls detect_human_move() to match legal moves
    4. Pushes the move if found
    """
    global _board_tracker, _game_move_history

    if _board_tracker is None:
        return {"success": False, "error": "No active game. Call /game/init first."}

    try:
        # Move robot out of camera view
        _move_to_vision_pos()

        # Call vision for current occupancy
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{VISION_SERVICE_URL}/capture/occupancy", timeout=30.0)
            vision_data = resp.json()

        # Move robot back to work position
        _move_to_work_pos()

        if not vision_data.get("success"):
            return {
                "success": False,
                "error": f"Vision failed: {vision_data.get('error', 'Unknown')}",
            }

        curr_occupied = set(vision_data["occupied_squares"])
        prev_occupied = _board_tracker.get_occupancy()

        # Detect the move
        result = _board_tracker.detect_human_move(prev_occupied, curr_occupied)

        if not result.success:
            return {
                "success": False,
                "error": result.error,
                "candidates": result.candidates,
                "prev_count": len(prev_occupied),
                "curr_count": len(curr_occupied),
                "emptied": sorted(prev_occupied - curr_occupied),
                "filled": sorted(curr_occupied - prev_occupied),
            }

        # Apply the move
        import chess
        move = chess.Move.from_uci(result.uci_move)
        _board_tracker.push_move(move)

        # Record in history
        move_record = {
            "uci_move": result.uci_move,
            "piece": result.piece,
            "is_capture": result.is_capture,
            "captured_piece": result.captured_piece,
            "is_castling": result.is_castling,
            "is_en_passant": result.is_en_passant,
            "promotion": result.promotion,
            "move_number": len(_game_move_history) + 1,
        }
        _game_move_history.append(move_record)

        piece_map = _board_tracker.get_piece_map()
        fen = _board_tracker.fen

        print(f"[game/detect_move] Detected: {result.uci_move} ({result.piece}), capture={result.is_capture}")

        return {
            "success": True,
            "uci_move": result.uci_move,
            "piece": result.piece,
            "is_capture": result.is_capture,
            "captured_piece": result.captured_piece,
            "is_castling": result.is_castling,
            "is_en_passant": result.is_en_passant,
            "promotion": result.promotion,
            "fen": fen,
            "piece_map": piece_map,
            "move_number": len(_game_move_history),
            "turn": _board_tracker.turn,
        }

    except Exception as e:
        traceback.print_exc()
        _move_to_work_pos()
        return {"success": False, "error": str(e)}


@app.get("/game/state")
async def game_state():
    """
    Get current game state from tracker.
    """
    if _board_tracker is None:
        return {
            "active": False,
            "fen": None,
            "piece_map": {},
            "turn": None,
            "move_count": 0,
            "move_history": [],
        }

    return {
        "active": True,
        "fen": _board_tracker.fen,
        "piece_map": _board_tracker.get_piece_map(),
        "turn": _board_tracker.turn,
        "move_count": len(_game_move_history),
        "move_history": _game_move_history,
        "white_side": _game_white_side,
    }


@app.post("/game/reset")
async def game_reset():
    """
    Reset the game tracker and history.
    """
    global _board_tracker, _game_move_history, _game_white_side

    _board_tracker = None
    _game_move_history = []
    _game_white_side = None

    print("[game/reset] Game reset")
    return {"success": True, "message": "Game reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
