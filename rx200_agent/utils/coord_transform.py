# rx200_agent/utils/coord_transform.py
"""
Coordinate transformation utilities for camera-to-robot transforms.
"""

import numpy as np
import yaml
from typing import Tuple, Optional
from pathlib import Path

from ..config import (
    BOARD_SQUARE_SIZE,
    BOARD_ORIGIN,
    HAND_EYE_CALIBRATION,
    PIECE_HEIGHTS,
)


def load_calibration(calibration_file: Optional[str] = None) -> dict:
    """
    Load hand-eye calibration from YAML file.

    Returns dict with:
        - R_cam2base: 3x3 rotation matrix
        - t_cam2base: 3x1 translation vector
    """
    if calibration_file is None:
        calibration_file = str(HAND_EYE_CALIBRATION)

    path = Path(calibration_file)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")

    with open(path, "r") as f:
        calib = yaml.safe_load(f)

    return {
        "R_cam2base": np.array(calib.get("R_cam2gripper", calib.get("rotation", np.eye(3)))),
        "t_cam2base": np.array(calib.get("t_cam2gripper", calib.get("translation", [0, 0, 0]))),
    }


def square_to_robot_xyz(
    square: str,
    board_origin: Optional[list] = None,
    square_size: Optional[float] = None,
) -> np.ndarray:
    """
    Convert a chess square name to robot XYZ coordinates.

    Args:
        square: Chess square name (e.g., "e4")
        board_origin: XYZ of a1 corner in robot frame (default from config)
        square_size: Size of each square in meters (default from config)

    Returns:
        np.ndarray of [x, y, z] in robot base frame
    """
    if board_origin is None:
        board_origin = BOARD_ORIGIN
    if square_size is None:
        square_size = BOARD_SQUARE_SIZE

    # Parse square name
    file_char = square[0].lower()
    rank_char = square[1]

    file_idx = ord(file_char) - ord('a')  # 0-7 for a-h
    rank_idx = int(rank_char) - 1          # 0-7 for 1-8

    # Calculate center of square
    # a1 is at board_origin, h8 is at origin + 7*square_size in both x and y
    x = board_origin[0] + (file_idx + 0.5) * square_size
    y = board_origin[1] + (rank_idx + 0.5) * square_size
    z = board_origin[2]

    return np.array([x, y, z])


def get_piece_grasp_height(piece_type: str) -> float:
    """
    Get the appropriate grasp height for a piece type.

    Args:
        piece_type: e.g., "white_pawn", "black_queen"

    Returns:
        Height in meters for grasping
    """
    # Extract piece name without color
    if "_" in piece_type:
        piece_name = piece_type.split("_")[1]
    else:
        piece_name = piece_type.lower()

    return PIECE_HEIGHTS.get(piece_name, 0.05)  # Default 5cm


def camera_to_robot(
    camera_xyz: np.ndarray,
    R_cam2base: np.ndarray,
    t_cam2base: np.ndarray,
) -> np.ndarray:
    """
    Transform a point from camera frame to robot base frame.

    Args:
        camera_xyz: Point in camera coordinates [x, y, z]
        R_cam2base: 3x3 rotation matrix
        t_cam2base: 3x1 translation vector

    Returns:
        Point in robot base coordinates
    """
    camera_xyz = np.array(camera_xyz).reshape(3, 1)
    robot_xyz = R_cam2base @ camera_xyz + t_cam2base.reshape(3, 1)
    return robot_xyz.flatten()


def robot_to_camera(
    robot_xyz: np.ndarray,
    R_cam2base: np.ndarray,
    t_cam2base: np.ndarray,
) -> np.ndarray:
    """
    Transform a point from robot base frame to camera frame.

    Args:
        robot_xyz: Point in robot coordinates [x, y, z]
        R_cam2base: 3x3 rotation matrix
        t_cam2base: 3x1 translation vector

    Returns:
        Point in camera coordinates
    """
    robot_xyz = np.array(robot_xyz).reshape(3, 1)
    t = t_cam2base.reshape(3, 1)
    camera_xyz = R_cam2base.T @ (robot_xyz - t)
    return camera_xyz.flatten()


def get_approach_pose(
    target_xyz: np.ndarray,
    approach_height: float = 0.15,
) -> np.ndarray:
    """
    Get an approach position above the target.

    Args:
        target_xyz: Target position
        approach_height: Height above target for approach

    Returns:
        Approach position [x, y, z]
    """
    approach = target_xyz.copy()
    approach[2] = target_xyz[2] + approach_height
    return approach


def interpolate_path(
    start: np.ndarray,
    end: np.ndarray,
    num_points: int = 10,
) -> list:
    """
    Generate a linear interpolated path between two points.

    Returns list of waypoints.
    """
    waypoints = []
    for i in range(num_points):
        t = i / (num_points - 1)
        point = start + t * (end - start)
        waypoints.append(point)
    return waypoints
