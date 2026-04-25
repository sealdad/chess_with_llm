# chess_vision/depth_utils.py
"""
Depth-camera utilities for chess piece measurement, collision avoidance,
and piece height validation.

All functions accept generic numpy depth arrays and CameraIntrinsics —
no hardware SDK imports required.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .camera import CameraIntrinsics, deproject_pixel_to_point


# Expected piece heights in meters (standard tournament pieces, ~3.75" king)
EXPECTED_PIECE_HEIGHTS = {
    "pawn": 0.025,
    "knight": 0.033,
    "bishop": 0.035,
    "rook": 0.040,
    "queen": 0.045,
    "king": 0.050,
}

# Tolerance for height validation (meters)
HEIGHT_TOLERANCE = 0.010  # +/-10mm


def _get_depth_m(depth_map: np.ndarray, depth_scale: float,
                 px: int, py: int) -> float:
    """Read depth in meters at pixel (px, py), returning 0.0 on OOB."""
    h, w = depth_map.shape[:2]
    if 0 <= px < w and 0 <= py < h:
        return float(depth_map[py, px]) * depth_scale
    return 0.0


def measure_piece_depth(
    bbox: List[int],
    depth_map: np.ndarray,
    depth_scale: float,
    intrinsics: CameraIntrinsics,
) -> Optional[Tuple[float, float, float]]:
    """
    Measure the 3D position of a piece's top surface from its bounding box.

    Samples depth in the upper-center region of the bbox (the top of the piece
    is closest to the camera, so it has the minimum depth value).

    Args:
        bbox: [x1, y1, x2, y2] bounding box in pixel coordinates
        depth_map: (H, W) raw depth array (uint16 or float)
        depth_scale: meters per raw depth unit
        intrinsics: camera intrinsic parameters

    Returns:
        (X, Y, Z) in camera frame (meters), or None if measurement failed
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    # Sample upper-center region (top 30% of bbox, middle 50% width)
    sample_x1 = int(x1 + w * 0.25)
    sample_x2 = int(x1 + w * 0.75)
    sample_y1 = int(y1)
    sample_y2 = int(y1 + h * 0.3)

    # Collect valid depth samples
    depths = []
    pixels = []
    for py in range(sample_y1, sample_y2, 2):
        for px in range(sample_x1, sample_x2, 2):
            d = _get_depth_m(depth_map, depth_scale, px, py)
            if 0.05 < d < 2.0:  # Valid range: 5cm to 2m
                depths.append(d)
                pixels.append((px, py))

    if len(depths) < 3:
        return None

    # Use minimum depth values (top of piece = closest to camera)
    depths = np.array(depths)
    # Take the 10th percentile to be robust against noise
    threshold = np.percentile(depths, 10)
    mask = depths <= threshold
    valid_depths = depths[mask]
    valid_pixels = [pixels[i] for i in range(len(pixels)) if mask[i]]

    if len(valid_depths) == 0:
        return None

    # Average the closest points
    avg_depth = float(np.mean(valid_depths))
    avg_px = float(np.mean([p[0] for p in valid_pixels]))
    avg_py = float(np.mean([p[1] for p in valid_pixels]))

    # Deproject to 3D
    point_3d = deproject_pixel_to_point(intrinsics, [avg_px, avg_py], avg_depth)
    return tuple(point_3d)


def measure_board_surface_z(
    depth_map: np.ndarray,
    depth_scale: float,
    intrinsics: CameraIntrinsics,
    grid_9x9: Optional[np.ndarray] = None,
    board_state: Optional[Dict] = None,
) -> Optional[float]:
    """
    Measure the Z depth of the board surface in camera frame.

    Samples depth at empty squares (grid cells without pieces) to get
    the board surface depth. Falls back to grid corner midpoints if
    no board_state is provided.

    Args:
        depth_map: (H, W) raw depth array
        depth_scale: meters per raw depth unit
        intrinsics: camera intrinsic parameters
        grid_9x9: shape (9, 9, 2) grid intersection points (optional)
        board_state: dict of {square: piece_info} for occupied squares

    Returns:
        Median Z depth of board surface in meters, or None
    """
    sample_depths = []

    if grid_9x9 is not None:
        # Find empty squares by checking which grid cells have no pieces
        occupied_cells = set()
        if board_state:
            files = "abcdefgh"
            for square in board_state.keys():
                col = files.index(square[0])
                row = 7 - (int(square[1]) - 1)  # rank 1 -> row 7
                occupied_cells.add((row, col))

        # Sample center of empty squares
        for row in range(8):
            for col in range(8):
                if (row, col) in occupied_cells:
                    continue
                # Center of grid cell
                tl = grid_9x9[row, col]
                br = grid_9x9[row + 1, col + 1]
                cx = int((tl[0] + br[0]) / 2)
                cy = int((tl[1] + br[1]) / 2)

                d = _get_depth_m(depth_map, depth_scale, cx, cy)
                if 0.05 < d < 2.0:
                    sample_depths.append(d)
    else:
        # Fallback: sample a grid of points across the frame center
        width = intrinsics.width
        height = intrinsics.height
        for fy in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for fx in [0.3, 0.4, 0.5, 0.6, 0.7]:
                px = int(width * fx)
                py = int(height * fy)
                d = _get_depth_m(depth_map, depth_scale, px, py)
                if 0.05 < d < 2.0:
                    sample_depths.append(d)

    if len(sample_depths) < 3:
        return None

    return float(np.median(sample_depths))


def compute_piece_heights(
    board_state: Dict,
    depth_map: np.ndarray,
    depth_scale: float,
    intrinsics: CameraIntrinsics,
    grid_9x9: Optional[np.ndarray] = None,
    calibration_transform: Optional[np.ndarray] = None,
) -> Dict[str, Dict]:
    """
    Compute the height of each detected piece above the board surface.

    Args:
        board_state: dict of {square: {"piece": ..., "bbox": [x1,y1,x2,y2], ...}}
        depth_map: (H, W) raw depth array
        depth_scale: meters per raw depth unit
        intrinsics: camera intrinsic parameters
        grid_9x9: optional grid for board surface measurement
        calibration_transform: optional 4x4 camera-to-robot transform

    Returns:
        Dict of {square: {"height_m": float, "piece_z_cam": float,
                          "board_z_cam": float, "grasp_z_robot": float|None}}
    """
    # Measure board surface Z
    board_z = measure_board_surface_z(
        depth_map, depth_scale, intrinsics, grid_9x9, board_state
    )
    if board_z is None:
        return {}

    results = {}
    for square, info in board_state.items():
        bbox = info.get("bbox")
        if bbox is None:
            continue

        piece_top = measure_piece_depth(bbox, depth_map, depth_scale, intrinsics)
        if piece_top is None:
            continue

        piece_z = piece_top[2]  # Z in camera frame
        # Height = board surface Z minus piece top Z
        # (camera Z increases away from camera, so board is farther = larger Z)
        height_m = board_z - piece_z

        # Clamp to reasonable range
        if height_m < 0.005 or height_m > 0.15:
            continue

        entry = {
            "height_m": round(height_m, 4),
            "piece_z_cam": round(piece_z, 4),
            "board_z_cam": round(board_z, 4),
            "piece_top_cam": [round(c, 4) for c in piece_top],
        }

        # Compute robot-frame grasp Z if calibration is available
        if calibration_transform is not None:
            pt_cam_h = np.array([piece_top[0], piece_top[1], piece_top[2], 1.0])
            pt_robot = calibration_transform @ pt_cam_h
            entry["grasp_z_robot"] = round(float(pt_robot[2]), 4)
            entry["grasp_xyz_robot"] = [round(float(pt_robot[i]), 4) for i in range(3)]

        results[square] = entry

    return results


def check_path_clearance(
    start_xyz: List[float],
    end_xyz: List[float],
    depth_map: np.ndarray,
    depth_scale: float,
    intrinsics: CameraIntrinsics,
    calibration_transform: np.ndarray,
    approach_height: float = 0.15,
    clearance_margin: float = 0.02,
    num_waypoints: int = 10,
) -> Dict:
    """
    Check if a straight-line path at approach height is clear of obstacles.

    Generates waypoints along the path in robot frame, transforms them to
    camera frame, projects to pixel coordinates, and compares actual depth
    vs expected depth.

    Args:
        start_xyz: [x, y, z] start position in robot frame (meters)
        end_xyz: [x, y, z] end position in robot frame (meters)
        depth_map: (H, W) raw depth array
        depth_scale: meters per raw depth unit
        intrinsics: camera intrinsic parameters
        calibration_transform: 4x4 camera-to-robot transform
        approach_height: Z height for the travel path in robot frame
        clearance_margin: minimum clearance required (meters)
        num_waypoints: number of points to sample along path

    Returns:
        {"clear": bool, "obstacles": [...], "min_clearance": float,
         "num_checked": int}
    """
    # Compute inverse calibration (robot -> camera)
    try:
        T_robot_to_camera = np.linalg.inv(calibration_transform)
    except np.linalg.LinAlgError:
        return {"clear": False, "obstacles": [], "min_clearance": 0.0,
                "num_checked": 0, "error": "Invalid calibration transform"}

    start = np.array(start_xyz, dtype=float)
    end = np.array(end_xyz, dtype=float)

    obstacles = []
    min_clearance = float("inf")
    num_checked = 0

    for i in range(num_waypoints):
        t = i / max(num_waypoints - 1, 1)
        # Interpolate XY, use approach_height for Z
        wp_robot = start + t * (end - start)
        wp_robot[2] = approach_height

        # Transform robot -> camera
        wp_robot_h = np.array([wp_robot[0], wp_robot[1], wp_robot[2], 1.0])
        wp_cam = T_robot_to_camera @ wp_robot_h

        # Project camera 3D -> pixel using intrinsics
        if wp_cam[2] <= 0:
            continue  # Behind camera

        px = int(intrinsics.fx * wp_cam[0] / wp_cam[2] + intrinsics.ppx)
        py = int(intrinsics.fy * wp_cam[1] / wp_cam[2] + intrinsics.ppy)

        # Check bounds
        if px < 0 or px >= intrinsics.width or py < 0 or py >= intrinsics.height:
            continue

        # Get actual depth at this pixel
        actual_depth = _get_depth_m(depth_map, depth_scale, px, py)
        if actual_depth <= 0:
            continue

        num_checked += 1

        # Expected depth is the camera-frame Z of our waypoint
        expected_depth = float(wp_cam[2])

        # If actual depth is less than expected, something is in the way
        clearance = actual_depth - expected_depth + clearance_margin
        min_clearance = min(min_clearance, clearance)

        if actual_depth < expected_depth - clearance_margin:
            obstacles.append({
                "waypoint_idx": i,
                "robot_xyz": wp_robot.tolist(),
                "pixel": [px, py],
                "expected_depth": round(expected_depth, 4),
                "actual_depth": round(actual_depth, 4),
            })

    if min_clearance == float("inf"):
        min_clearance = 0.0

    return {
        "clear": len(obstacles) == 0,
        "obstacles": obstacles,
        "min_clearance": round(min_clearance, 4),
        "num_checked": num_checked,
    }


def validate_piece_by_height(
    measured_height: float,
    classified_type: str,
    tolerance: float = HEIGHT_TOLERANCE,
) -> Dict:
    """
    Validate a piece classification against its measured height.

    Args:
        measured_height: measured height in meters
        classified_type: classified piece name (e.g., "white_pawn", "black_rook")
        tolerance: acceptable deviation in meters

    Returns:
        {"valid": bool, "expected_height": float, "measured_height": float,
         "deviation": float, "suggested_type": str or None}
    """
    # Extract base type (remove color prefix)
    base_type = classified_type.split("_", 1)[-1] if "_" in classified_type else classified_type
    color = classified_type.split("_")[0] if "_" in classified_type else ""

    expected = EXPECTED_PIECE_HEIGHTS.get(base_type)
    if expected is None:
        return {
            "valid": True,
            "expected_height": None,
            "measured_height": round(measured_height, 4),
            "deviation": None,
            "suggested_type": None,
        }

    deviation = measured_height - expected
    valid = abs(deviation) <= tolerance

    # Find best matching piece type by height if invalid
    suggested_type = None
    if not valid:
        best_match = None
        best_diff = float("inf")
        for ptype, pheight in EXPECTED_PIECE_HEIGHTS.items():
            diff = abs(measured_height - pheight)
            if diff < best_diff:
                best_diff = diff
                best_match = ptype
        if best_match and best_match != base_type:
            suggested_type = f"{color}_{best_match}" if color else best_match

    return {
        "valid": valid,
        "expected_height": round(expected, 4),
        "measured_height": round(measured_height, 4),
        "deviation": round(deviation, 4),
        "suggested_type": suggested_type,
    }
