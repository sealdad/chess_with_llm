"""
apriltag_detector.py - AprilTag detection and pose estimation

Provides AprilTag detection for hand-eye calibration.
Supports multiple AprilTag libraries (pupil-apriltags, dt-apriltags, apriltag).
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass


@dataclass
class AprilTagDetection:
    """Detected AprilTag with pose information."""
    tag_id: int
    corners: np.ndarray          # 4x2 corner points in image
    center: np.ndarray           # 2D center point

    # Pose (if camera matrix provided)
    rvec: Optional[np.ndarray] = None   # Rotation vector (Rodrigues)
    tvec: Optional[np.ndarray] = None   # Translation vector
    R: Optional[np.ndarray] = None      # 3x3 Rotation matrix

    # Quality metrics
    decision_margin: float = 0.0
    hamming: int = 0


class AprilTagDetector:
    """
    AprilTag detector with pose estimation.

    Supports multiple backends:
    - pupil-apriltags (recommended, pip install pupil-apriltags)
    - dt-apriltags (pip install dt-apriltags)
    - apriltag (pip install apriltag)
    """

    def __init__(
        self,
        tag_family: str = "tag36h11",
        tag_size: float = 0.05,  # Tag size in meters
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
    ):
        """
        Initialize AprilTag detector.

        Args:
            tag_family: AprilTag family (tag36h11, tag25h9, etc.)
            tag_size: Physical size of tag in meters (edge length)
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
        """
        self.tag_family = tag_family
        self.tag_size = tag_size
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)

        self.detector = None
        self.backend = None

        self._init_detector()

    def _init_detector(self):
        """Initialize the AprilTag detector with available backend."""

        # Try pupil-apriltags first (best performance)
        try:
            from pupil_apriltags import Detector
            self.detector = Detector(
                families=self.tag_family,
                nthreads=4,
                quad_decimate=1.0,  # 1.0 = full resolution, increase for speed
                quad_sigma=0.8,     # Gaussian blur to reduce noise
                refine_edges=True,
                decode_sharpening=0.25,
            )
            self.backend = "pupil-apriltags"
            print(f"[AprilTag] Using pupil-apriltags backend, family={self.tag_family}")
            return
        except ImportError:
            pass

        # Try dt-apriltags
        try:
            from dt_apriltags import Detector
            self.detector = Detector(
                families=self.tag_family,
                nthreads=4,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=True,
                decode_sharpening=0.25,
            )
            self.backend = "dt-apriltags"
            print(f"[AprilTag] Using dt-apriltags backend")
            return
        except ImportError:
            pass

        # Try apriltag (slower, pure Python)
        try:
            import apriltag
            self.detector = apriltag.Detector(apriltag.DetectorOptions(
                families=self.tag_family,
            ))
            self.backend = "apriltag"
            print(f"[AprilTag] Using apriltag backend")
            return
        except ImportError:
            pass

        raise ImportError(
            "No AprilTag library found. Install one of:\n"
            "  pip install pupil-apriltags  (recommended)\n"
            "  pip install dt-apriltags\n"
            "  pip install apriltag"
        )

    def detect(
        self,
        image: np.ndarray,
        estimate_pose: bool = True,
    ) -> List[AprilTagDetection]:
        """
        Detect AprilTags in image.

        Args:
            image: BGR or grayscale image
            estimate_pose: Whether to estimate 3D pose

        Returns:
            List of AprilTagDetection objects
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            # Try BGR first (OpenCV default), but also works for RGB
            # since AprilTag detection mainly needs contrast, not exact luminance
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print(f"[AprilTag] Gray image shape: {gray.shape}, min/max: {gray.min()}/{gray.max()}")
        else:
            gray = image
            print(f"[AprilTag] Already grayscale: {gray.shape}")

        # Save grayscale for debugging
        cv2.imwrite('/app/captures/debug_gray.png', gray)

        detections = []

        if self.backend in ["pupil-apriltags", "dt-apriltags"]:
            # These backends support pose estimation directly
            if estimate_pose and self.camera_matrix is not None:
                results = self.detector.detect(
                    gray,
                    estimate_tag_pose=True,
                    camera_params=(
                        self.camera_matrix[0, 0],  # fx
                        self.camera_matrix[1, 1],  # fy
                        self.camera_matrix[0, 2],  # cx
                        self.camera_matrix[1, 2],  # cy
                    ),
                    tag_size=self.tag_size,
                )
            else:
                results = self.detector.detect(gray)

            for r in results:
                det = AprilTagDetection(
                    tag_id=r.tag_id,
                    corners=r.corners,
                    center=r.center,
                    decision_margin=r.decision_margin,
                    hamming=r.hamming,
                )

                if hasattr(r, 'pose_R') and r.pose_R is not None:
                    det.R = r.pose_R
                    det.tvec = r.pose_t.flatten()
                    det.rvec, _ = cv2.Rodrigues(r.pose_R)
                    det.rvec = det.rvec.flatten()

                detections.append(det)

        elif self.backend == "apriltag":
            results = self.detector.detect(gray)

            for r in results:
                det = AprilTagDetection(
                    tag_id=r.tag_id,
                    corners=r.corners,
                    center=r.center,
                    decision_margin=r.decision_margin,
                    hamming=r.hamming,
                )

                # Estimate pose using solvePnP
                if estimate_pose and self.camera_matrix is not None:
                    rvec, tvec = self._estimate_pose_pnp(r.corners)
                    if rvec is not None:
                        det.rvec = rvec
                        det.tvec = tvec
                        det.R, _ = cv2.Rodrigues(rvec)

                detections.append(det)

        return detections

    def _estimate_pose_pnp(
        self,
        corners: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate tag pose using solvePnP.

        Args:
            corners: 4x2 array of corner points

        Returns:
            (rvec, tvec) or (None, None) if failed
        """
        if self.camera_matrix is None:
            return None, None

        # Define 3D points of tag corners (centered at origin)
        half_size = self.tag_size / 2
        object_points = np.array([
            [-half_size, -half_size, 0],
            [ half_size, -half_size, 0],
            [ half_size,  half_size, 0],
            [-half_size,  half_size, 0],
        ], dtype=np.float32)

        # Ensure corners are in right format
        image_points = corners.astype(np.float32).reshape(4, 2)

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        if success:
            return rvec.flatten(), tvec.flatten()
        return None, None

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[AprilTagDetection],
        draw_axes: bool = True,
        axis_length: float = 0.03,
    ) -> np.ndarray:
        """
        Draw detected tags on image.

        Args:
            image: BGR image to draw on
            detections: List of detections
            draw_axes: Whether to draw 3D axes
            axis_length: Length of axes in meters

        Returns:
            Image with drawings
        """
        vis = image.copy()

        for det in detections:
            # Draw corners and edges
            corners = det.corners.astype(int)
            for i in range(4):
                pt1 = tuple(corners[i])
                pt2 = tuple(corners[(i + 1) % 4])
                cv2.line(vis, pt1, pt2, (0, 255, 0), 2)

            # Draw center
            center = tuple(det.center.astype(int))
            cv2.circle(vis, center, 5, (0, 0, 255), -1)

            # Draw ID
            cv2.putText(
                vis, f"ID:{det.tag_id}",
                (center[0] - 20, center[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
            )

            # Draw 3D axes if pose available
            if draw_axes and det.rvec is not None and self.camera_matrix is not None:
                cv2.drawFrameAxes(
                    vis,
                    self.camera_matrix,
                    self.dist_coeffs,
                    det.rvec,
                    det.tvec,
                    axis_length,
                )

        return vis

    def set_camera_params(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
    ):
        """Update camera parameters."""
        self.camera_matrix = camera_matrix
        if dist_coeffs is not None:
            self.dist_coeffs = dist_coeffs


def load_camera_intrinsics(yaml_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load camera intrinsics from YAML file.

    Returns:
        (camera_matrix, dist_coeffs)
    """
    import yaml

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    camera_matrix = np.array(data['camera_matrix'], dtype=np.float64)
    dist_coeffs = np.array(data.get('dist_coeffs', [0, 0, 0, 0, 0]), dtype=np.float64)

    return camera_matrix, dist_coeffs
