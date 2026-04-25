"""
handeye_calibration.py - Hand-Eye Calibration with AprilTag

Complete workflow for eye-to-hand calibration:
1. Collect robot poses and corresponding AprilTag poses
2. Run OpenCV calibrateHandEye
3. Validate and save results
"""

import os
import time
import json
import numpy as np
import cv2
import yaml
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .apriltag_detector import AprilTagDetector, AprilTagDetection, load_camera_intrinsics


@dataclass
class CalibrationSample:
    """Single calibration sample with robot and tag poses."""
    sample_id: int
    timestamp: float

    # Robot end-effector pose (gripper → base)
    robot_position: List[float]    # [x, y, z] in meters
    robot_orientation: List[float]  # [qx, qy, qz, qw] quaternion or [rx, ry, rz] euler

    # AprilTag pose (tag → camera)
    tag_id: int
    tag_rvec: List[float]    # Rodrigues rotation vector
    tag_tvec: List[float]    # Translation vector [x, y, z] in meters

    # Optional: raw image path for debugging
    image_path: Optional[str] = None

    # Quality metrics
    reprojection_error: float = 0.0
    tag_decision_margin: float = 0.0


@dataclass
class CalibrationSession:
    """Calibration session data."""
    session_id: str
    created_at: float
    tag_size: float
    tag_family: str
    camera_intrinsics_path: Optional[str]

    samples: List[CalibrationSample] = field(default_factory=list)

    # Results
    R_cam2gripper: Optional[List[List[float]]] = None
    t_cam2gripper: Optional[List[float]] = None
    calibration_method: str = "TSAI"
    reprojection_error: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'CalibrationSession':
        samples = [CalibrationSample(**s) for s in data.pop('samples', [])]
        return cls(samples=samples, **data)


class HandEyeCalibrator:
    """
    Hand-Eye calibration manager.

    Workflow:
    1. Initialize with camera intrinsics and AprilTag settings
    2. Collect samples: robot_pose + camera_image
    3. Run calibration
    4. Validate and save results
    """

    METHODS = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        tag_size: float = 0.05,
        tag_family: str = "tag36h11",
        output_dir: str = "calibration_data",
    ):
        """
        Initialize calibrator.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            tag_size: AprilTag size in meters
            tag_family: AprilTag family
            output_dir: Directory to save calibration data
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.tag_size = tag_size
        self.tag_family = tag_family
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize AprilTag detector
        self.detector = AprilTagDetector(
            tag_family=tag_family,
            tag_size=tag_size,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )

        # Current session
        self.session: Optional[CalibrationSession] = None

    def new_session(self) -> CalibrationSession:
        """Start a new calibration session."""
        session_id = f"handeye_{int(time.time())}"
        self.session = CalibrationSession(
            session_id=session_id,
            created_at=time.time(),
            tag_size=self.tag_size,
            tag_family=self.tag_family,
            camera_intrinsics_path=None,
        )
        print(f"[Calibration] New session: {session_id}")
        return self.session

    def load_session(self, session_path: str) -> CalibrationSession:
        """Load existing session from file."""
        with open(session_path, 'r') as f:
            data = json.load(f)
        self.session = CalibrationSession.from_dict(data)
        print(f"[Calibration] Loaded session: {self.session.session_id} with {len(self.session.samples)} samples")
        return self.session

    def save_session(self, path: Optional[str] = None) -> str:
        """Save current session to file."""
        if self.session is None:
            raise ValueError("No active session")

        if path is None:
            path = str(self.output_dir / f"{self.session.session_id}.json")

        with open(path, 'w') as f:
            json.dump(self.session.to_dict(), f, indent=2)

        print(f"[Calibration] Session saved: {path}")
        return path

    def detect_tag(self, image: np.ndarray) -> Optional[AprilTagDetection]:
        """
        Detect AprilTag in image and return the best detection.

        Args:
            image: BGR image

        Returns:
            Best AprilTagDetection or None
        """
        detections = self.detector.detect(image, estimate_pose=True)

        if not detections:
            return None

        # Return detection with highest confidence
        return max(detections, key=lambda d: d.decision_margin)

    def add_sample(
        self,
        image: np.ndarray,
        robot_position: List[float],
        robot_orientation: List[float],
        save_image: bool = True,
    ) -> Optional[CalibrationSample]:
        """
        Add a calibration sample.

        Args:
            image: BGR image from camera
            robot_position: [x, y, z] end-effector position in base frame
            robot_orientation: [qx, qy, qz, qw] or [rx, ry, rz] orientation

        Returns:
            CalibrationSample if successful, None otherwise
        """
        if self.session is None:
            self.new_session()

        # Detect AprilTag
        detection = self.detect_tag(image)

        if detection is None:
            print("[Calibration] No AprilTag detected in image")
            return None

        if detection.rvec is None:
            print("[Calibration] Could not estimate tag pose")
            return None

        # Create sample
        sample_id = len(self.session.samples)
        sample = CalibrationSample(
            sample_id=sample_id,
            timestamp=time.time(),
            robot_position=list(robot_position),
            robot_orientation=list(robot_orientation),
            tag_id=detection.tag_id,
            tag_rvec=detection.rvec.tolist(),
            tag_tvec=detection.tvec.tolist(),
            tag_decision_margin=detection.decision_margin,
        )

        # Optionally save image
        if save_image:
            img_path = self.output_dir / f"sample_{sample_id:03d}.jpg"
            vis = self.detector.draw_detections(image, [detection])
            cv2.imwrite(str(img_path), vis)
            sample.image_path = str(img_path)

        self.session.samples.append(sample)
        print(f"[Calibration] Added sample {sample_id}: tag_id={detection.tag_id}, "
              f"margin={detection.decision_margin:.2f}")

        return sample

    def get_sample_count(self) -> int:
        """Get number of samples in current session."""
        if self.session is None:
            return 0
        return len(self.session.samples)

    def remove_last_sample(self) -> bool:
        """Remove the last added sample."""
        if self.session and self.session.samples:
            sample = self.session.samples.pop()
            if sample.image_path and os.path.exists(sample.image_path):
                os.remove(sample.image_path)
            print(f"[Calibration] Removed sample {sample.sample_id}")
            return True
        return False

    def calibrate(
        self,
        method: str = "TSAI",
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Run hand-eye calibration.

        Args:
            method: Calibration method (TSAI, PARK, HORAUD, ANDREFF, DANIILIDIS)

        Returns:
            (R_cam2gripper, t_cam2gripper, reprojection_error)
        """
        if self.session is None or len(self.session.samples) < 3:
            print("[Calibration] Need at least 3 samples")
            return None, None, float('inf')

        # Prepare data for calibrateHandEye
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        for sample in self.session.samples:
            # Robot pose (gripper → base)
            R_g2b = self._orientation_to_rotation_matrix(sample.robot_orientation)
            t_g2b = np.array(sample.robot_position).reshape(3, 1)
            R_gripper2base.append(R_g2b)
            t_gripper2base.append(t_g2b)

            # Tag pose (target → camera)
            R_t2c, _ = cv2.Rodrigues(np.array(sample.tag_rvec))
            t_t2c = np.array(sample.tag_tvec).reshape(3, 1)
            R_target2cam.append(R_t2c)
            t_target2cam.append(t_t2c)

        # Run calibration
        cv_method = self.METHODS.get(method.upper(), cv2.CALIB_HAND_EYE_TSAI)

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base=R_gripper2base,
            t_gripper2base=t_gripper2base,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
            method=cv_method,
        )

        # Calculate reprojection error
        error = self._calculate_reprojection_error(
            R_cam2gripper, t_cam2gripper,
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
        )

        # Store results
        self.session.R_cam2gripper = R_cam2gripper.tolist()
        self.session.t_cam2gripper = t_cam2gripper.flatten().tolist()
        self.session.calibration_method = method
        self.session.reprojection_error = error

        print(f"[Calibration] Completed with method={method}, error={error:.6f}")

        return R_cam2gripper, t_cam2gripper, error

    def calibrate_all_methods(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
        """
        Run calibration with all methods and compare.

        Returns:
            Dict of method -> (R, t, error)
        """
        results = {}

        for method in self.METHODS.keys():
            R, t, error = self.calibrate(method)
            if R is not None:
                results[method] = (R.copy(), t.copy(), error)
                print(f"  {method}: error = {error:.6f}")

        # Find best method
        if results:
            best = min(results.items(), key=lambda x: x[1][2])
            print(f"\n[Calibration] Best method: {best[0]} (error={best[1][2]:.6f})")

        return results

    def save_calibration(
        self,
        output_path: str = "hand_eye_calibration.yaml",
    ) -> str:
        """
        Save calibration result to YAML file.

        Args:
            output_path: Output file path

        Returns:
            Path to saved file
        """
        if self.session is None or self.session.R_cam2gripper is None:
            raise ValueError("No calibration result to save")

        data = {
            "R_cam2gripper": self.session.R_cam2gripper,
            "t_cam2gripper": self.session.t_cam2gripper,
            "method": self.session.calibration_method,
            "reprojection_error": self.session.reprojection_error,
            "num_samples": len(self.session.samples),
            "tag_size": self.session.tag_size,
            "tag_family": self.session.tag_family,
            "session_id": self.session.session_id,
        }

        with open(output_path, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False)

        print(f"[Calibration] Saved to {output_path}")
        return output_path

    def _orientation_to_rotation_matrix(self, orientation: List[float]) -> np.ndarray:
        """Convert orientation to rotation matrix."""
        if len(orientation) == 4:
            # Quaternion [qx, qy, qz, qw]
            return self._quaternion_to_rotation_matrix(orientation)
        elif len(orientation) == 3:
            # Euler angles [rx, ry, rz] in radians
            return self._euler_to_rotation_matrix(orientation)
        else:
            raise ValueError(f"Invalid orientation: {orientation}")

    def _quaternion_to_rotation_matrix(self, q: List[float]) -> np.ndarray:
        """Convert quaternion [qx, qy, qz, qw] to rotation matrix."""
        qx, qy, qz, qw = q

        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
        ])

        return R

    def _euler_to_rotation_matrix(self, euler: List[float]) -> np.ndarray:
        """Convert Euler angles [rx, ry, rz] to rotation matrix."""
        from scipy.spatial.transform import Rotation
        return Rotation.from_euler('xyz', euler).as_matrix()

    def _calculate_reprojection_error(
        self,
        R_cam2gripper: np.ndarray,
        t_cam2gripper: np.ndarray,
        R_gripper2base: List[np.ndarray],
        t_gripper2base: List[np.ndarray],
        R_target2cam: List[np.ndarray],
        t_target2cam: List[np.ndarray],
    ) -> float:
        """Calculate reprojection error for validation."""
        errors = []

        for i in range(len(R_gripper2base)):
            # Compute target in base frame two ways:
            # 1. Through camera: base <- gripper <- camera <- target
            # 2. Using calibration result

            # T_gripper2base
            T_g2b = np.eye(4)
            T_g2b[:3, :3] = R_gripper2base[i]
            T_g2b[:3, 3:4] = t_gripper2base[i]

            # T_cam2gripper (calibration result)
            T_c2g = np.eye(4)
            T_c2g[:3, :3] = R_cam2gripper
            T_c2g[:3, 3:4] = t_cam2gripper

            # T_target2cam
            T_t2c = np.eye(4)
            T_t2c[:3, :3] = R_target2cam[i]
            T_t2c[:3, 3:4] = t_target2cam[i]

            # T_target2base = T_g2b @ T_c2g @ T_t2c
            T_t2b = T_g2b @ T_c2g @ T_t2c

            # The target position should be consistent across all samples
            errors.append(T_t2b[:3, 3])

        # Calculate variance as error metric
        errors = np.array(errors)
        mean_pos = np.mean(errors, axis=0)
        variance = np.mean(np.linalg.norm(errors - mean_pos, axis=1))

        return variance


def check_calibration_requirements() -> Dict[str, Any]:
    """
    Check what's available/missing for hand-eye calibration.

    Returns:
        Dict with status of each component
    """
    results = {
        "apriltag_library": False,
        "apriltag_library_name": None,
        "camera_intrinsics": False,
        "camera_intrinsics_path": None,
        "robot_interface": False,
        "robot_interface_type": None,
        "missing": [],
        "ready": False,
    }

    # Check AprilTag libraries
    for lib_name in ["pupil_apriltags", "dt_apriltags", "apriltag"]:
        try:
            __import__(lib_name)
            results["apriltag_library"] = True
            results["apriltag_library_name"] = lib_name
            break
        except ImportError:
            pass

    if not results["apriltag_library"]:
        results["missing"].append("AprilTag library (pip install pupil-apriltags)")

    # Check for camera intrinsics file
    intrinsics_paths = [
        "camera_intrinsics.yaml",
        "camera_intrinsics_chess.yaml",
        "config/camera_intrinsics.yaml",
    ]
    for path in intrinsics_paths:
        if os.path.exists(path):
            results["camera_intrinsics"] = True
            results["camera_intrinsics_path"] = path
            break

    if not results["camera_intrinsics"]:
        results["missing"].append("Camera intrinsics (run camera calibration first)")

    # Check for robot interface
    try:
        from interbotix_xs_modules.arm import InterbotixManipulatorXS
        results["robot_interface"] = True
        results["robot_interface_type"] = "interbotix"
    except ImportError:
        results["missing"].append("Robot interface (interbotix_xs_modules)")

    # Overall status
    results["ready"] = (
        results["apriltag_library"] and
        results["camera_intrinsics"]
        # Robot interface is optional - can input poses manually
    )

    return results
