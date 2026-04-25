#!/usr/bin/env python3
"""
Vision Service API

Provides REST API for chess board vision analysis.
Wraps the chess_vision pipeline.

Endpoints:
    POST /capture       - Capture and analyze current board state
    POST /analyze       - Analyze an uploaded image
    GET  /health        - Health check
    GET  /status        - Get service status
    GET  /camera/frame  - Get raw camera frame as JPEG
    POST /detect/apriltag - Detect AprilTags in current frame
"""

import time
import base64
import io
import traceback
from typing import Optional, List
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
import asyncio

# Import vision modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chess_vision.vision_pipeline import ChessVisionPipeline
from chess_vision.chess_algo import (
    detect_pieces,
    detect_and_classify_pieces,
    draw_segmentation_overlay,
    draw_detection_boxes,
    draw_classification_results,
    draw_final_board,
)
from chess_vision.camera import RealSenseCamera, CameraIntrinsics, deproject_pixel_to_point
from chess_vision.apriltag_detector import AprilTagDetector
from chess_vision.depth_utils import (
    compute_piece_heights,
    check_path_clearance,
    validate_piece_by_height,
)


# Global instances
camera: Optional[RealSenseCamera] = None
pipeline: Optional[ChessVisionPipeline] = None
apriltag_detector: Optional[AprilTagDetector] = None


class BoardStateResponse(BaseModel):
    """Response model for board state."""
    success: bool
    fen: Optional[str] = None
    ascii_board: Optional[str] = None
    piece_positions: Optional[dict] = None
    is_valid: Optional[bool] = None
    warnings: Optional[list] = None
    depth_info: Optional[dict] = None
    height_warnings: Optional[list] = None
    timestamp: Optional[float] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    camera_connected: bool
    pipeline_ready: bool
    timestamp: float


class StatusResponse(BaseModel):
    """Service status response."""
    service: str
    version: str
    camera_connected: bool
    pipeline_ready: bool
    apriltag_ready: bool = False
    last_capture_time: Optional[float] = None


class AprilTagDetectionResult(BaseModel):
    """Single AprilTag detection."""
    tag_id: int
    center: List[float]
    corners: List[List[float]]
    pose_valid: bool = False
    tvec: Optional[List[float]] = None  # Translation [x, y, z] in meters
    rvec: Optional[List[float]] = None  # Rotation vector


class AprilTagResponse(BaseModel):
    """AprilTag detection response."""
    success: bool
    detections: List[AprilTagDetectionResult] = []
    image_base64: Optional[str] = None  # Annotated image
    timestamp: float
    error: Optional[str] = None


class CalibrationPoint(BaseModel):
    """A calibration point with tag ID and 3D position."""
    tag_id: int
    position: List[float]  # [x, y, z] in meters (camera frame)
    pixel: List[float]  # [u, v] pixel coordinates


class CalibrationDetectionResponse(BaseModel):
    """Response for calibration tag detection."""
    success: bool
    points: List[CalibrationPoint] = []
    image_base64: Optional[str] = None
    timestamp: float
    error: Optional[str] = None


# Track last capture time, grid, and depth
last_capture_time: Optional[float] = None
last_grid_9x9: Optional[np.ndarray] = None
last_depth_img = None    # cached depth array (numpy uint16)
last_board_surface_z_cam: Optional[float] = None  # last measured board surface Z in camera frame

# Depth retry settings
DEPTH_RETRY_ATTEMPTS = 3
DEPTH_RETRY_DELAY = 0.1  # seconds


def _get_frame_with_depth_retry():
    """
    Get a camera frame, retrying depth acquisition if the first attempt
    returns a color frame but no depth image.

    Returns (color_img, depth_img) — pure numpy arrays, no SDK objects.
    """
    import time as _time

    color_img, depth_img = camera.get_frame()
    if color_img is not None and depth_img is None:
        for attempt in range(DEPTH_RETRY_ATTEMPTS):
            _time.sleep(DEPTH_RETRY_DELAY)
            ci, di = camera.get_frame()
            if di is not None:
                print(f"[Vision] Depth frame recovered on retry #{attempt+1}")
                depth_img = di
                if ci is not None:
                    color_img = ci
                break
        else:
            print(f"[Vision] Depth frame unavailable after {DEPTH_RETRY_ATTEMPTS} retries — proceeding without depth")

    return color_img, depth_img


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    global camera, pipeline, apriltag_detector

    print("[Vision Service] Initializing...")

    # Initialize camera with higher resolution (1280x720 is safe for USB bandwidth)
    try:
        camera = RealSenseCamera(
            color_res=(1280, 720, 30),
            depth_res=(1280, 720, 30),
        )
        print("[Vision Service] Camera connected (1280x720)")

        # Get camera intrinsics for AprilTag pose estimation
        camera_matrix = np.array([
            [camera.color_intrinsics.fx, 0, camera.color_intrinsics.ppx],
            [0, camera.color_intrinsics.fy, camera.color_intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.array(camera.color_intrinsics.coeffs, dtype=np.float64)

        # Initialize AprilTag detector with camera params
        apriltag_detector = AprilTagDetector(
            tag_family="tag36h11",
            tag_size=0.045,  # 45mm measured
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        print("[Vision Service] AprilTag detector ready")
    except Exception as e:
        print(f"[Vision Service] Camera init failed: {e}")
        camera = None
        apriltag_detector = None

    # Initialize pipeline
    try:
        pipeline = ChessVisionPipeline(lazy_load=True)
        print("[Vision Service] Pipeline ready")
    except Exception as e:
        print(f"[Vision Service] Pipeline init failed: {e}")
        pipeline = None

    yield

    # Cleanup
    if camera:
        camera.release()
    print("[Vision Service] Shutdown complete")


app = FastAPI(
    title="Chess Vision API",
    description="REST API for chess board computer vision: board detection, piece recognition, depth measurement. Camera-agnostic — reference implementation uses Intel RealSense D4xx.",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        camera_connected=camera is not None,
        pipeline_ready=pipeline is not None,
        timestamp=time.time(),
    )


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get service status."""
    return StatusResponse(
        service="vision",
        version="1.0.0",
        camera_connected=camera is not None,
        pipeline_ready=pipeline is not None,
        apriltag_ready=apriltag_detector is not None,
        last_capture_time=last_capture_time,
    )


@app.get("/camera/frame")
async def get_camera_frame(annotate: bool = Query(False, description="Draw crosshair on image")):
    """
    Get raw camera frame as JPEG.

    Returns:
        JPEG image
    """
    if camera is None:
        raise HTTPException(status_code=503, detail="Camera not connected")

    try:
        color_img, _ = camera.get_frame()

        if color_img is None:
            raise HTTPException(status_code=500, detail="Failed to capture frame")

        if annotate:
            h, w = color_img.shape[:2]
            cx, cy = w // 2, h // 2
            cv2.line(color_img, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
            cv2.line(color_img, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)

        _, jpeg = cv2.imencode('.jpg', color_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return Response(content=jpeg.tobytes(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/camera/depth")
async def get_depth_frame(colormap: bool = Query(True, description="Apply colormap to depth")):
    """
    Get depth camera frame as JPEG.

    Returns:
        JPEG image (colorized depth)
    """
    if camera is None:
        raise HTTPException(status_code=503, detail="Camera not connected")

    try:
        _, depth_img = camera.get_frame()

        if depth_img is None:
            raise HTTPException(status_code=500, detail="Failed to capture depth frame")

        if colormap:
            # Normalize and apply colormap
            depth_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        else:
            # Just normalize to 8-bit grayscale
            depth_colored = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        _, jpeg = cv2.imencode('.jpg', depth_colored, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return Response(content=jpeg.tobytes(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_mjpeg_stream(stream_type: str = "rgb"):
    """Generator for MJPEG stream."""
    while True:
        if camera is None:
            break

        try:
            color_img, depth_img = camera.get_frame()

            if stream_type == "rgb" and color_img is not None:
                frame = color_img
            elif stream_type == "depth" and depth_img is not None:
                depth_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                frame = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            else:
                continue

            _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        except Exception:
            break


@app.get("/camera/stream/rgb")
async def stream_rgb():
    """
    MJPEG stream of RGB camera.

    Returns:
        Multipart MJPEG stream
    """
    if camera is None:
        raise HTTPException(status_code=503, detail="Camera not connected")

    return StreamingResponse(
        generate_mjpeg_stream("rgb"),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/camera/stream/depth")
async def stream_depth():
    """
    MJPEG stream of depth camera (colorized).

    Returns:
        Multipart MJPEG stream
    """
    if camera is None:
        raise HTTPException(status_code=503, detail="Camera not connected")

    return StreamingResponse(
        generate_mjpeg_stream("depth"),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


def generate_yolo_stream():
    """Generator for MJPEG stream with real-time YOLO detection overlays."""
    while True:
        if camera is None or pipeline is None:
            break

        try:
            color_img, depth_img = camera.get_frame()
            if color_img is None:
                continue

            pipeline._ensure_models_loaded()

            # Run YOLO detection only (no classification)
            pieces = detect_pieces(
                pipeline._det_model,
                color_img,
            )

            # Draw detection results on the frame
            annotated = draw_detection_boxes(
                color_img,
                np.array([p["bbox"] for p in pieces], dtype=np.float32) if pieces else np.array([]),
                np.array([p["det_conf"] for p in pieces], dtype=np.float32) if pieces else np.array([]),
            )

            _, jpeg = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        except Exception:
            break


@app.get("/camera/stream/yolo")
async def stream_yolo():
    """
    MJPEG stream with real-time YOLO piece detection overlays.

    Returns:
        Multipart MJPEG stream with bounding boxes and classifications
    """
    if camera is None:
        raise HTTPException(status_code=503, detail="Camera not connected")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Vision pipeline not ready")

    return StreamingResponse(
        generate_yolo_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/detect/apriltag", response_model=AprilTagResponse)
async def detect_apriltag(return_image: bool = Query(True, description="Return annotated image")):
    """
    Detect AprilTags in current camera frame.

    Returns:
        AprilTagResponse with detections and optional annotated image
    """
    if camera is None:
        return AprilTagResponse(
            success=False,
            timestamp=time.time(),
            error="Camera not connected"
        )

    if apriltag_detector is None:
        return AprilTagResponse(
            success=False,
            timestamp=time.time(),
            error="AprilTag detector not initialized"
        )

    try:
        color_img, _ = camera.get_frame()

        if color_img is None:
            return AprilTagResponse(
                success=False,
                timestamp=time.time(),
                error="Failed to capture frame"
            )

        # Detect AprilTags
        detections = apriltag_detector.detect(color_img, estimate_pose=True)

        results = []
        for det in detections:
            result = AprilTagDetectionResult(
                tag_id=det.tag_id,
                center=det.center.tolist(),
                corners=det.corners.tolist(),
                pose_valid=det.tvec is not None,
            )
            if det.tvec is not None:
                result.tvec = det.tvec.tolist()
                result.rvec = det.rvec.tolist()
            results.append(result)

        # Generate annotated image
        image_base64 = None
        if return_image:
            vis = apriltag_detector.draw_detections(color_img, detections)
            _, jpeg = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            image_base64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

        return AprilTagResponse(
            success=True,
            detections=results,
            image_base64=image_base64,
            timestamp=time.time(),
        )

    except Exception as e:
        return AprilTagResponse(
            success=False,
            timestamp=time.time(),
            error=str(e)
        )


@app.get("/calibration/save_camera_frame")
async def save_camera_frame():
    """Save current camera frame and run detection on it."""
    if camera is None:
        return {"success": False, "error": "Camera not connected"}
    if apriltag_detector is None:
        return {"success": False, "error": "Detector not initialized"}

    try:
        # Capture frame
        color_img, _ = camera.get_frame()
        if color_img is None:
            return {"success": False, "error": "Failed to capture frame"}

        # Save original
        cv2.imwrite('/app/captures/camera_capture.png', color_img)

        # Run detection
        detections = apriltag_detector.detect(color_img, estimate_pose=False)

        # Draw detections on image
        vis = color_img.copy()
        for det in detections:
            corners = det.corners.astype(int)
            cv2.polylines(vis, [corners], True, (0, 255, 0), 2)
            cx, cy = int(det.center[0]), int(det.center[1])
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(vis, f"ID:{det.tag_id}", (cx-20, cy-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if len(detections) == 0:
            cv2.putText(vis, "NO TAGS DETECTED", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Save annotated
        cv2.imwrite('/app/captures/camera_capture_annotated.png', vis)

        # Return base64 image
        _, jpeg = cv2.imencode('.jpg', vis)
        image_base64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

        return {
            "success": True,
            "image_shape": list(color_img.shape),
            "num_detections": len(detections),
            "detections": [{"tag_id": d.tag_id, "center": d.center.tolist()} for d in detections],
            "saved_to": ["./captures/camera_capture.png", "./captures/camera_capture_annotated.png"],
            "image_base64": image_base64
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.get("/calibration/test_detector")
async def test_apriltag_detector():
    """Test AprilTag detector with downloaded official tag image."""
    import urllib.request

    if apriltag_detector is None:
        return {"success": False, "error": "Detector not initialized"}

    try:
        # Download official tag36h11 ID=0 image
        url = "https://raw.githubusercontent.com/AprilRobotics/apriltag-imgs/master/tag36h11/tag36_11_00000.png"
        with urllib.request.urlopen(url, timeout=5) as response:
            img_data = np.asarray(bytearray(response.read()), dtype=np.uint8)
            img_small = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

        if img_small is None:
            return {"success": False, "error": "Failed to decode downloaded image"}

        # The downloaded image is tiny (10x10), scale it up with white border
        scale = 20
        margin = 60
        img_scaled = cv2.resize(img_small, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        # Add white margin
        h, w = img_scaled.shape[:2]
        img_bgr = np.ones((h + 2*margin, w + 2*margin, 3), dtype=np.uint8) * 255
        img_bgr[margin:margin+h, margin:margin+w] = img_scaled

        print(f"[Test] Image shape after scaling: {img_bgr.shape}")

        # Run detection
        detections = apriltag_detector.detect(img_bgr, estimate_pose=False)
        print(f"[Test] Detected {len(detections)} tags")

        result = {
            "success": True,
            "backend": apriltag_detector.backend,
            "image_shape": list(img_bgr.shape),
            "num_detections": len(detections),
            "detections": [{"tag_id": d.tag_id, "center": d.center.tolist()} for d in detections]
        }

        # Encode image
        _, jpeg = cv2.imencode('.jpg', img_bgr)
        result["image_base64"] = base64.b64encode(jpeg.tobytes()).decode('utf-8')

        return result

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/calibration/detect_tags", response_model=CalibrationDetectionResponse)
async def detect_calibration_tags(return_image: bool = Query(True, description="Return annotated image")):
    """
    Detect AprilTags for calibration. Returns 3D positions using depth camera.

    Returns:
        CalibrationDetectionResponse with tag center positions in camera frame
    """
    if camera is None:
        return CalibrationDetectionResponse(
            success=False,
            timestamp=time.time(),
            error="Camera not connected"
        )

    if apriltag_detector is None:
        return CalibrationDetectionResponse(
            success=False,
            timestamp=time.time(),
            error="AprilTag detector not initialized"
        )

    try:
        # Get both color and depth frames
        color_img, depth_img = camera.get_frame()

        if color_img is None:
            return CalibrationDetectionResponse(
                success=False,
                timestamp=time.time(),
                error="Failed to capture frame"
            )

        # Detect AprilTags
        print(f"[Calibration] Image shape: {color_img.shape}, dtype: {color_img.dtype}")
        print(f"[Calibration] Image min/max: {color_img.min()}/{color_img.max()}")
        print(f"[Calibration] Using backend: {apriltag_detector.backend}")
        print(f"[Calibration] Detector object: {apriltag_detector.detector}")
        detections = apriltag_detector.detect(color_img, estimate_pose=True)
        print(f"[Calibration] Detected {len(detections)} AprilTags")

        points = []
        for det in detections:
            # Get tag center pixel coordinates
            cx, cy = int(det.center[0]), int(det.center[1])

            # Get depth at tag center
            if depth_img is not None:
                depth_scale = camera.get_depth_scale()
                depth_value = float(depth_img[cy, cx]) * depth_scale

                # Deproject pixel to 3D point using camera intrinsics
                intrinsics = camera.get_intrinsics()
                point_3d = deproject_pixel_to_point(
                    intrinsics, [cx, cy], depth_value
                )
                position = list(point_3d)
            elif det.tvec is not None:
                # Fallback to AprilTag pose estimation
                position = det.tvec.flatten().tolist()
            else:
                continue  # Skip if no 3D position available

            points.append(CalibrationPoint(
                tag_id=det.tag_id,
                position=position,
                pixel=[float(cx), float(cy)]
            ))

        # Generate annotated image (always return image for debugging)
        image_base64 = None
        if return_image:
            vis = color_img.copy()
            if len(detections) == 0:
                # Show message when no tags found
                cv2.putText(vis, "No AprilTags detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                for det in detections:
                    # Draw tag outline
                    corners = det.corners.astype(int)
                    cv2.polylines(vis, [corners], True, (0, 255, 0), 2)
                    # Draw center
                    cx, cy = int(det.center[0]), int(det.center[1])
                    cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
                    # Draw tag ID and position
                    for p in points:
                        if p.tag_id == det.tag_id:
                            label = f"ID:{det.tag_id} ({p.position[0]:.3f}, {p.position[1]:.3f}, {p.position[2]:.3f})"
                            cv2.putText(vis, label, (cx - 50, cy - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Save annotated image for debugging
            cv2.imwrite('/app/captures/calibration_annotated.png', vis)

            _, jpeg = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            image_base64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

        return CalibrationDetectionResponse(
            success=True,
            points=points,
            image_base64=image_base64,
            timestamp=time.time(),
        )

    except Exception as e:
        traceback.print_exc()
        return CalibrationDetectionResponse(
            success=False,
            timestamp=time.time(),
            error=str(e)
        )


OCCUPANCY_DURATION = 1.0  # Seconds to continuously capture for robust occupancy


@app.post("/capture/occupancy")
async def capture_occupancy():
    """
    Continuously capture for OCCUPANCY_DURATION seconds and return the UNION of occupied squares.

    Runs the pipeline on every frame captured during the window.
    If a piece is detected on a square even once, it counts as occupied.

    Returns:
        {success, occupied_squares: ["a1",...], total_count, shots, per_shot_counts, duration, timestamp}
    """
    global last_capture_time, last_grid_9x9, last_depth_img

    if camera is None:
        return {"success": False, "error": "Camera not connected", "timestamp": time.time()}
    if pipeline is None:
        return {"success": False, "error": "Pipeline not initialized", "timestamp": time.time()}

    try:
        union_occupied = set()
        piece_votes = {}  # {square: {cls_name: count}} for majority-vote piece class
        per_shot_counts = []
        good_shots = 0
        total_shots = 0

        t_start = time.time()
        while time.time() - t_start < OCCUPANCY_DURATION:
            total_shots += 1
            color_img, depth_img = camera.get_frame()
            if color_img is None:
                per_shot_counts.append(0)
                continue

            result = pipeline.analyze_image(color_img)

            if result is None:
                per_shot_counts.append(0)
                continue

            good_shots += 1
            shot_squares = set(result.board_state.keys())
            per_shot_counts.append(len(shot_squares))
            union_occupied |= shot_squares

            # Track piece class votes per square
            for sq, info in result.board_state.items():
                if sq not in piece_votes:
                    piece_votes[sq] = {}
                cls = info.get("piece", "piece")
                piece_votes[sq][cls] = piece_votes[sq].get(cls, 0) + 1

            # Keep last successful result for grid/depth
            last_grid_9x9 = result.grid_9x9
            if depth_img is not None:
                last_depth_img = depth_img

        elapsed = time.time() - t_start
        last_capture_time = time.time()

        if good_shots == 0:
            return {"success": False, "error": "Board not detected in any frame", "timestamp": last_capture_time}

        # Build piece_positions from majority vote
        piece_positions = {}
        for sq in union_occupied:
            if sq in piece_votes:
                best_cls = max(piece_votes[sq], key=piece_votes[sq].get)
                piece_positions[sq] = best_cls

        occupied_squares = sorted(union_occupied)
        print(f"[occupancy] {total_shots} shots in {elapsed:.1f}s, good={good_shots}, union={len(occupied_squares)}")
        return {
            "success": True,
            "occupied_squares": occupied_squares,
            "piece_positions": piece_positions,
            "total_count": len(occupied_squares),
            "shots": total_shots,
            "good_shots": good_shots,
            "per_shot_counts": per_shot_counts,
            "duration": round(elapsed, 2),
            "timestamp": last_capture_time,
        }

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e), "timestamp": time.time()}


@app.post("/capture/occupancy_init")
async def capture_occupancy_init():
    """
    Continuously capture for OCCUPANCY_DURATION seconds with occupancy + orientation hint.

    Unions occupied squares across all frames. If a piece is detected on a
    square even once, it counts. Also aggregates white-piece orientation votes.

    Returns:
        {success, occupied_squares, white_side: "bottom"|"top", timestamp}
    """
    global last_capture_time, last_grid_9x9, last_depth_img

    if camera is None:
        return {"success": False, "error": "Camera not connected", "timestamp": time.time()}
    if pipeline is None:
        return {"success": False, "error": "Pipeline not initialized", "timestamp": time.time()}

    try:
        union_occupied = set()
        total_white_bottom = 0
        total_white_top = 0
        per_shot_counts = []
        good_shots = 0
        total_shots = 0

        t_start = time.time()
        while time.time() - t_start < OCCUPANCY_DURATION:
            total_shots += 1
            color_img, depth_img = camera.get_frame()
            if color_img is None:
                per_shot_counts.append(0)
                continue

            result = pipeline.analyze_image(color_img)

            if result is None:
                per_shot_counts.append(0)
                continue

            good_shots += 1
            shot_squares = set(result.board_state.keys())
            per_shot_counts.append(len(shot_squares))
            union_occupied |= shot_squares

            last_grid_9x9 = result.grid_9x9
            if depth_img is not None:
                last_depth_img = depth_img

            # Accumulate orientation votes from this shot
            for sq, info in result.board_state.items():
                piece_name = info.get("piece", "") if isinstance(info, dict) else str(info)
                rank = int(sq[1])
                if piece_name.startswith("white_"):
                    if rank <= 2:
                        total_white_bottom += 1
                    elif rank >= 7:
                        total_white_top += 1

        elapsed = time.time() - t_start
        last_capture_time = time.time()

        if good_shots == 0:
            return {"success": False, "error": "Board not detected in any frame", "timestamp": last_capture_time}

        occupied_squares = sorted(union_occupied)
        white_side = "bottom" if total_white_bottom >= total_white_top else "top"
        print(f"[occupancy_init] {total_shots} shots in {elapsed:.1f}s, good={good_shots}, union={len(occupied_squares)}, "
              f"white_bottom={total_white_bottom}, white_top={total_white_top} -> {white_side}")

        return {
            "success": True,
            "occupied_squares": occupied_squares,
            "total_count": len(occupied_squares),
            "white_side": white_side,
            "white_bottom_count": total_white_bottom,
            "white_top_count": total_white_top,
            "shots": total_shots,
            "good_shots": good_shots,
            "per_shot_counts": per_shot_counts,
            "duration": round(elapsed, 2),
            "timestamp": last_capture_time,
        }

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e), "timestamp": time.time()}


@app.post("/capture", response_model=BoardStateResponse)
async def capture_and_analyze():
    """
    Capture image from camera and analyze board state.

    Returns:
        BoardStateResponse with FEN, piece positions, etc.
    """
    global last_capture_time, last_grid_9x9, last_depth_img, last_board_surface_z_cam

    if camera is None:
        raise HTTPException(status_code=503, detail="Camera not connected")

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Capture frame (color + depth) with depth retry
        color_img, depth_img = _get_frame_with_depth_retry()

        if color_img is None:
            return BoardStateResponse(
                success=False,
                error="Failed to capture camera frame",
                timestamp=time.time(),
            )

        # Analyze with pipeline
        result = pipeline.analyze_image(color_img)

        last_capture_time = time.time()

        # Cache grid and depth for /square_position endpoint
        if result is not None:
            last_grid_9x9 = result.grid_9x9
            if depth_img is not None:
                last_depth_img = depth_img

        if result is None:
            return BoardStateResponse(
                success=False,
                error="Board not detected in image",
                timestamp=last_capture_time,
            )

        # Measure piece heights using depth data
        depth_info = None
        height_warnings = []
        if depth_img is not None and result.board_state:
            try:
                intrinsics = camera.get_intrinsics()
                heights = compute_piece_heights(
                    board_state=result.board_state,
                    depth_map=depth_img,
                    depth_scale=camera.get_depth_scale(),
                    intrinsics=intrinsics,
                    grid_9x9=result.grid_9x9,
                )
                if heights:
                    depth_info = {sq: info["height_m"] for sq, info in heights.items()}

                    # Cache board surface Z for fallback use
                    first_entry = next(iter(heights.values()), None)
                    if first_entry and "board_z_cam" in first_entry:
                        last_board_surface_z_cam = first_entry["board_z_cam"]

                    # Validate each piece by height
                    for sq, height_data in heights.items():
                        piece_info = result.board_state.get(sq)
                        if piece_info:
                            validation = validate_piece_by_height(
                                measured_height=height_data["height_m"],
                                classified_type=piece_info["piece"],
                            )
                            if not validation["valid"] and validation["suggested_type"]:
                                height_warnings.append(
                                    f"{sq}: classified as {piece_info['piece']} "
                                    f"but height {height_data['height_m']*100:.1f}cm "
                                    f"suggests {validation['suggested_type']}"
                                )
            except Exception as e:
                print(f"[Capture] Depth measurement error: {e}")

        return BoardStateResponse(
            success=True,
            fen=result.get_fen(),
            ascii_board=result.get_ascii_board(),
            piece_positions=result.board_state,
            is_valid=result.analysis["is_valid"],
            warnings=result.analysis.get("warnings", []),
            depth_info=depth_info,
            height_warnings=height_warnings if height_warnings else None,
            timestamp=last_capture_time,
        )

    except Exception as e:
        return BoardStateResponse(
            success=False,
            error=str(e),
            timestamp=time.time(),
        )


@app.get("/square_position/{square}")
async def get_square_position(square: str):
    """
    Get the 3D camera coordinates of a chess square center.
    Uses the grid from the last capture and current depth.

    Args:
        square: Chess square name (e.g., "e4")

    Returns:
        {"success": true, "camera_xyz": [x, y, z], "pixel_xy": [px, py]}
    """
    global last_grid_9x9, last_depth_img

    square = square.lower()
    if len(square) != 2 or square[0] not in "abcdefgh" or square[1] not in "12345678":
        return {"success": False, "error": f"Invalid square: {square}"}

    if last_grid_9x9 is None:
        return {"success": False, "error": "No board captured yet. Run /capture first."}

    if camera is None:
        return {"success": False, "error": "Camera not connected"}

    try:
        # Get file/rank indices
        file_idx = ord(square[0]) - ord('a')  # 0-7
        rank_idx = int(square[1]) - 1          # 0-7

        # Grid is 9x9 intersections. Cell (file, rank) has corners at:
        # grid[7-rank, file], grid[7-rank, file+1], grid[8-rank, file], grid[8-rank, file+1]
        # (grid row 0 = rank 8, row 7 = rank 1)
        row = 7 - rank_idx
        col = file_idx

        # Get the 4 corners of this square
        tl = last_grid_9x9[row, col]
        tr = last_grid_9x9[row, col + 1]
        bl = last_grid_9x9[row + 1, col]
        br = last_grid_9x9[row + 1, col + 1]

        # Square center in pixel coordinates
        center_x = (tl[0] + tr[0] + bl[0] + br[0]) / 4
        center_y = (tl[1] + tr[1] + bl[1] + br[1]) / 4
        px, py = int(round(center_x)), int(round(center_y))

        # Get depth at square center (use retry helper, then fallback to cached)
        _, depth_img_local = _get_frame_with_depth_retry()
        if depth_img_local is None:
            depth_img_local = last_depth_img

        intrinsics = camera.get_intrinsics()
        depth_scale = camera.get_depth_scale()

        if depth_img_local is None:
            # Final fallback: use cached board surface Z as approximate depth
            if last_board_surface_z_cam is not None:
                depth_m = last_board_surface_z_cam
                pt = deproject_pixel_to_point(intrinsics, [px, py], depth_m)
                return {
                    "success": True,
                    "square": square,
                    "pixel_xy": [px, py],
                    "camera_xyz": [round(pt[0], 6), round(pt[1], 6), round(pt[2], 6)],
                    "depth_source": "cached_board_z",
                }
            return {"success": False, "error": "No depth data available"}

        # Sample depth around center (average of 5x5 region)
        depths = []
        h_img, w_img = depth_img_local.shape[:2]
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                sx, sy = px + dx, py + dy
                if 0 <= sx < w_img and 0 <= sy < h_img:
                    d = float(depth_img_local[sy, sx]) * depth_scale
                    if 0.1 < d < 2.0:  # Valid range
                        depths.append(d)

        if not depths:
            return {"success": False, "error": f"No valid depth at square {square}"}

        depth_m = np.median(depths)
        pt = deproject_pixel_to_point(intrinsics, [px, py], depth_m)

        return {
            "success": True,
            "square": square,
            "pixel_xy": [px, py],
            "camera_xyz": [round(pt[0], 6), round(pt[1], 6), round(pt[2], 6)],
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/analyze", response_model=BoardStateResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze an uploaded image.

    Args:
        file: Image file (JPEG, PNG)

    Returns:
        BoardStateResponse with analysis results
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Read uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return BoardStateResponse(
                success=False,
                error="Failed to decode image",
                timestamp=time.time(),
            )

        # Analyze
        result = pipeline.analyze_image(img)

        if result is None:
            return BoardStateResponse(
                success=False,
                error="Board not detected in image",
                timestamp=time.time(),
            )

        return BoardStateResponse(
            success=True,
            fen=result.get_fen(),
            ascii_board=result.get_ascii_board(),
            piece_positions=result.board_state,
            is_valid=result.analysis["is_valid"],
            warnings=result.analysis.get("warnings", []),
            timestamp=time.time(),
        )

    except Exception as e:
        return BoardStateResponse(
            success=False,
            error=str(e),
            timestamp=time.time(),
        )


TRAINING_IMAGES_DIR = "/app/captures/training"


class SaveImageResponse(BaseModel):
    """Response for saving a training image."""
    success: bool
    filename: Optional[str] = None
    path: Optional[str] = None
    count: Optional[int] = None
    timestamp: float
    error: Optional[str] = None


@app.post("/capture/save_image", response_model=SaveImageResponse)
async def save_training_image(label: Optional[str] = Query(None, description="Optional label/subfolder for the image")):
    """
    Capture and save an RGB image for training data collection.

    Images are saved to /app/captures/training/ (mounted to ./captures/training/ on host).
    Optionally provide a label query param to save into a subfolder.
    """
    if camera is None:
        return SaveImageResponse(
            success=False, timestamp=time.time(), error="Camera not connected"
        )

    try:
        color_img, _ = camera.get_frame()
        if color_img is None:
            return SaveImageResponse(
                success=False, timestamp=time.time(), error="Failed to capture frame"
            )

        # Determine save directory
        save_dir = Path(TRAINING_IMAGES_DIR)
        if label:
            save_dir = save_dir / label
        save_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamped filename
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}.jpg"
        filepath = save_dir / filename

        cv2.imwrite(str(filepath), color_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Count images in this directory
        count = len(list(save_dir.glob("*.jpg")))

        return SaveImageResponse(
            success=True,
            filename=filename,
            path=str(filepath),
            count=count,
            timestamp=time.time(),
        )

    except Exception as e:
        traceback.print_exc()
        return SaveImageResponse(
            success=False, timestamp=time.time(), error=str(e)
        )


@app.get("/capture/training_count")
async def training_image_count(label: Optional[str] = Query(None)):
    """Get the number of saved training images."""
    save_dir = Path(TRAINING_IMAGES_DIR)
    if label:
        save_dir = save_dir / label
    if not save_dir.exists():
        return {"count": 0, "label": label}
    count = len(list(save_dir.glob("*.jpg")))
    return {"count": count, "label": label, "path": str(save_dir)}


class MeasurePiecesResponse(BaseModel):
    """Response for depth piece measurement."""
    success: bool
    pieces: Optional[dict] = None
    board_surface_z: Optional[float] = None
    timestamp: float
    error: Optional[str] = None


class CollisionCheckRequest(BaseModel):
    """Request for collision check."""
    start_xyz: List[float] = Field(..., description="[x,y,z] start in robot frame")
    end_xyz: List[float] = Field(..., description="[x,y,z] end in robot frame")
    calibration_transform: List[List[float]] = Field(..., description="4x4 camera-to-robot transform")


class CollisionCheckResponse(BaseModel):
    """Response for collision check."""
    success: bool
    clear: Optional[bool] = None
    obstacles: Optional[list] = None
    min_clearance: Optional[float] = None
    num_checked: Optional[int] = None
    timestamp: float
    error: Optional[str] = None


@app.post("/depth/measure_pieces", response_model=MeasurePiecesResponse)
async def measure_pieces_depth():
    """
    Capture color+depth frame, detect pieces, and measure their heights.

    Returns per-piece height data including height above board surface.
    """
    if camera is None:
        return MeasurePiecesResponse(
            success=False, timestamp=time.time(), error="Camera not connected"
        )
    if pipeline is None:
        return MeasurePiecesResponse(
            success=False, timestamp=time.time(), error="Pipeline not initialized"
        )

    try:
        color_img, depth_img = _get_frame_with_depth_retry()

        if color_img is None:
            return MeasurePiecesResponse(
                success=False, timestamp=time.time(),
                error="Failed to capture color frame"
            )
        if depth_img is None:
            depth_img = last_depth_img
        if depth_img is None:
            return MeasurePiecesResponse(
                success=False, timestamp=time.time(),
                error="No depth data available (live or cached)"
            )

        # Run piece detection
        result = pipeline.analyze_image(color_img)
        if result is None:
            return MeasurePiecesResponse(
                success=False, timestamp=time.time(),
                error="Board not detected in image"
            )

        intrinsics = camera.get_intrinsics()
        depth_scale = camera.get_depth_scale()
        heights = compute_piece_heights(
            board_state=result.board_state,
            depth_map=depth_img,
            depth_scale=depth_scale,
            intrinsics=intrinsics,
            grid_9x9=result.grid_9x9,
        )

        from chess_vision.depth_utils import measure_board_surface_z
        board_z = measure_board_surface_z(
            depth_img, depth_scale, intrinsics, result.grid_9x9, result.board_state
        )

        return MeasurePiecesResponse(
            success=True,
            pieces=heights,
            board_surface_z=round(board_z, 4) if board_z else None,
            timestamp=time.time(),
        )

    except Exception as e:
        traceback.print_exc()
        return MeasurePiecesResponse(
            success=False, timestamp=time.time(), error=str(e)
        )


@app.post("/depth/check_collision", response_model=CollisionCheckResponse)
async def check_collision(request: CollisionCheckRequest):
    """
    Check if a straight-line path between two robot positions is clear of obstacles.

    Uses depth camera to detect objects in the path at approach height.
    """
    if camera is None:
        return CollisionCheckResponse(
            success=False, timestamp=time.time(), error="Camera not connected"
        )

    try:
        _, depth_img = _get_frame_with_depth_retry()

        if depth_img is None:
            depth_img = last_depth_img
        if depth_img is None:
            return CollisionCheckResponse(
                success=False, timestamp=time.time(),
                error="No depth data available (live or cached)"
            )

        intrinsics = camera.get_intrinsics()
        depth_scale = camera.get_depth_scale()
        transform = np.array(request.calibration_transform)

        result = check_path_clearance(
            start_xyz=request.start_xyz,
            end_xyz=request.end_xyz,
            depth_map=depth_img,
            depth_scale=depth_scale,
            intrinsics=intrinsics,
            calibration_transform=transform,
        )

        return CollisionCheckResponse(
            success=True,
            clear=result["clear"],
            obstacles=result["obstacles"],
            min_clearance=result["min_clearance"],
            num_checked=result["num_checked"],
            timestamp=time.time(),
        )

    except Exception as e:
        traceback.print_exc()
        return CollisionCheckResponse(
            success=False, timestamp=time.time(), error=str(e)
        )


def _encode_jpeg_base64(img_bgr, quality=85):
    """Encode a BGR image as base64 JPEG string."""
    _, jpeg = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(jpeg.tobytes()).decode('utf-8')


@app.post("/capture/debug")
async def capture_debug():
    """
    Run the full vision pipeline step-by-step and return debug info
    with annotated images for each stage.
    """
    if camera is None:
        raise HTTPException(status_code=503, detail="Camera not connected")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        t_start = time.time()
        color_img, depth_img = camera.get_frame()

        if color_img is None:
            return {
                "success": False,
                "error": "Failed to capture camera frame",
                "steps": [],
                "total_duration_ms": 0,
                "timestamp": time.time(),
            }

        # Run debug pipeline
        debug_data = pipeline.analyze_image_debug(color_img)

        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d%H%M%S")

        steps = []

        # Step 1: Raw capture
        cv2.imwrite(f"/app/captures/debug_{ts}_1_capture.png", color_img)
        steps.append({
            "step_name": "capture",
            "step_label": "拍照 (Capture)",
            "success": True,
            "image_base64": _encode_jpeg_base64(color_img),
            "metadata": {"shape": list(color_img.shape)},
            "error": None,
            "duration_ms": None,
        })

        # Step 2: Board segmentation + grid
        seg_mask = debug_data.get("seg_mask")
        grid_info = debug_data.get("grid_info")
        if seg_mask is not None:
            vis_seg = draw_segmentation_overlay(color_img, seg_mask, grid_info)
            cv2.imwrite(f"/app/captures/debug_{ts}_2_segmentation.png", vis_seg)
            reproj_err = grid_info.get("reprojection_error") if grid_info else None
            seg_meta = {
                "mask_area_ratio": round(float(seg_mask.sum() / seg_mask.size), 4),
                "seg_duration_ms": round(debug_data.get("seg_duration_ms", 0), 1),
                "grid_duration_ms": round(debug_data.get("grid_duration_ms", 0), 1),
                "grid_found": grid_info is not None,
                "reprojection_max_px": reproj_err["max_px"] if reproj_err else None,
                "reprojection_mean_px": reproj_err["mean_px"] if reproj_err else None,
            }
            steps.append({
                "step_name": "segmentation",
                "step_label": "辨識棋盤 (Board Detection)",
                "success": grid_info is not None,
                "image_base64": _encode_jpeg_base64(vis_seg),
                "metadata": seg_meta,
                "error": None if grid_info is not None else "Grid extraction failed",
                "duration_ms": round(debug_data.get("seg_duration_ms", 0) + debug_data.get("grid_duration_ms", 0), 1),
            })
        else:
            steps.append({
                "step_name": "segmentation",
                "step_label": "辨識棋盤 (Board Detection)",
                "success": False,
                "image_base64": _encode_jpeg_base64(color_img),
                "metadata": {"seg_duration_ms": round(debug_data.get("seg_duration_ms", 0), 1)},
                "error": "No chessboard detected in image",
                "duration_ms": round(debug_data.get("seg_duration_ms", 0), 1),
            })

        # Step 2.5: Board-masked image
        masked_img = debug_data.get("masked_image")
        if masked_img is not None:
            cv2.imwrite(f"/app/captures/debug_{ts}_2b_masked.png", masked_img)
            steps.append({
                "step_name": "masked",
                "step_label": "遮罩棋盤 (Board Mask Applied)",
                "success": True,
                "image_base64": _encode_jpeg_base64(masked_img),
                "metadata": {},
                "error": None,
                "duration_ms": 0,
            })

        # Step 3: Piece detection (YOLO boxes)
        det_boxes = debug_data.get("det_boxes")
        det_confs = debug_data.get("det_confs")
        if det_boxes is not None:
            vis_det = draw_detection_boxes(masked_img if masked_img is not None else color_img, det_boxes, det_confs)
            cv2.imwrite(f"/app/captures/debug_{ts}_3_detection.png", vis_det)
            det_count = len(det_boxes) if det_boxes is not None else 0
            avg_conf = float(det_confs.mean()) if det_count > 0 else 0
            steps.append({
                "step_name": "detection",
                "step_label": "棋子偵測 (Piece Detection)",
                "success": det_count > 0,
                "image_base64": _encode_jpeg_base64(vis_det),
                "metadata": {
                    "detected_count": det_count,
                    "avg_det_conf": round(avg_conf, 3),
                },
                "error": None if det_count > 0 else "No pieces detected",
                "duration_ms": round(debug_data.get("det_duration_ms", 0), 1),
            })
        elif debug_data.get("failed_at") not in ("segmentation", "grid_extraction"):
            steps.append({
                "step_name": "detection",
                "step_label": "棋子偵測 (Piece Detection)",
                "success": False,
                "image_base64": None,
                "metadata": {},
                "error": "Skipped (previous step failed)",
                "duration_ms": None,
            })

        # Step 4: Color classification (white/black detection)
        pieces = debug_data.get("pieces")
        if pieces:
            vis_cls = draw_classification_results(masked_img if masked_img is not None else color_img, pieces)
            cv2.imwrite(f"/app/captures/debug_{ts}_4_color.png", vis_cls)
            white_count = sum(1 for p in pieces if p["cls_name"].startswith("white"))
            black_count = sum(1 for p in pieces if p["cls_name"].startswith("black"))
            steps.append({
                "step_name": "color_detection",
                "step_label": "黑白辨識 (Color Detection)",
                "success": True,
                "image_base64": _encode_jpeg_base64(vis_cls),
                "metadata": {
                    "white_count": white_count,
                    "black_count": black_count,
                    "total": len(pieces),
                },
                "error": None,
                "duration_ms": None,
            })

        # Step 5: Final result
        board_state = debug_data.get("board_state")
        board_result = debug_data.get("board_result")
        if board_state is not None and grid_info is not None:
            fen = board_result.get_fen() if board_result else ""
            vis_final = draw_final_board(color_img, board_state, grid_info["grid_orig"], fen)

            # Build warp-space debug image: warped board + grid + projected piece points
            from chess_vision.board_state import get_piece_bottom_center, find_square_for_point_warp
            from chess_vision.chess_algo import WARP_SIZE

            H_mat = grid_info["H"]
            vis_warp = grid_info["warped"].copy()
            warp_sz = WARP_SIZE
            cell = warp_sz / 8.0

            # Draw uniform grid lines on warped image
            for i in range(9):
                x = int(i * cell)
                y = int(i * cell)
                cv2.line(vis_warp, (x, 0), (x, warp_sz - 1), (200, 200, 0), 1)
                cv2.line(vis_warp, (0, y), (warp_sz - 1, y), (0, 200, 200), 1)

            # Project each piece center into warp space and draw
            mapped_count = 0
            unmapped_count = 0
            sample_warp_pts = []
            if pieces:
                from chess_vision.board_state import get_piece_center
                for i, p in enumerate(pieces):
                    bbox = p["bbox"]
                    # Use center (not bottom-center) — more robust with angled camera
                    cx, cy = get_piece_center(bbox)

                    # Project to warp space via H
                    pt = np.array([[[cx, cy]]], dtype=np.float32)
                    warped_pt = cv2.perspectiveTransform(pt, H_mat)[0, 0]
                    wx, wy = float(warped_pt[0]), float(warped_pt[1])
                    wxi, wyi = int(round(wx)), int(round(wy))

                    in_bounds = 0 <= wx < warp_sz and 0 <= wy < warp_sz
                    sq = None
                    if in_bounds:
                        col_idx = min(int(wx / cell), 7)
                        row_idx = min(int(wy / cell), 7)
                        file_idx = col_idx
                        rank_idx = 7 - row_idx
                        sq = "abcdefgh"[file_idx] + str(rank_idx + 1)

                    if i < 5:
                        sample_warp_pts.append({
                            "cls": p["cls_name"],
                            "orig": [round(cx, 1), round(cy, 1)],
                            "warp": [round(wx, 1), round(wy, 1)],
                            "in_bounds": in_bounds,
                            "sq": sq,
                        })

                    if sq:
                        cv2.circle(vis_warp, (wxi, wyi), 5, (0, 255, 0), -1)
                        cv2.circle(vis_warp, (wxi, wyi), 5, (0, 0, 0), 1)
                        cv2.putText(vis_warp, sq, (wxi + 6, wyi - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
                        mapped_count += 1
                    else:
                        # Draw even if out of bounds (clamp to image for visibility)
                        dx = max(0, min(wxi, warp_sz - 1))
                        dy = max(0, min(wyi, warp_sz - 1))
                        cv2.circle(vis_warp, (dx, dy), 5, (0, 0, 255), -1)
                        cv2.circle(vis_warp, (dx, dy), 5, (0, 0, 0), 1)
                        unmapped_count += 1

            cv2.putText(vis_warp, f"Mapped: {mapped_count}  Unmapped: {unmapped_count}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            # Save warp-space debug image
            debug_path = f"/app/captures/debug_{ts}_5_warp.png"
            cv2.imwrite(debug_path, vis_warp)

            debug_coords = {
                "warp_size": warp_sz,
                "mapped_count": mapped_count,
                "unmapped_count": unmapped_count,
                "debug_image": debug_path,
                "sample_warp_pts": sample_warp_pts,
            }

            # Step 5b: Warp-space debug view
            steps.append({
                "step_name": "warp_debug",
                "step_label": "投影偵錯 (Warp Debug)",
                "success": mapped_count > 0,
                "image_base64": _encode_jpeg_base64(vis_warp),
                "metadata": debug_coords,
                "error": None if mapped_count > 0 else "No pieces mapped in warp space",
                "duration_ms": None,
            })

            # Draw quad corners and piece points with orig+warp coords on final image
            quad = grid_info["quad"]  # (4,2) TL,TR,BR,BL in original space
            dst_pts = np.array([[0,0],[warp_sz-1,0],[warp_sz-1,warp_sz-1],[0,warp_sz-1]], dtype=np.float32)
            corner_labels = ["TL","TR","BR","BL"]
            quad_info = []
            for ci, (label, orig_pt, warp_pt) in enumerate(zip(corner_labels, quad, dst_pts)):
                ox, oy = int(round(orig_pt[0])), int(round(orig_pt[1]))
                wx_c, wy_c = int(round(warp_pt[0])), int(round(warp_pt[1]))
                # Draw corner on final image
                cv2.circle(vis_final, (ox, oy), 8, (0, 255, 255), 2)
                txt = f"{label} o({ox},{oy}) w({wx_c},{wy_c})"
                cv2.putText(vis_final, txt, (ox + 10, oy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                quad_info.append({"label": label, "orig": [round(float(orig_pt[0]),1), round(float(orig_pt[1]),1)],
                                  "warp": [round(float(warp_pt[0]),1), round(float(warp_pt[1]),1)]})

            # Draw piece center points with orig+warp coords
            piece_coord_info = []
            if pieces:
                from chess_vision.board_state import get_piece_center as _get_center
                for i, p in enumerate(pieces):
                    bbox = p["bbox"]
                    cx, cy = _get_center(bbox)
                    pt_arr = np.array([[[cx, cy]]], dtype=np.float32)
                    wpt = cv2.perspectiveTransform(pt_arr, H_mat)[0, 0]
                    wx_p, wy_p = float(wpt[0]), float(wpt[1])
                    oxi, oyi = int(round(cx)), int(round(cy))

                    # Draw on final image
                    cv2.circle(vis_final, (oxi, oyi), 4, (255, 0, 255), -1)
                    if i < 10:
                        txt = f"o({oxi},{oyi})w({int(round(wx_p))},{int(round(wy_p))})"
                        cv2.putText(vis_final, txt, (oxi + 5, oyi + 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1, cv2.LINE_AA)
                    if i < 10:
                        piece_coord_info.append({
                            "cls": p["cls_name"],
                            "orig": [round(cx,1), round(cy,1)],
                            "warp": [round(wx_p,1), round(wy_p,1)],
                        })

            # Save annotated final image
            final_debug_path = f"/app/captures/debug_{ts}_6_final.png"
            cv2.imwrite(final_debug_path, vis_final)

            steps.append({
                "step_name": "result",
                "step_label": "最後結果 (Final Result)",
                "success": True,
                "image_base64": _encode_jpeg_base64(vis_final),
                "metadata": {
                    "fen": fen,
                    "piece_count": len(board_state),
                    "is_valid": board_result.analysis["is_valid"] if board_result else False,
                    "warnings": board_result.analysis.get("warnings", []) if board_result else [],
                    "occupied_squares": sorted(board_state.keys()),
                    "quad_corners": quad_info,
                    "piece_coords": piece_coord_info,
                },
                "error": None,
                "duration_ms": round(debug_data.get("map_duration_ms", 0), 1),
            })
        elif debug_data.get("failed_at") is None:
            steps.append({
                "step_name": "result",
                "step_label": "最後結果 (Final Result)",
                "success": False,
                "image_base64": None,
                "metadata": {},
                "error": "No board state generated",
                "duration_ms": None,
            })

        total_ms = (time.time() - t_start) * 1000

        # Include piece_positions at top level so frontend can render the board
        piece_positions = {}
        if board_state is not None:
            piece_positions = dict(board_state)

        return {
            "success": debug_data.get("failed_at") is None,
            "failed_at": debug_data.get("failed_at"),
            "steps": steps,
            "piece_positions": piece_positions,
            "total_duration_ms": round(total_ms, 1),
            "timestamp": time.time(),
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "steps": [],
            "total_duration_ms": 0,
            "timestamp": time.time(),
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
