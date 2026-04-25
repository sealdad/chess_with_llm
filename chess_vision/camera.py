# chess_vision/camera.py
"""
Camera abstraction for chess vision system.

BaseCamera defines the interface any depth camera must implement.
RealSenseCamera is the reference implementation for Intel RealSense D4xx.
"""
import os
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, NamedTuple

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Camera abstraction types
# ---------------------------------------------------------------------------

class CameraIntrinsics(NamedTuple):
    """Pinhole camera intrinsic parameters.

    Attribute names match RealSense conventions so existing code that
    accesses ``intrinsics.fx``, ``intrinsics.ppx``, etc. works unchanged.
    """
    fx: float       # focal length x (pixels)
    fy: float       # focal length y (pixels)
    ppx: float      # principal point x (pixels)
    ppy: float      # principal point y (pixels)
    width: int      # image width
    height: int     # image height
    coeffs: tuple = ()   # distortion coefficients
    model: str = "pinhole"  # distortion model name


def deproject_pixel_to_point(
    intrinsics: CameraIntrinsics,
    pixel: list,
    depth: float,
) -> list:
    """Deproject a 2D pixel + depth to a 3D camera-frame point.

    Pure pinhole math — drop-in replacement for
    ``pyrealsense2.rs2_deproject_pixel_to_point()``.
    """
    x = (pixel[0] - intrinsics.ppx) * depth / intrinsics.fx
    y = (pixel[1] - intrinsics.ppy) * depth / intrinsics.fy
    return [x, y, depth]


class BaseCamera(ABC):
    """Abstract camera interface for the chess vision system.

    Any depth camera (RealSense, Kinect, ZED, OAK-D, ...) can be used by
    implementing these four methods.  The rest of the vision pipeline
    operates on numpy arrays returned here.
    """

    @abstractmethod
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture aligned color + depth images.

        Returns:
            (color_bgr, depth_raw) where
            - color_bgr: (H, W, 3) uint8 BGR numpy array, or None
            - depth_raw: (H, W) uint16 raw depth array, or None
        """
        ...

    @abstractmethod
    def get_intrinsics(self) -> CameraIntrinsics:
        """Return color camera intrinsic parameters."""
        ...

    @abstractmethod
    def get_depth_scale(self) -> float:
        """Return depth scale: meters per raw depth unit."""
        ...

    @abstractmethod
    def release(self) -> None:
        """Release camera resources."""
        ...

    # ---- convenience (concrete) ------------------------------------------

    def get_depth_meters(self, depth_raw: np.ndarray, x: int, y: int) -> float:
        """Read depth at pixel (x, y) in meters from a raw depth image."""
        return float(depth_raw[y, x]) * self.get_depth_scale()

    def pixel_to_3d(self, pixel_x: int, pixel_y: int,
                    depth_m: float) -> Optional[Tuple[float, float, float]]:
        """Deproject pixel + depth (meters) to 3D camera-frame point."""
        if depth_m <= 0:
            return None
        pt = deproject_pixel_to_point(self.get_intrinsics(),
                                      [pixel_x, pixel_y], depth_m)
        return (pt[0], pt[1], pt[2])


# ---------------------------------------------------------------------------
# Intel RealSense implementation
# ---------------------------------------------------------------------------

class RealSenseCamera(BaseCamera):
    """Intel RealSense D4xx depth camera (reference implementation).

    Wraps ``pyrealsense2`` to satisfy the :class:`BaseCamera` interface.
    Also exposes the raw RealSense frame objects for callers that need them
    via :meth:`get_frame_raw`.
    """

    def __init__(self,
                 depth_res=(640, 480, 30),
                 color_res=(640, 480, 30),
                 enable_auto_exposure=True,
                 virtual=False):
        import pyrealsense2 as rs
        self._rs = rs
        self.virtual = virtual

        if self.virtual:
            self._pipeline = None
            self.depth_intrinsics = None
            self.color_intrinsics = None
            self.depth_scale = None
            return

        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, depth_res[0], depth_res[1],
                          rs.format.z16, depth_res[2])
        cfg.enable_stream(rs.stream.color, color_res[0], color_res[1],
                          rs.format.bgr8, color_res[2])

        profile = self._pipeline.start(cfg)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        try:
            depth_sensor.set_option(rs.option.enable_auto_exposure,
                                    1 if enable_auto_exposure else 0)
        except Exception:
            pass

        self._align = rs.align(rs.stream.color)
        self._profile = profile

        # Store intrinsics as CameraIntrinsics (used by BaseCamera helpers)
        di = profile.get_stream(rs.stream.depth) \
                     .as_video_stream_profile().get_intrinsics()
        ci = profile.get_stream(rs.stream.color) \
                     .as_video_stream_profile().get_intrinsics()

        self.depth_intrinsics = CameraIntrinsics(
            fx=ci.fx, fy=ci.fy, ppx=ci.ppx, ppy=ci.ppy,
            width=ci.width, height=ci.height,
            coeffs=tuple(ci.coeffs), model=str(ci.model),
        )
        self.color_intrinsics = self.depth_intrinsics  # aligned to color

        # Keep raw RS intrinsics for callers that need them
        self._rs_color_intrinsics = ci
        self._rs_depth_intrinsics = di

    # ---- BaseCamera interface --------------------------------------------

    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.virtual:
            return (None, None)
        cf, df, ci, di = self.get_frame_raw()
        return (ci, di)

    def get_intrinsics(self) -> CameraIntrinsics:
        return self.color_intrinsics

    def get_depth_scale(self) -> float:
        return self.depth_scale

    def release(self) -> None:
        if not self.virtual and self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass

    # ---- RealSense-specific (extra) --------------------------------------

    def get_frame_raw(self):
        """Return (color_frame, depth_frame, color_img, depth_img).

        The rs2 frame objects are needed by legacy callers.
        """
        if self.virtual:
            raise RuntimeError("[Virtual mode] get_frame_raw() not available.")

        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return (None, None, None, None)

        depth_img = np.asanyarray(depth_frame.get_data())   # uint16 z16
        color_img = np.asanyarray(color_frame.get_data())   # bgr8

        return (color_frame, depth_frame, color_img, depth_img)

    # ---- interactive utilities (RealSense-specific) ----------------------

    def _draw_overlay(self, img, fps=None, msg=None):
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.line(img, (cx - 12, cy), (cx + 12, cy), (0, 255, 0), 1)
        cv2.line(img, (cx, cy - 12), (cx, cy + 12), (0, 255, 0), 1)

        if msg is None:
            msg = "Press [C]apture  [S]ave both Color+Depth  [Q]/[ESC] exit"
        cv2.putText(img, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2, cv2.LINE_AA)

        if fps is not None:
            cv2.putText(img, f"FPS: {fps:.1f}", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)

    def photo_mode(self,
                   save_dir="captures",
                   save_depth=True,
                   save_npy=True,
                   window_name="RealSense Preview"):
        """Interactive photo capture mode."""
        if self.virtual:
            raise RuntimeError("[Virtual mode] photo_mode() not available.")

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        prev_t = time.time()
        fps = None
        counter = 0

        try:
            while True:
                out = self.get_frame_raw()
                if out[0] is None:
                    continue
                color_frame, depth_frame, color_img, depth_img = out

                now = time.time()
                dt = now - prev_t
                counter += 1
                if dt >= 0.5:
                    fps = counter / dt
                    counter = 0
                    prev_t = now

                preview = color_img.copy()
                self._draw_overlay(preview, fps=fps)

                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord('q'), 27):
                    break

                if key in (ord('c'), ord('s')):
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    color_path = os.path.join(save_dir, f"{ts}_color.jpg")
                    cv2.imwrite(color_path, color_img,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    saved = {"color": color_path}

                    if key == ord('s'):
                        if save_depth:
                            depth_png_path = os.path.join(save_dir,
                                                          f"{ts}_depth.png")
                            cv2.imwrite(depth_png_path, depth_img)
                            saved["depth_png"] = depth_png_path
                        if save_npy:
                            depth_npy_path = os.path.join(save_dir,
                                                          f"{ts}_depth.npy")
                            np.save(depth_npy_path, depth_img)
                            saved["depth_npy"] = depth_npy_path

                    intr = self.color_intrinsics
                    meta = {
                        "timestamp": ts,
                        "depth_scale_m_per_unit": self.depth_scale,
                        "color_intrinsics": {
                            "width": intr.width, "height": intr.height,
                            "ppx": intr.ppx, "ppy": intr.ppy,
                            "fx": intr.fx, "fy": intr.fy,
                            "model": intr.model,
                            "coeffs": list(intr.coeffs),
                        },
                        "saved_files": saved,
                    }
                    meta_path = os.path.join(save_dir, f"{ts}_metadata.json")
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)

                    print(f"[Saved] {saved}")
                    print(f"[Meta ] {meta_path}")

        finally:
            cv2.destroyWindow(window_name)

    def chess_stream(self,
                     window_name="RealSense Chess",
                     draw_board_grid=True,
                     do_refine_grid=False):
        """Live YOLO + board grid visualization."""
        if self.virtual:
            raise RuntimeError("[Virtual mode] chess_stream() not available.")

        import torch
        from .chess_algo import (
            DETECT_MODEL, SEGMENT_MODEL,
            load_piece_detector, load_board_segmenter, load_classifier,
            detect_and_classify_pieces, segment_chessboard,
            grid_from_mask_and_image, draw_box_and_label,
            draw_grid_lines, draw_corners,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {device}")

        det_model = load_piece_detector(DETECT_MODEL)
        cls_model, class_names, cls_tf = load_classifier(device)
        seg_model = load_board_segmenter(SEGMENT_MODEL) if draw_board_grid else None

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        prev_t = time.time()
        fps = None
        counter = 0

        try:
            while True:
                color_frame, depth_frame, color_img, depth_img = self.get_frame_raw()
                if color_frame is None:
                    continue

                now = time.time()
                dt = now - prev_t
                counter += 1
                if dt >= 0.5:
                    fps = counter / dt
                    counter = 0
                    prev_t = now

                vis = color_img.copy()

                pieces = detect_and_classify_pieces(
                    det_model=det_model, cls_model=cls_model,
                    cls_tf=cls_tf, class_names=class_names,
                    device=device, color_img_bgr=color_img,
                )
                for p in pieces:
                    x1, y1, x2, y2 = p["bbox"]
                    draw_box_and_label(vis, x1, y1, x2, y2, p["cls_name"])

                if draw_board_grid and seg_model is not None:
                    mask = segment_chessboard(seg_model, color_img)
                    info = grid_from_mask_and_image(
                        img_bgr=color_img, mask_01=mask,
                        do_refine=do_refine_grid,
                    )
                    if info is not None:
                        draw_grid_lines(vis, info["grid_orig"], thickness=2)
                        draw_corners(vis, info["grid_orig"], radius=3)

                msg = "Chess mode  |  Press [Q]/[ESC] to exit"
                self._draw_overlay(vis, fps=fps, msg=msg)

                cv2.imshow(window_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break

        finally:
            cv2.destroyWindow(window_name)


if __name__ == "__main__":
    cam = RealSenseCamera()
    try:
        cam.chess_stream(window_name="RealSense Chess",
                         draw_board_grid=True, do_refine_grid=False)
    finally:
        cam.release()
        cv2.destroyAllWindows()
