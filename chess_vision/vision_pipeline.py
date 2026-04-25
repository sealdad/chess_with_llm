# chess_vision/vision_pipeline.py
"""
Unified vision pipeline for chess board analysis.

This module provides a complete pipeline:
  Image -> Board Segmentation -> Grid Extraction -> Piece Detection -> Board State -> LLM Output

Supports both live camera input and offline static image analysis.
"""

import cv2
import numpy as np
import torch
from typing import Optional, Dict, Tuple, Union
from pathlib import Path

from .chess_algo import (
    load_piece_detector,
    load_board_segmenter,
    load_classifier,
    detect_pieces,
    detect_and_classify_pieces,
    segment_chessboard,
    mask_board_region,
    grid_from_mask_and_image,
    draw_box_and_label,
    draw_grid_lines,
    draw_corners,
    draw_segmentation_overlay,
    draw_detection_boxes,
    draw_classification_results,
    draw_final_board,
    DETECT_MODEL,
    SEGMENT_MODEL,
    DETECT_IMSZ,
    DETECT_CONF,
    WARP_SIZE,
)

from .board_state import (
    map_pieces_to_squares,
    BoardStateResult,
    FEN_PIECE_MAP,
)


class ChessVisionPipeline:
    """
    Complete chess vision pipeline for board state recognition.

    Usage:
        # Initialize once
        pipeline = ChessVisionPipeline()

        # Analyze an image
        result = pipeline.analyze_image("path/to/image.jpg")

        # Get outputs
        print(result.get_fen())
        print(result.get_llm_description())
        print(result.get_ascii_board())
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        detect_model_path: str = DETECT_MODEL,
        segment_model_path: str = SEGMENT_MODEL,
        lazy_load: bool = False,
    ):
        """
        Initialize the pipeline.

        Args:
            device: torch device (auto-detect if None)
            detect_model_path: path to YOLO detection model
            segment_model_path: path to YOLO segmentation model
            lazy_load: if True, defer model loading until first use
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.detect_model_path = detect_model_path
        self.segment_model_path = segment_model_path

        self._det_model = None
        self._seg_model = None
        self._cls_model = None
        self._cls_tf = None
        self._class_names = None

        if not lazy_load:
            self._load_models()

    def _load_models(self):
        """Load detection and segmentation models (classifier not needed)."""
        print(f"[ChessVisionPipeline] Loading models on {self.device}...")

        self._det_model = load_piece_detector(self.detect_model_path)
        self._seg_model = load_board_segmenter(self.segment_model_path)

        print("[ChessVisionPipeline] Models loaded successfully.")

    def _load_classifier(self):
        """Load classifier on demand (only needed for debug pipeline)."""
        if self._cls_model is None:
            print("[ChessVisionPipeline] Loading classifier...")
            self._cls_model, self._class_names, self._cls_tf = load_classifier(self.device)
            print("[ChessVisionPipeline] Classifier loaded.")

    def _ensure_models_loaded(self):
        """Ensure models are loaded (for lazy loading)."""
        if self._det_model is None:
            self._load_models()

    def analyze_image(
        self,
        image: Union[str, Path, np.ndarray],
        board_orientation: str = "auto",
        do_refine_grid: bool = False,
        use_bottom_center: bool = False,
    ) -> Optional[BoardStateResult]:
        """
        Analyze a chess board image and return complete board state.

        Args:
            image: file path or BGR numpy array
            board_orientation: "white_bottom", "black_bottom", or "auto"
            do_refine_grid: whether to refine grid corners with Harris detector
            use_bottom_center: use bottom-center of piece bbox for square mapping (default: center)

        Returns:
            BoardStateResult object with complete analysis, or None if failed
        """
        self._ensure_models_loaded()

        # Load image if path provided
        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                print(f"[Error] Could not load image: {image}")
                return None
        else:
            img_bgr = image

        # Step 1: Segment the chessboard
        mask = segment_chessboard(self._seg_model, img_bgr)
        if mask is None:
            print("[Warning] No chessboard detected in image")
            return None

        # Step 2: Extract grid from mask
        grid_info = grid_from_mask_and_image(
            img_bgr=img_bgr,
            mask_01=mask,
            warp_size=WARP_SIZE,
            do_refine=do_refine_grid,
        )
        if grid_info is None:
            print("[Warning] Could not extract grid from board mask")
            return None

        grid_9x9 = grid_info["grid_orig"]

        # Step 3: Detect pieces on full image (masking can clip edge pieces)
        pieces = detect_pieces(
            det_model=self._det_model,
            color_img_bgr=img_bgr,
        )

        # Step 4: Map pieces to squares (in warp space for accuracy)
        board_state = map_pieces_to_squares(
            pieces=pieces,
            grid_9x9=grid_9x9,
            use_bottom_center=use_bottom_center,
            board_orientation=board_orientation,
            H=grid_info["H"],
            warp_size=WARP_SIZE,
        )

        # Step 5: Create result object
        result = BoardStateResult(
            board_state=board_state,
            grid_9x9=grid_9x9,
            image_shape=img_bgr.shape,
            orientation=board_orientation if board_orientation != "auto" else "white_bottom",
        )

        return result

    def analyze_image_debug(
        self,
        image: Union[str, Path, np.ndarray],
        board_orientation: str = "auto",
        do_refine_grid: bool = False,
        use_bottom_center: bool = False,
    ) -> dict:
        """
        Run the full pipeline and return intermediate results for each step.

        Returns dict with keys:
            raw_image, seg_mask, seg_duration_ms, grid_info, grid_duration_ms,
            det_boxes, det_confs, det_duration_ms, pieces, cls_duration_ms,
            board_state, board_result, map_duration_ms, failed_at
        """
        import time as _time
        self._ensure_models_loaded()

        # Load image
        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                return {"failed_at": "load_image", "raw_image": None}
        else:
            img_bgr = image

        result = {"raw_image": img_bgr, "failed_at": None}

        # Step 1: Segmentation
        t0 = _time.time()
        mask = segment_chessboard(self._seg_model, img_bgr)
        result["seg_mask"] = mask
        result["seg_duration_ms"] = (_time.time() - t0) * 1000
        if mask is None:
            result["failed_at"] = "segmentation"
            return result

        # Step 2: Grid extraction
        t0 = _time.time()
        grid_info = grid_from_mask_and_image(
            img_bgr=img_bgr,
            mask_01=mask,
            warp_size=WARP_SIZE,
            do_refine=do_refine_grid,
        )
        result["grid_info"] = grid_info
        result["grid_duration_ms"] = (_time.time() - t0) * 1000
        if grid_info is None:
            result["failed_at"] = "grid_extraction"
            return result

        # Step 2.5: Keep masked image for debug visualization only
        img_masked = mask_board_region(img_bgr, mask)
        result["masked_image"] = img_masked

        # Step 3: Detection on full image (masking can clip edge pieces)
        t0 = _time.time()
        pieces = detect_pieces(
            det_model=self._det_model,
            color_img_bgr=img_bgr,
        )
        result["pieces"] = pieces
        # Also extract raw boxes/confs for debug visualization
        if pieces:
            result["det_boxes"] = np.array([p["bbox"] for p in pieces], dtype=np.float32)
            result["det_confs"] = np.array([p["det_conf"] for p in pieces], dtype=np.float32)
        else:
            result["det_boxes"] = np.array([])
            result["det_confs"] = np.array([])
        result["det_duration_ms"] = (_time.time() - t0) * 1000

        # Step 5: Map to squares + final result
        t0 = _time.time()
        if board_orientation == "auto":
            from .board_state import detect_board_orientation
            board_orientation = detect_board_orientation(grid_info["grid_orig"])

        board_state = map_pieces_to_squares(
            pieces=pieces,
            grid_9x9=grid_info["grid_orig"],
            use_bottom_center=use_bottom_center,
            board_orientation=board_orientation,
            H=grid_info["H"],
            warp_size=WARP_SIZE,
        )
        board_result = BoardStateResult(
            board_state=board_state,
            grid_9x9=grid_info["grid_orig"],
            image_shape=img_bgr.shape,
            orientation=board_orientation,
        )
        result["board_state"] = board_state
        result["board_result"] = board_result
        result["map_duration_ms"] = (_time.time() - t0) * 1000

        return result

    def analyze_and_visualize(
        self,
        image: Union[str, Path, np.ndarray],
        output_path: Optional[str] = None,
        board_orientation: str = "auto",
        do_refine_grid: bool = False,
    ) -> Tuple[Optional[BoardStateResult], Optional[np.ndarray]]:
        """
        Analyze image and create visualization.

        Returns:
            (BoardStateResult, visualization_image) or (None, None) if failed
        """
        self._ensure_models_loaded()

        # Load image
        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                return None, None
        else:
            img_bgr = image.copy()

        # Get result
        result = self.analyze_image(
            image=img_bgr,
            board_orientation=board_orientation,
            do_refine_grid=do_refine_grid,
        )

        if result is None:
            return None, None

        # Create visualization
        vis = img_bgr.copy()

        # Draw grid
        draw_grid_lines(vis, result.grid_9x9, thickness=2)
        draw_corners(vis, result.grid_9x9, radius=4)

        # Draw pieces with labels
        for square, info in result.board_state.items():
            bbox = info["bbox"]
            draw_box_and_label(vis, bbox[0], bbox[1], bbox[2], bbox[3], info["piece"])

            # Also label the square
            cx = (bbox[0] + bbox[2]) // 2
            cy = bbox[3] + 15
            cv2.putText(
                vis, square, (cx - 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA
            )

        # Save if path provided
        if output_path:
            cv2.imwrite(output_path, vis)
            print(f"[Saved] Visualization to {output_path}")

        return result, vis

    def get_llm_prompt(
        self,
        result: BoardStateResult,
        context: str = "chess_game",
        include_fen: bool = True,
        include_ascii: bool = True,
    ) -> str:
        """
        Generate a complete prompt for LLM with board state information.

        Args:
            result: BoardStateResult from analyze_image
            context: "chess_game" or "instruction" or custom context string
            include_fen: include FEN notation
            include_ascii: include ASCII board view

        Returns:
            Formatted string ready for LLM input
        """
        lines = []

        # Header based on context
        if context == "chess_game":
            lines.append("The current chess board position is as follows:")
        elif context == "instruction":
            lines.append("Please analyze the following chess position:")
        else:
            lines.append(context)

        lines.append("")

        # FEN notation
        if include_fen:
            lines.append(f"FEN: {result.get_fen()}")
            lines.append("")

        # ASCII board
        if include_ascii:
            lines.append("Board visualization:")
            lines.append(result.get_ascii_board())
            lines.append("")

        # Piece list
        lines.append(result.get_llm_description())

        # Validity warnings
        if not result.analysis["is_valid"]:
            lines.append("")
            lines.append("WARNINGS:")
            for warning in result.analysis["warnings"]:
                lines.append(f"  - {warning}")

        return "\n".join(lines)


def analyze_image_file(
    image_path: str,
    output_path: Optional[str] = None,
    print_result: bool = True,
) -> Optional[BoardStateResult]:
    """
    Convenience function to analyze a single image file.

    Args:
        image_path: path to image file
        output_path: optional path for visualization output
        print_result: whether to print results to console

    Returns:
        BoardStateResult or None
    """
    pipeline = ChessVisionPipeline()

    if output_path:
        result, vis = pipeline.analyze_and_visualize(image_path, output_path)
    else:
        result = pipeline.analyze_image(image_path)

    if result is None:
        print(f"[Error] Failed to analyze {image_path}")
        return None

    if print_result:
        print("\n" + "=" * 60)
        print(f"Analysis of: {image_path}")
        print("=" * 60)
        print(f"\nFEN: {result.get_fen()}")
        print(f"\n{result.get_ascii_board()}")
        print(f"\n{result.get_llm_description()}")

        if not result.analysis["is_valid"]:
            print("\nWarnings:")
            for w in result.analysis["warnings"]:
                print(f"  - {w}")

    return result


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chess board vision analysis")
    parser.add_argument("image", help="Path to chess board image")
    parser.add_argument("-o", "--output", help="Output visualization path")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    result = analyze_image_file(
        image_path=args.image,
        output_path=args.output,
        print_result=not args.quiet,
    )
