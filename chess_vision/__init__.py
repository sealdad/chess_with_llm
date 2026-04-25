# chess_vision package
"""
Chess Vision System

Modules:
    - camera: Camera abstraction (BaseCamera) and RealSense implementation
    - calibration: Camera and hand-eye calibration tools
    - chess_algo: Chess piece detection and board segmentation
    - board_state: Board state analysis, FEN generation, LLM output
    - vision_pipeline: Unified pipeline for complete board analysis
    - depth_utils: Depth-based piece height measurement (camera-agnostic)
"""

from .chess_algo import (
    load_piece_detector,
    load_board_segmenter,
    load_classifier,
    detect_pieces,
    detect_and_classify_pieces,
    segment_chessboard,
    grid_from_mask_and_image,
)

from .board_state import (
    map_pieces_to_squares,
    generate_fen,
    generate_llm_board_description,
    generate_compact_board_view,
    BoardStateResult,
)

from .vision_pipeline import (
    ChessVisionPipeline,
    analyze_image_file,
)

__version__ = "0.1.0"
