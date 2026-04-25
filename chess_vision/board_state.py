# chess_vision/board_state.py
"""
Board state analysis module:
- Map detected pieces to board squares (a1-h8)
- Generate FEN notation
- Create LLM-friendly board state descriptions
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

# Standard chess piece notation for FEN
FEN_PIECE_MAP = {
    "white_king": "K",
    "white_queen": "Q",
    "white_rook": "R",
    "white_bishop": "B",
    "white_knight": "N",
    "white_pawn": "P",
    "black_king": "k",
    "black_queen": "q",
    "black_rook": "r",
    "black_bishop": "b",
    "black_knight": "n",
    "black_pawn": "p",
    "piece": "X",  # Generic detected piece (no classification)
}

# Full piece names for LLM descriptions
PIECE_FULL_NAME = {
    "white_king": "White King",
    "white_queen": "White Queen",
    "white_rook": "White Rook",
    "white_bishop": "White Bishop",
    "white_knight": "White Knight",
    "white_pawn": "White Pawn",
    "black_king": "Black King",
    "black_queen": "Black Queen",
    "black_rook": "Black Rook",
    "black_bishop": "Black Bishop",
    "black_knight": "Black Knight",
    "black_pawn": "Black Pawn",
    "piece": "Piece",
}

# Files and ranks
FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
RANKS = ['1', '2', '3', '4', '5', '6', '7', '8']


def get_piece_center(bbox: List[int]) -> Tuple[float, float]:
    """Get anchor point of a bounding box [x1, y1, x2, y2].

    Uses horizontal center, 3/4 down from top (1/4 up from bottom).
    This is closer to the piece base, giving more accurate square mapping
    with an angled camera where tall pieces lean away from their base.
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, y1 + (y2 - y1) * 0.75)


def get_piece_bottom_center(bbox: List[int]) -> Tuple[float, float]:
    """
    Get bottom-center of bounding box - more accurate for piece placement
    since pieces "stand" on squares.
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, y2)


def point_in_quad(px: float, py: float, quad: np.ndarray) -> bool:
    """
    Check if point (px, py) is inside a quadrilateral.
    quad: 4x2 array of corners in order (TL, TR, BR, BL) or any consistent order.
    Uses cross product method — works for both CW and CCW winding.
    """
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    p = np.array([px, py])
    n = len(quad)

    signs = []
    for i in range(n):
        o = quad[i]
        a = quad[(i + 1) % n]
        signs.append(cross(o, a, p))

    # All same sign (or zero) means point is inside, regardless of winding
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)


def find_square_for_point_warp(
    px: float,
    py: float,
    H: np.ndarray,
    warp_size: int,
    board_orientation: str = "auto"
) -> Optional[str]:
    """
    Project point (px, py) from original image into warp space via H,
    then do a simple grid cell lookup in the uniform warp grid.

    Returns: square name like "e4" or None if point is outside board
    """
    pt = np.array([[[px, py]]], dtype=np.float32)
    warped_pt = cv2.perspectiveTransform(pt, H)[0, 0]
    wx, wy = warped_pt

    # Check bounds
    if wx < 0 or wx >= warp_size or wy < 0 or wy >= warp_size:
        return None

    # Cell size in warp space (uniform grid)
    cell = warp_size / 8.0
    col = int(wx / cell)
    row = int(wy / cell)

    col = min(col, 7)
    row = min(row, 7)

    if board_orientation == "white_bottom":
        file_idx = col
        rank_idx = 7 - row
    elif board_orientation == "black_bottom":
        file_idx = 7 - col
        rank_idx = row
    else:
        file_idx = col
        rank_idx = 7 - row

    return FILES[file_idx] + RANKS[rank_idx]


def find_square_for_point(
    px: float,
    py: float,
    grid_9x9: np.ndarray,
    board_orientation: str = "auto"
) -> Optional[str]:
    """
    Fallback: find which chess square contains the point using point-in-quad
    test in original image space.
    """
    for row in range(8):
        for col in range(8):
            tl = grid_9x9[row, col]
            tr = grid_9x9[row, col + 1]
            br = grid_9x9[row + 1, col + 1]
            bl = grid_9x9[row + 1, col]

            quad = np.array([tl, tr, br, bl])

            if point_in_quad(px, py, quad):
                if board_orientation == "white_bottom":
                    file_idx = col
                    rank_idx = 7 - row
                elif board_orientation == "black_bottom":
                    file_idx = 7 - col
                    rank_idx = row
                else:
                    file_idx = col
                    rank_idx = 7 - row

                square = FILES[file_idx] + RANKS[rank_idx]
                return square

    return None


def detect_board_orientation(grid_9x9: np.ndarray) -> str:
    """
    Detect board orientation based on grid geometry.

    Standard setup: looking from white's side, a1 is bottom-left.
    From camera above/behind white: a1 appears at bottom-left of image.

    Returns: "white_bottom" or "black_bottom"
    """
    # Get corners of the board
    tl = grid_9x9[0, 0]      # top-left in image
    tr = grid_9x9[0, 8]      # top-right
    bl = grid_9x9[8, 0]      # bottom-left
    br = grid_9x9[8, 8]      # bottom-right

    # For now, assume standard orientation (white at bottom)
    # In future, could use piece positions to detect orientation
    return "white_bottom"


def map_pieces_to_squares(
    pieces: List[Dict],
    grid_9x9: np.ndarray,
    use_bottom_center: bool = False,
    board_orientation: str = "auto",
    H: Optional[np.ndarray] = None,
    warp_size: Optional[int] = None,
) -> Dict[str, Dict]:
    """
    Map detected pieces to board squares.

    Args:
        pieces: list of dicts with keys: bbox, cls_name, cls_conf, det_conf
        grid_9x9: shape (9, 9, 2) intersection points
        use_bottom_center: if True, use bottom-center of bbox; if False, use center (default)
        board_orientation: "white_bottom", "black_bottom", or "auto"
        H: homography matrix (original -> warp). If provided, maps in warp space.
        warp_size: size of the warped image (required if H is provided)

    Returns:
        Dict mapping square names to piece info
    """
    if board_orientation == "auto":
        board_orientation = detect_board_orientation(grid_9x9)

    use_warp = H is not None and warp_size is not None

    board_state = {}
    unmapped_pieces = []

    for piece in pieces:
        bbox = piece["bbox"]

        if use_bottom_center:
            px, py = get_piece_bottom_center(bbox)
        else:
            px, py = get_piece_center(bbox)

        if use_warp:
            square = find_square_for_point_warp(px, py, H, warp_size, board_orientation)
        else:
            square = find_square_for_point(px, py, grid_9x9, board_orientation)

        if square:
            # Handle multiple pieces detected in same square (keep highest confidence)
            if square in board_state:
                existing_conf = board_state[square]["confidence"]
                new_conf = piece["cls_conf"]
                if new_conf <= existing_conf:
                    continue

            board_state[square] = {
                "piece": piece["cls_name"],
                "confidence": piece["cls_conf"],
                "det_conf": piece["det_conf"],
                "bbox": bbox,
            }
        else:
            unmapped_pieces.append(piece)

    return board_state


def board_state_to_matrix(board_state: Dict[str, Dict]) -> List[List[Optional[str]]]:
    """
    Convert board state dict to 8x8 matrix.

    Returns: 8x8 list where [0][0] is a8, [7][7] is h1
             Each cell is piece class name or None
    """
    matrix = [[None for _ in range(8)] for _ in range(8)]

    for square, info in board_state.items():
        file_idx = FILES.index(square[0])
        rank_idx = int(square[1]) - 1

        # Matrix row 0 = rank 8, row 7 = rank 1
        row = 7 - rank_idx
        col = file_idx

        matrix[row][col] = info["piece"]

    return matrix


def generate_fen(board_state: Dict[str, Dict],
                 active_color: str = "w",
                 castling: str = "-",
                 en_passant: str = "-",
                 halfmove: int = 0,
                 fullmove: int = 1) -> str:
    """
    Generate FEN notation from board state.

    FEN format: piece_placement active_color castling en_passant halfmove fullmove
    Example: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

    Args:
        board_state: dict mapping squares to piece info
        active_color: "w" or "b" (who moves next)
        castling: castling availability (e.g., "KQkq", "Kq", "-")
        en_passant: en passant target square or "-"
        halfmove: halfmove clock (moves since pawn move or capture)
        fullmove: fullmove number

    Returns: FEN string
    """
    matrix = board_state_to_matrix(board_state)

    fen_rows = []
    for row in matrix:
        fen_row = ""
        empty_count = 0

        for cell in row:
            if cell is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += FEN_PIECE_MAP.get(cell, "?")

        if empty_count > 0:
            fen_row += str(empty_count)

        fen_rows.append(fen_row)

    piece_placement = "/".join(fen_rows)

    return f"{piece_placement} {active_color} {castling} {en_passant} {halfmove} {fullmove}"


def generate_llm_board_description(
    board_state: Dict[str, Dict],
    include_confidence: bool = False,
    include_empty_squares: bool = False
) -> str:
    """
    Generate a natural language description of the board state for LLM input.

    Returns a structured text description suitable for chess AI/LLM processing.
    """
    lines = []
    lines.append("=== CHESS BOARD STATE ===")
    lines.append("")

    # Organize by color
    white_pieces = []
    black_pieces = []

    for square, info in sorted(board_state.items()):
        piece = info["piece"]
        full_name = PIECE_FULL_NAME.get(piece, piece)

        if include_confidence:
            conf_str = f" (conf: {info['confidence']:.2f})"
        else:
            conf_str = ""

        entry = f"{full_name} on {square}{conf_str}"

        if piece.startswith("white_"):
            white_pieces.append((square, entry))
        else:
            black_pieces.append((square, entry))

    # White pieces
    lines.append("WHITE PIECES:")
    if white_pieces:
        for square, entry in white_pieces:
            lines.append(f"  - {entry}")
    else:
        lines.append("  (none detected)")

    lines.append("")

    # Black pieces
    lines.append("BLACK PIECES:")
    if black_pieces:
        for square, entry in black_pieces:
            lines.append(f"  - {entry}")
    else:
        lines.append("  (none detected)")

    lines.append("")

    # Summary counts
    piece_counts = {}
    for square, info in board_state.items():
        piece = info["piece"]
        piece_counts[piece] = piece_counts.get(piece, 0) + 1

    lines.append("PIECE COUNT:")
    lines.append(f"  White: {sum(1 for p in piece_counts if p.startswith('white_'))} pieces")
    lines.append(f"  Black: {sum(1 for p in piece_counts if p.startswith('black_'))} pieces")

    return "\n".join(lines)


def generate_compact_board_view(board_state: Dict[str, Dict]) -> str:
    """
    Generate a compact ASCII representation of the board.

    Returns something like:
        a b c d e f g h
      8 r n b q k b n r 8
      7 p p p p p p p p 7
      6 . . . . . . . . 6
      5 . . . . . . . . 5
      4 . . . . . . . . 4
      3 . . . . . . . . 3
      2 P P P P P P P P 2
      1 R N B Q K B N R 1
        a b c d e f g h
    """
    matrix = board_state_to_matrix(board_state)

    lines = []
    lines.append("    a b c d e f g h")
    lines.append("  +-----------------+")

    for row_idx, row in enumerate(matrix):
        rank = 8 - row_idx
        row_chars = []
        for cell in row:
            if cell is None:
                row_chars.append(".")
            else:
                row_chars.append(FEN_PIECE_MAP.get(cell, "?"))

        lines.append(f"  {rank}| {' '.join(row_chars)} |{rank}")

    lines.append("  +-----------------+")
    lines.append("    a b c d e f g h")

    return "\n".join(lines)


def analyze_board_state(board_state: Dict[str, Dict]) -> Dict:
    """
    Analyze the board state and return useful information for game logic.

    Returns dict with:
        - piece_counts: count of each piece type
        - king_positions: location of kings
        - is_valid: basic validity check (both kings present, etc.)
        - warnings: list of potential issues
    """
    analysis = {
        "piece_counts": {},
        "white_king_pos": None,
        "black_king_pos": None,
        "is_valid": True,
        "warnings": [],
    }

    for square, info in board_state.items():
        piece = info["piece"]
        analysis["piece_counts"][piece] = analysis["piece_counts"].get(piece, 0) + 1

        if piece == "white_king":
            if analysis["white_king_pos"] is not None:
                analysis["warnings"].append("Multiple white kings detected")
                analysis["is_valid"] = False
            analysis["white_king_pos"] = square
        elif piece == "black_king":
            if analysis["black_king_pos"] is not None:
                analysis["warnings"].append("Multiple black kings detected")
                analysis["is_valid"] = False
            analysis["black_king_pos"] = square

    # Check for missing kings
    if analysis["white_king_pos"] is None:
        analysis["warnings"].append("White king not detected")
        analysis["is_valid"] = False
    if analysis["black_king_pos"] is None:
        analysis["warnings"].append("Black king not detected")
        analysis["is_valid"] = False

    # Check for too many pieces
    max_counts = {
        "white_pawn": 8, "black_pawn": 8,
        "white_rook": 2, "black_rook": 2,
        "white_knight": 2, "black_knight": 2,
        "white_bishop": 2, "black_bishop": 2,
        "white_queen": 1, "black_queen": 1,
        "white_king": 1, "black_king": 1,
    }

    for piece, count in analysis["piece_counts"].items():
        max_count = max_counts.get(piece, 10)
        # Allow for promoted pieces (pawns can become queens, etc.)
        if piece.endswith("_pawn") and count > max_count:
            analysis["warnings"].append(f"Too many {piece}: {count} (max {max_count})")

    return analysis


class BoardStateResult:
    """
    Container for complete board state analysis results.
    """
    def __init__(
        self,
        board_state: Dict[str, Dict],
        grid_9x9: np.ndarray,
        image_shape: Tuple[int, int, int],
        orientation: str = "white_bottom"
    ):
        self.board_state = board_state
        self.grid_9x9 = grid_9x9
        self.image_shape = image_shape
        self.orientation = orientation
        self.analysis = analyze_board_state(board_state)

    def get_fen(self, **kwargs) -> str:
        """Get FEN notation."""
        return generate_fen(self.board_state, **kwargs)

    def get_llm_description(self, **kwargs) -> str:
        """Get LLM-friendly description."""
        return generate_llm_board_description(self.board_state, **kwargs)

    def get_ascii_board(self) -> str:
        """Get ASCII board representation."""
        return generate_compact_board_view(self.board_state)

    def get_piece_on_square(self, square: str) -> Optional[str]:
        """Get piece on a specific square, or None if empty."""
        if square in self.board_state:
            return self.board_state[square]["piece"]
        return None

    def get_squares_with_piece(self, piece_type: str) -> List[str]:
        """Get all squares containing a specific piece type."""
        return [sq for sq, info in self.board_state.items() if info["piece"] == piece_type]

    def to_dict(self) -> Dict:
        """Export complete state as dictionary."""
        return {
            "board_state": self.board_state,
            "fen": self.get_fen(),
            "orientation": self.orientation,
            "analysis": self.analysis,
            "piece_count": len(self.board_state),
        }

    def __repr__(self):
        return f"BoardStateResult(pieces={len(self.board_state)}, valid={self.analysis['is_valid']})"
