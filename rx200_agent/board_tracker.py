# rx200_agent/board_tracker.py
"""
Authoritative board state tracker using python-chess.

Vision detects occupancy changes (which squares changed), the tracker
knows what pieces are where. This eliminates misclassification errors
and enforces piece constraints (captured pieces can't reappear).
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import chess


class DiscrepancyType(str, Enum):
    """Types of discrepancy between tracker and vision."""
    EXTRA = "extra"                       # Vision sees piece, tracker says empty
    MISSING = "missing"                   # Tracker has piece, vision sees empty
    WRONG = "wrong"                       # Both have piece, but different type
    CAPTURED_REAPPEARED = "captured_reappeared"  # Captured piece shows up again


@dataclass
class Discrepancy:
    """A single discrepancy between tracker state and vision."""
    square: str
    type: DiscrepancyType
    tracker_piece: Optional[str]   # e.g. "white_pawn" or None
    vision_piece: Optional[str]    # e.g. "black_rook" or None
    confidence: float = 1.0


@dataclass
class MoveDetectionResult:
    """Result of human move detection."""
    success: bool
    uci_move: Optional[str] = None
    piece: Optional[str] = None            # e.g. "white_pawn"
    is_capture: bool = False
    captured_piece: Optional[str] = None   # e.g. "black_knight"
    is_castling: bool = False
    is_en_passant: bool = False
    promotion: Optional[str] = None
    discrepancies: List[Discrepancy] = field(default_factory=list)
    candidates: List[str] = field(default_factory=list)  # UCI strings of candidates
    error: Optional[str] = None


def _piece_to_str(piece: chess.Piece) -> str:
    """Convert chess.Piece to 'color_name' string."""
    color = "white" if piece.color == chess.WHITE else "black"
    return f"{color}_{chess.piece_name(piece.piece_type)}"


def _square_name_set(squares: Set[int]) -> Set[str]:
    """Convert set of square indices to set of square name strings."""
    return {chess.square_name(sq) for sq in squares}


class BoardTracker:
    """
    Authoritative board state tracker wrapping chess.Board.

    Maintains the true game state. Vision is used only for occupancy
    change detection, not for piece identification.
    """

    def __init__(self, fen: Optional[str] = None):
        """
        Initialize tracker.

        Args:
            fen: Starting FEN. Defaults to standard starting position.
        """
        if fen is None:
            fen = chess.STARTING_FEN
        self._board = chess.Board(fen)
        self._captured_pieces: List[str] = []  # History of captured pieces

    @property
    def board(self) -> chess.Board:
        return self._board

    @property
    def fen(self) -> str:
        return self._board.fen()

    @property
    def turn(self) -> str:
        """'white' or 'black'."""
        return "white" if self._board.turn == chess.WHITE else "black"

    def get_piece_map(self) -> Dict[str, str]:
        """
        Get current piece positions from tracker.

        Returns:
            {square_name: "color_piecetype"} e.g. {"e1": "white_king", ...}
        """
        result = {}
        for sq, piece in self._board.piece_map().items():
            result[chess.square_name(sq)] = _piece_to_str(piece)
        return result

    def get_occupancy(self) -> Set[str]:
        """Get set of occupied square names."""
        return {chess.square_name(sq) for sq in self._board.piece_map().keys()}

    def validate_initial_position(
        self, vision_pieces: Dict[str, str]
    ) -> Tuple[bool, List[Discrepancy]]:
        """
        Compare vision-detected pieces vs expected starting position.

        Args:
            vision_pieces: {square: "color_piecetype"} from vision

        Returns:
            (is_valid, list_of_discrepancies)
        """
        tracker_pieces = self.get_piece_map()
        discrepancies = self._compare_pieces(tracker_pieces, vision_pieces)
        return len(discrepancies) == 0, discrepancies

    def detect_human_move(
        self,
        prev_occupancy: Set[str],
        curr_occupancy: Set[str],
        vision_pieces: Optional[Dict[str, str]] = None,
    ) -> MoveDetectionResult:
        """
        Detect which legal move the human made.

        Primary method: color-filtered detection — only track the moving
        player's pieces (white or black). This handles all scenarios:
          - Regular move: piece leaves from_sq, appears on to_sq
          - Capture: piece leaves from_sq, appears on to_sq (replaces opponent)
          - Castling: king + rook both move
          - En passant: pawn moves diagonally, opponent pawn disappears
          - Promotion: pawn leaves, promoted piece appears (same color)

        Fallback: occupancy-based detection (ignores color).

        Args:
            prev_occupancy: Occupied squares before the move (from tracker)
            curr_occupancy: Occupied squares now (from vision)
            vision_pieces: Piece color classification from vision
                           (e.g. {"e2": "white_piece", "e7": "black_piece"})

        Returns:
            MoveDetectionResult
        """
        # Step 1: Check for any change
        emptied = prev_occupancy - curr_occupancy
        filled = curr_occupancy - prev_occupancy

        if not emptied and not filled:
            return MoveDetectionResult(
                success=False,
                error="No occupancy change detected",
            )

        all_legal = list(self._board.legal_moves)
        all_legal_ucis = [m.uci() for m in all_legal]

        # ── Step 2: Color-filtered detection (PRIMARY) ──────────────
        # Only diff the moving player's color → unambiguous even for captures
        if vision_pieces:
            result = self._detect_by_color(all_legal, vision_pieces)
            if result:
                return result

        # ── Step 3: Occupancy-based detection (FALLBACK) ────────────
        # Used when vision_pieces is empty or color detection fails
        candidates = []
        for move in all_legal:
            expected_emptied, expected_filled = self._move_occupancy_pattern(move)
            if expected_emptied == emptied and expected_filled == filled:
                candidates.append(move)

        candidate_ucis = [m.uci() for m in candidates]

        if len(candidates) == 1:
            return self._build_result(candidates[0], candidate_ucis)

        if len(candidates) > 1:
            return MoveDetectionResult(
                success=False,
                candidates=candidate_ucis,
                error=f"Ambiguous: {len(candidates)} legal moves match occupancy (no color info)",
            )

        # ── Step 4: Fuzzy match (1 square tolerance for vision noise) ──
        fuzzy_candidates = []
        for move in all_legal:
            expected_emptied, expected_filled = self._move_occupancy_pattern(move)
            extra = (emptied - expected_emptied) | (filled - expected_filled)
            missing = (expected_emptied - emptied) | (expected_filled - filled)
            if len(extra) + len(missing) <= 1:
                fuzzy_candidates.append(move)

        fuzzy_ucis = [m.uci() for m in fuzzy_candidates]

        if len(fuzzy_candidates) == 1:
            result = self._build_result(fuzzy_candidates[0], fuzzy_ucis)
            result.discrepancies.append(
                Discrepancy(
                    square="(fuzzy)",
                    type=DiscrepancyType.EXTRA,
                    tracker_piece=None,
                    vision_piece=None,
                    confidence=0.8,
                )
            )
            return result

        # No match
        return MoveDetectionResult(
            success=False,
            candidates=candidate_ucis or fuzzy_ucis,
            error=(
                f"No legal move matches. "
                f"Emptied: {sorted(emptied)}, Filled: {sorted(filled)}"
            ),
        )

    def push_move(self, move: chess.Move) -> None:
        """Apply a move to the tracker board."""
        # Track captured piece
        captured = self._board.piece_at(move.to_square)
        if captured:
            self._captured_pieces.append(_piece_to_str(captured))
        elif self._board.is_en_passant(move):
            # En passant capture
            color = "black" if self._board.turn == chess.WHITE else "white"
            self._captured_pieces.append(f"{color}_pawn")

        self._board.push(move)

    def push_uci(self, uci: str) -> None:
        """Apply a move in UCI format."""
        move = chess.Move.from_uci(uci)
        self.push_move(move)

    def validate_against_vision(
        self, vision_pieces: Dict[str, str]
    ) -> List[Discrepancy]:
        """
        Cross-check tracker state against vision detection.

        This is advisory only - the tracker is authoritative.

        Args:
            vision_pieces: {square: "color_piecetype"} from vision

        Returns:
            List of discrepancies (empty if all matches)
        """
        tracker_pieces = self.get_piece_map()
        return self._compare_pieces(tracker_pieces, vision_pieces)

    def to_dict(self) -> dict:
        """Serialize tracker state for storage."""
        return {
            "fen": self._board.fen(),
            "captured_pieces": self._captured_pieces.copy(),
            "move_stack": [m.uci() for m in self._board.move_stack],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BoardTracker":
        """Restore tracker from serialized state."""
        # Rebuild from move history for correct board state
        tracker = cls()
        for uci in data.get("move_stack", []):
            tracker.push_uci(uci)
        tracker._captured_pieces = data.get("captured_pieces", [])
        # Sanity check: FEN should match
        if tracker.fen.split()[0] != data["fen"].split()[0]:
            # Fallback: use stored FEN directly
            tracker._board = chess.Board(data["fen"])
        return tracker

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _build_result(
        self, move: chess.Move, candidate_ucis: List[str]
    ) -> MoveDetectionResult:
        """Build a successful MoveDetectionResult from a matched move."""
        piece = self._board.piece_at(move.from_square)
        piece_str = _piece_to_str(piece) if piece else None

        # Handle promotion
        promotion = None
        if move.promotion:
            promotion = chess.piece_name(move.promotion)
            if piece:
                color = "white" if piece.color == chess.WHITE else "black"
                piece_str = f"{color}_{chess.piece_name(move.promotion)}"

        # Determine capture info
        is_capture = False
        captured_piece = None
        captured = self._board.piece_at(move.to_square)
        if captured:
            is_capture = True
            captured_piece = _piece_to_str(captured)
        elif self._board.is_en_passant(move):
            is_capture = True
            color = "black" if self._board.turn == chess.WHITE else "white"
            captured_piece = f"{color}_pawn"

        return MoveDetectionResult(
            success=True,
            uci_move=move.uci(),
            piece=_piece_to_str(piece) if piece else None,
            is_capture=is_capture,
            captured_piece=captured_piece,
            is_castling=self._board.is_castling(move),
            is_en_passant=self._board.is_en_passant(move),
            promotion=promotion,
            candidates=candidate_ucis,
        )

    def _move_occupancy_pattern(self, move: chess.Move) -> Tuple[Set[str], Set[str]]:
        """
        Compute expected (emptied, filled) occupancy change for a move.

        Returns:
            (emptied_squares, filled_squares) as sets of square names
        """
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)

        emptied = set()
        filled = set()

        if self._board.is_castling(move):
            # King moves
            emptied.add(from_sq)
            filled.add(to_sq)
            # Rook moves
            rank = chess.square_rank(move.from_square)
            if chess.square_file(move.to_square) > chess.square_file(move.from_square):
                # Kingside
                rook_from = chess.square_name(chess.square(7, rank))
                rook_to = chess.square_name(chess.square(5, rank))
            else:
                # Queenside
                rook_from = chess.square_name(chess.square(0, rank))
                rook_to = chess.square_name(chess.square(3, rank))
            emptied.add(rook_from)
            filled.add(rook_to)

        elif self._board.is_en_passant(move):
            emptied.add(from_sq)
            filled.add(to_sq)
            # The captured pawn's square also becomes empty
            cap_rank = chess.square_rank(move.from_square)
            cap_file = chess.square_file(move.to_square)
            cap_sq = chess.square_name(chess.square(cap_file, cap_rank))
            emptied.add(cap_sq)

        elif self._board.piece_at(move.to_square) is not None:
            # Capture: from_sq empties, to_sq stays occupied (was occupied, still occupied)
            emptied.add(from_sq)
            # to_sq is NOT in filled (it was already occupied)
            # to_sq is NOT in emptied (it stays occupied)

        else:
            # Regular move
            emptied.add(from_sq)
            filled.add(to_sq)

        return emptied, filled

    def _detect_by_color(
        self,
        legal_moves: List[chess.Move],
        vision_pieces: Dict[str, str],
    ) -> Optional[MoveDetectionResult]:
        """
        Primary detection: compare moving player's piece positions before/after.

        Only tracks ONE color (the player whose turn it is). The opponent's
        pieces are assumed unchanged — we don't even look at them.

        Handles all move types:
          - Regular (e2e4): white leaves e2, white appears e4
          - Capture (Qh5xf2): white leaves h5, white appears f2
          - Castling (e1g1): white leaves {e1,h1}, white appears {g1,f1}
          - En passant: white leaves from, white appears to (opponent pawn vanishes separately)
          - Promotion: white leaves from, white appears to (color stays same)

        Returns MoveDetectionResult if resolved, None if color info insufficient.
        """
        moving_color = "white" if self._board.turn == chess.WHITE else "black"

        # Tracker: squares occupied by the moving player's pieces
        tracker_sqs = set()
        for sq_idx, piece in self._board.piece_map().items():
            if (piece.color == chess.WHITE) == (moving_color == "white"):
                tracker_sqs.add(chess.square_name(sq_idx))

        # Vision: squares where vision sees the moving player's color
        vision_sqs = set()
        for sq, cls in vision_pieces.items():
            if cls.startswith(moving_color):
                vision_sqs.add(sq)

        # Color-specific diff
        color_left = tracker_sqs - vision_sqs   # player's piece was here, now gone
        color_arrived = vision_sqs - tracker_sqs  # player's piece appeared here (new)

        if not color_left and not color_arrived:
            return None  # no color change detected, fall through to occupancy

        # Compute expected color pattern for each legal move
        candidates = []
        for move in legal_moves:
            expected_left, expected_arrived = self._move_color_pattern(move)
            if expected_left == color_left and expected_arrived == color_arrived:
                candidates.append(move)

        candidate_ucis = [m.uci() for m in candidates]

        if len(candidates) == 1:
            print(f"[ColorDetect] Resolved: {candidates[0].uci()} "
                  f"(left={color_left}, arrived={color_arrived})")
            return self._build_result(candidates[0], candidate_ucis)

        if len(candidates) > 1:
            # Check if all candidates are promotions to the same square (e7e8q/r/b/n)
            promos = [m for m in candidates if m.promotion]
            if len(promos) == len(candidates) and len(promos) > 0:
                # All are promotions — default to queen
                queen_promo = next((m for m in promos if m.promotion == chess.QUEEN), promos[0])
                print(f"[ColorDetect] Promotion resolved: {queen_promo.uci()} (default queen)")
                return self._build_result(queen_promo, candidate_ucis)
            print(f"[ColorDetect] Still ambiguous ({len(candidates)}): {candidate_ucis}")
            return None  # fall through to occupancy-based

        # No match with exact color diff — try with 1-square tolerance
        fuzzy = []
        for move in legal_moves:
            expected_left, expected_arrived = self._move_color_pattern(move)
            diff = (
                len(color_left.symmetric_difference(expected_left)) +
                len(color_arrived.symmetric_difference(expected_arrived))
            )
            if diff <= 1:
                fuzzy.append(move)

        if len(fuzzy) == 1:
            print(f"[ColorDetect] Fuzzy resolved: {fuzzy[0].uci()}")
            result = self._build_result(fuzzy[0], [m.uci() for m in fuzzy])
            result.discrepancies.append(
                Discrepancy(
                    square="(color-fuzzy)",
                    type=DiscrepancyType.EXTRA,
                    tracker_piece=None, vision_piece=None, confidence=0.85,
                )
            )
            return result

        print(f"[ColorDetect] No match (left={color_left}, arrived={color_arrived})")
        return None  # fall through to occupancy-based

    def _move_color_pattern(self, move: chess.Move) -> Tuple[Set[str], Set[str]]:
        """
        Compute expected (left, arrived) for the moving player's color.

        Unlike _move_occupancy_pattern which tracks all squares,
        this only tracks the moving player's pieces.

        For captures: piece leaves from_sq AND arrives at to_sq
        (unlike occupancy where to_sq was already occupied → no 'filled').
        """
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)

        left = {from_sq}      # player's piece leaves this square
        arrived = {to_sq}     # player's piece appears on this square

        if self._board.is_castling(move):
            # Rook also moves (same color)
            rank = chess.square_rank(move.from_square)
            if chess.square_file(move.to_square) > chess.square_file(move.from_square):
                left.add(chess.square_name(chess.square(7, rank)))      # h1/h8
                arrived.add(chess.square_name(chess.square(5, rank)))   # f1/f8
            else:
                left.add(chess.square_name(chess.square(0, rank)))      # a1/a8
                arrived.add(chess.square_name(chess.square(3, rank)))   # d1/d8

        # En passant: no extra color pattern needed — the captured pawn is
        # opponent color (we don't track it). from/to is sufficient.

        # Promotion: same color, just piece type changes. from/to is sufficient.

        return left, arrived

    def _compare_pieces(
        self,
        tracker_pieces: Dict[str, str],
        vision_pieces: Dict[str, str],
    ) -> List[Discrepancy]:
        """Compare tracker vs vision piece maps."""
        discrepancies = []
        all_squares = set(tracker_pieces.keys()) | set(vision_pieces.keys())

        captured_set = set(self._captured_pieces)

        for sq in sorted(all_squares):
            t_piece = tracker_pieces.get(sq)
            v_piece = vision_pieces.get(sq)

            if t_piece is None and v_piece is not None:
                # Vision sees piece on empty square
                dtype = DiscrepancyType.EXTRA
                if v_piece in captured_set:
                    dtype = DiscrepancyType.CAPTURED_REAPPEARED
                discrepancies.append(Discrepancy(
                    square=sq,
                    type=dtype,
                    tracker_piece=None,
                    vision_piece=v_piece,
                ))
            elif t_piece is not None and v_piece is None:
                discrepancies.append(Discrepancy(
                    square=sq,
                    type=DiscrepancyType.MISSING,
                    tracker_piece=t_piece,
                    vision_piece=None,
                ))
            elif t_piece is not None and v_piece is not None and t_piece != v_piece:
                discrepancies.append(Discrepancy(
                    square=sq,
                    type=DiscrepancyType.WRONG,
                    tracker_piece=t_piece,
                    vision_piece=v_piece,
                ))

        return discrepancies
