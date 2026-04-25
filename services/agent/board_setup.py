"""
Board setup algorithm for Teach mode.

Computes the minimal set of physical moves needed to rearrange the board
from the current position to a target position (FEN).

Used when transitioning between lesson steps that have different FEN positions.
"""

import chess
from typing import Dict, List, Tuple, Optional


def _fen_to_piece_map(fen: str) -> Dict[str, str]:
    """Convert FEN to a dict of {square_name: piece_symbol}.

    Piece symbols use chess.Piece.symbol() convention:
    uppercase = white (P, N, B, R, Q, K), lowercase = black (p, n, b, r, q, k).
    """
    board = chess.Board(fen)
    piece_map = {}
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            piece_map[chess.square_name(sq)] = piece.symbol()
    return piece_map


def _piece_type_name(symbol: str) -> str:
    """Convert piece symbol to type name for robot API (e.g. 'P' -> 'pawn')."""
    names = {
        'p': 'pawn', 'n': 'knight', 'b': 'bishop',
        'r': 'rook', 'q': 'queen', 'k': 'king',
    }
    return names.get(symbol.lower(), 'pawn')


def compute_setup_moves(
    current_fen: str,
    target_fen: str,
) -> List[dict]:
    """Compute the moves needed to rearrange the board from current to target position.

    Returns a list of move instructions, each a dict with:
        {"action": "remove", "from": "e4", "piece_type": "pawn"}
            — pick up piece and place in capture zone
        {"action": "move", "from": "e2", "to": "e4", "piece_type": "pawn"}
            — pick up piece from one square and place on another
        {"action": "need_piece", "to": "e4", "piece_type": "queen", "piece_symbol": "Q"}
            — piece needed but not available on board (must come from capture zone / spare)

    The algorithm:
    1. Find pieces that need to be removed (on current but not target, or wrong type)
    2. Find pieces that need to be placed (on target but not current)
    3. Match removable pieces to needed placements by piece type (greedy)
    4. Unmatched removals → "remove" to capture zone
    5. Unmatched needs → "need_piece" (flag for human intervention or spare pieces)
    6. Order: removals first, then moves, then need_piece flags
    """
    current_map = _fen_to_piece_map(current_fen)
    target_map = _fen_to_piece_map(target_fen)

    all_squares = set(current_map.keys()) | set(target_map.keys())

    # Classify each square
    correct = []      # square has the right piece already
    wrong_piece = []  # square has a piece but wrong type — needs swap
    needs_piece = []  # target has a piece here but current doesn't
    extra_piece = []  # current has a piece here but target doesn't

    for sq in all_squares:
        cur = current_map.get(sq)
        tgt = target_map.get(sq)

        if cur == tgt:
            correct.append(sq)
        elif cur and tgt:
            # Wrong piece — need to remove current and place correct one
            wrong_piece.append(sq)
        elif cur and not tgt:
            extra_piece.append(sq)
        elif not cur and tgt:
            needs_piece.append(sq)

    # Build pools of available pieces (from extra + wrong squares)
    # and needed pieces (for needs_piece + wrong squares)
    available = []  # (square, symbol) — pieces we can pick up and reuse
    for sq in extra_piece:
        available.append((sq, current_map[sq]))
    for sq in wrong_piece:
        available.append((sq, current_map[sq]))

    needed = []  # (square, symbol) — pieces we need to place
    for sq in needs_piece:
        needed.append((sq, target_map[sq]))
    for sq in wrong_piece:
        needed.append((sq, target_map[sq]))

    # Greedy matching: pair available pieces with needed pieces by type
    moves = []
    used_available = set()
    used_needed = set()

    for i, (need_sq, need_sym) in enumerate(needed):
        for j, (avail_sq, avail_sym) in enumerate(available):
            if j in used_available:
                continue
            if avail_sym == need_sym:
                # Match! Move this piece from avail_sq to need_sq
                moves.append({
                    "action": "move",
                    "from": avail_sq,
                    "to": need_sq,
                    "piece_type": _piece_type_name(avail_sym),
                })
                used_available.add(j)
                used_needed.add(i)
                break

    # Unmatched available → remove to capture zone
    removals = []
    for j, (avail_sq, avail_sym) in enumerate(available):
        if j not in used_available:
            removals.append({
                "action": "remove",
                "from": avail_sq,
                "piece_type": _piece_type_name(avail_sym),
            })

    # Unmatched needed → need_piece (must come from spares)
    need_flags = []
    for i, (need_sq, need_sym) in enumerate(needed):
        if i not in used_needed:
            need_flags.append({
                "action": "need_piece",
                "to": need_sq,
                "piece_type": _piece_type_name(need_sym),
                "piece_symbol": need_sym,
            })

    # Order: removals first (clear squares), then moves, then flags
    # Within moves, do them in order: first pick from squares that need to be
    # cleared (wrong_piece sources), to avoid blocking
    result = removals + moves + need_flags
    return result


def count_setup_moves(current_fen: str, target_fen: str) -> int:
    """Quick count of how many physical robot moves are needed."""
    return len(compute_setup_moves(current_fen, target_fen))
