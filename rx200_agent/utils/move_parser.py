# rx200_agent/utils/move_parser.py
"""
Chess move parsing and detection utilities.
"""

from typing import Optional, Tuple, Dict, List
import chess


def parse_uci_move(uci_move: str) -> Tuple[str, str, Optional[str]]:
    """
    Parse a UCI format move string.

    Args:
        uci_move: Move in UCI format (e.g., "e2e4", "e7e8q" for promotion)

    Returns:
        (from_square, to_square, promotion) tuple
        promotion is None or piece letter ('q', 'r', 'b', 'n')
    """
    if len(uci_move) < 4:
        raise ValueError(f"Invalid UCI move: {uci_move}")

    from_square = uci_move[0:2]
    to_square = uci_move[2:4]
    promotion = uci_move[4] if len(uci_move) > 4 else None

    return from_square, to_square, promotion


def is_capture_move(fen: str, uci_move: str) -> bool:
    """
    Check if a move is a capture.

    Args:
        fen: Current board position in FEN
        uci_move: Move in UCI format

    Returns:
        True if the move captures a piece
    """
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci_move)

    # Check if destination square has a piece
    if board.piece_at(move.to_square) is not None:
        return True

    # Check for en passant
    if board.is_en_passant(move):
        return True

    return False


def get_captured_piece(fen: str, uci_move: str) -> Optional[str]:
    """
    Get the piece that will be captured by a move.

    Returns piece name like "black_pawn" or None if not a capture.
    """
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci_move)

    # Regular capture
    captured = board.piece_at(move.to_square)
    if captured:
        color = "white" if captured.color == chess.WHITE else "black"
        piece_name = chess.piece_name(captured.piece_type)
        return f"{color}_{piece_name}"

    # En passant
    if board.is_en_passant(move):
        # The captured pawn is on the same file as destination, same rank as source
        color = "white" if board.turn == chess.BLACK else "black"
        return f"{color}_pawn"

    return None


def is_castling_move(fen: str, uci_move: str) -> bool:
    """Check if a move is castling."""
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci_move)
    return board.is_castling(move)


def get_castling_rook_move(fen: str, uci_move: str) -> Optional[Tuple[str, str]]:
    """
    For a castling move, return the rook's from and to squares.

    Returns:
        (rook_from, rook_to) or None if not castling
    """
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci_move)

    if not board.is_castling(move):
        return None

    # Determine rook movement based on king movement
    from_file = chess.square_file(move.from_square)
    to_file = chess.square_file(move.to_square)
    rank = chess.square_rank(move.from_square)

    if to_file > from_file:  # Kingside
        rook_from = chess.square_name(chess.square(7, rank))  # h-file
        rook_to = chess.square_name(chess.square(5, rank))    # f-file
    else:  # Queenside
        rook_from = chess.square_name(chess.square(0, rank))  # a-file
        rook_to = chess.square_name(chess.square(3, rank))    # d-file

    return rook_from, rook_to


def detect_move_from_fen_diff(old_fen: str, new_fen: str) -> Optional[Dict]:
    """
    Detect what move was made by comparing two FEN positions.

    Args:
        old_fen: FEN before the move
        new_fen: FEN after the move

    Returns:
        Dict with move info or None if invalid/unclear
        {
            "uci": "e2e4",
            "from_square": "e2",
            "to_square": "e4",
            "piece": "white_pawn",
            "is_capture": False,
            "captured_piece": None,
            "is_castling": False,
            "promotion": None,
        }
    """
    old_board = chess.Board(old_fen)
    new_board = chess.Board(new_fen)

    # Find all legal moves from old position
    for move in old_board.legal_moves:
        # Apply move to a copy
        test_board = old_board.copy()
        test_board.push(move)

        # Compare piece placement (ignore turn, castling rights, etc.)
        if test_board.board_fen() == new_board.board_fen():
            # Found the move
            from_sq = chess.square_name(move.from_square)
            to_sq = chess.square_name(move.to_square)

            piece = old_board.piece_at(move.from_square)
            color = "white" if piece.color == chess.WHITE else "black"
            piece_name = chess.piece_name(piece.piece_type)

            captured = old_board.piece_at(move.to_square)
            captured_piece = None
            if captured:
                cap_color = "white" if captured.color == chess.WHITE else "black"
                captured_piece = f"{cap_color}_{chess.piece_name(captured.piece_type)}"
            elif old_board.is_en_passant(move):
                cap_color = "black" if color == "white" else "white"
                captured_piece = f"{cap_color}_pawn"

            return {
                "uci": move.uci(),
                "from_square": from_sq,
                "to_square": to_sq,
                "piece": f"{color}_{piece_name}",
                "is_capture": captured_piece is not None,
                "captured_piece": captured_piece,
                "is_castling": old_board.is_castling(move),
                "promotion": chess.piece_name(move.promotion) if move.promotion else None,
            }

    return None


def validate_fen(fen: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a FEN string.

    Returns:
        (is_valid, error_message)
    """
    try:
        board = chess.Board(fen)
        # Check if position is legal
        if not board.is_valid():
            return False, "Invalid chess position"
        return True, None
    except ValueError as e:
        return False, str(e)


def get_game_status(fen: str) -> str:
    """
    Get the current game status from a FEN position.

    Returns: "playing", "checkmate", "stalemate", or "draw"
    """
    board = chess.Board(fen)

    if board.is_checkmate():
        return "checkmate"
    elif board.is_stalemate():
        return "stalemate"
    elif board.is_insufficient_material() or board.can_claim_draw():
        return "draw"
    else:
        return "playing"


def is_in_check(fen: str) -> bool:
    """Check if the side to move is in check."""
    board = chess.Board(fen)
    return board.is_check()


def parse_spoken_move(spoken: str, board: chess.Board) -> Optional[str]:
    """
    Parse natural language chess move to UCI format.

    Understands various spoken formats:
        "pawn to e4" -> "e2e4"
        "knight f3" -> "g1f3"
        "e2 to e4" -> "e2e4"
        "castle kingside" -> "e1g1"
        "queen takes d7" -> finds queen capture to d7

    Args:
        spoken: Natural language move description
        board: Current chess.Board for context and validation

    Returns:
        UCI move string or None if unparseable
    """
    import re

    spoken_lower = spoken.lower().strip()

    # Handle castling
    if "castle" in spoken_lower or "castling" in spoken_lower:
        if any(w in spoken_lower for w in ["king", "short", "o-o", "0-0"]):
            uci = "e1g1" if board.turn == chess.WHITE else "e8g8"
        elif any(w in spoken_lower for w in ["queen", "long", "o-o-o", "0-0-0"]):
            uci = "e1c1" if board.turn == chess.WHITE else "e8c8"
        else:
            return None

        move = chess.Move.from_uci(uci)
        return uci if move in board.legal_moves else None

    # Extract squares mentioned in the speech
    squares = re.findall(r"[a-h][1-8]", spoken_lower)

    # Piece type mapping
    piece_map = {
        "king": chess.KING,
        "queen": chess.QUEEN,
        "rook": chess.ROOK,
        "castle": chess.ROOK,  # Sometimes people say "castle" for rook
        "bishop": chess.BISHOP,
        "knight": chess.KNIGHT,
        "horse": chess.KNIGHT,  # Common alternative
        "pawn": chess.PAWN,
    }

    # Detect piece type from speech
    piece_type = None
    for name, ptype in piece_map.items():
        if name in spoken_lower:
            piece_type = ptype
            break

    # Check for capture indication
    is_capture = any(w in spoken_lower for w in ["take", "takes", "capture", "captures", "x"])

    # If two squares mentioned (e.g., "e2 to e4")
    if len(squares) == 2:
        from_sq, to_sq = squares
        try:
            uci = from_sq + to_sq
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves:
                # Verify piece type if specified
                piece = board.piece_at(move.from_square)
                if piece_type is None or (piece and piece.piece_type == piece_type):
                    return uci
        except ValueError:
            pass

    # If one square mentioned (destination)
    if len(squares) >= 1:
        target_sq = squares[-1]

        # Find matching legal moves
        candidates = []
        for move in board.legal_moves:
            to_sq = chess.square_name(move.to_square)
            if to_sq != target_sq:
                continue

            piece = board.piece_at(move.from_square)
            if piece is None:
                continue

            # Filter by piece type if specified
            if piece_type is not None and piece.piece_type != piece_type:
                continue

            # Filter by capture if specified
            if is_capture and not board.is_capture(move):
                continue

            candidates.append(move)

        # If exactly one match, use it
        if len(candidates) == 1:
            return candidates[0].uci()

        # If multiple matches and we have source square hint
        if len(candidates) > 1 and len(squares) == 2:
            from_sq = squares[0]
            for move in candidates:
                if chess.square_name(move.from_square) == from_sq:
                    return move.uci()

    # Try to find piece + destination (e.g., "knight f3")
    if piece_type is not None:
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            if piece is None or piece.piece_type != piece_type:
                continue

            to_sq = chess.square_name(move.to_square)
            if to_sq in spoken_lower:
                return move.uci()

    return None


def describe_move(uci_move: str, fen: str) -> str:
    """
    Generate natural language description of a chess move.

    Args:
        uci_move: Move in UCI format
        fen: Board position FEN (before the move)

    Returns:
        Human-readable move description
    """
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci_move)
    piece = board.piece_at(move.from_square)

    piece_names = {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
        chess.KING: "king",
    }

    piece_name = piece_names.get(piece.piece_type, "piece") if piece else "piece"
    from_sq = chess.square_name(move.from_square)
    to_sq = chess.square_name(move.to_square)

    # Check for special moves
    if board.is_castling(move):
        if move.to_square > move.from_square:
            return "castles kingside"
        else:
            return "castles queenside"

    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        if captured:
            cap_name = piece_names.get(captured.piece_type, "piece")
            return f"{piece_name} takes {cap_name} on {to_sq}"
        elif board.is_en_passant(move):
            return f"{piece_name} takes en passant on {to_sq}"
        return f"{piece_name} takes on {to_sq}"

    if move.promotion:
        promo_name = piece_names.get(move.promotion, "queen")
        return f"{piece_name} to {to_sq} promoting to {promo_name}"

    return f"{piece_name} to {to_sq}"
