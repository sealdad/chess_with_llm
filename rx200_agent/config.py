# rx200_agent/config.py
"""
Configuration constants for the chess robot agent.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
VISION_MODULE = PROJECT_ROOT / "chess_vision"

# Camera settings
CAMERA_DEPTH_RES = (640, 480, 30)
CAMERA_COLOR_RES = (640, 480, 30)

# Robot calibration
HAND_EYE_CALIBRATION = PROJECT_ROOT / "hand_eye_calibration.yaml"

# Stockfish settings
STOCKFISH_PATH = "/usr/games/stockfish"
STOCKFISH_DEPTH = 15
STOCKFISH_SKILL_LEVEL = 20

# Board geometry (in meters, robot frame)
BOARD_SQUARE_SIZE = 0.05  # 5cm squares
BOARD_ORIGIN = [0.30, -0.15, 0.02]  # XYZ of a1 corner in robot base frame
GRAVEYARD_WHITE = [0.10, 0.25, 0.02]  # Where captured white pieces go
GRAVEYARD_BLACK = [0.10, -0.25, 0.02]  # Where captured black pieces go

# Robot motion parameters
APPROACH_HEIGHT = 0.15  # Height above board for approach
GRASP_HEIGHT = 0.02     # Height for grasping pieces
MOTION_SPEED = 0.3      # Robot motion speed factor

# Human move detection
POLL_INTERVAL = 1.5     # Seconds between vision polls
STABILITY_THRESHOLD = 3  # Consecutive stable frames to confirm move
MAX_WAIT_TIME = 600     # Maximum wait for human move (seconds)

# Vision pipeline
VISION_RETRY_LIMIT = 3  # Max vision failures before stopping

# LLM settings
LLM_MODEL = "gpt-4"  # Default model
LLM_TEMPERATURE = 0.7

# Starting FEN (standard chess starting position)
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Piece heights for grasping (in meters)
PIECE_HEIGHTS = {
    "pawn": 0.035,
    "rook": 0.040,
    "knight": 0.045,
    "bishop": 0.055,
    "queen": 0.070,
    "king": 0.080,
}

# Default configuration
DEFAULT_CONFIG = {
    "robot_color": "black",
    "stockfish_depth": STOCKFISH_DEPTH,
    "poll_interval": POLL_INTERVAL,
    "use_manual_trigger": False,
    "debug": False,
}
