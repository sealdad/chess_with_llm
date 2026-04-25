#!/usr/bin/env python3
"""
Agent Service API

Provides REST API for the chess AI agent.
Orchestrates vision and robot services.

Endpoints:
    POST /game/start        - Start a new game
    POST /game/human_moved  - Notify that human has moved
    POST /game/voice_move   - Submit move via voice input
    GET  /game/state        - Get current game state
    POST /game/stop         - Stop current game
    GET  /health            - Health check

Voice Endpoints:
    WS   /ws/voice          - WebSocket for bidirectional voice
    POST /voice/speak       - TTS text to streaming audio
"""

import os
import re
import time
import asyncio
import base64
from typing import Optional, Literal
from contextlib import asynccontextmanager

import httpx
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Chess logic
import chess
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from rx200_agent.board_tracker import BoardTracker

# Voice services
from audio import STTService, TTSService, CosyVoiceTTSProvider
from websocket import VoiceHandler
from voice_events import VoiceEventQueue, VoiceEvent, VoiceEventPriority
from voice_processor import VoiceEventProcessor

# Conversation services
from conversation import (
    GameMode,
    AgentMode,  # backward-compatible alias
    DIFFICULTY_DEPTH,
    MODE_PROMPTS,
    MODE_VOICES,
    MODE_NAMES,
    LLMService,
    ConversationContext,
    IntentRouter,
    Intent,
    build_battle_prompt,
    build_teach_prompt,
    detect_mode_from_text,
    detect_language_from_text,
    get_mode_greeting,
    get_mode_name,
    get_mode_prompt,
    get_language_greeting,
    resolve_character_emotion,
)
from conversation.llm_service import GameContext

# Teach mode
from lessons import load_lesson, list_lessons, Lesson, LessonStep
from board_setup import compute_setup_moves


# Service URLs from environment or defaults
VISION_SERVICE_URL = os.getenv("VISION_SERVICE_URL", "http://localhost:8001")
ROBOT_SERVICE_URL = os.getenv("ROBOT_SERVICE_URL", "http://localhost:8002")
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "/usr/games/stockfish")

# ── Settings (persisted to agent_settings.yaml; this is the single source of truth) ──
_SETTINGS_FILE = Path("/app/agent_settings.yaml") if Path("/app").exists() else Path(__file__).parent.parent.parent / "agent_settings.yaml"


def _load_settings() -> dict:
    """Load settings from YAML file."""
    import yaml
    if _SETTINGS_FILE.exists():
        try:
            with open(_SETTINGS_FILE) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


def _save_settings(settings: dict):
    """Save settings to YAML file."""
    import yaml
    with open(_SETTINGS_FILE, "w") as f:
        yaml.dump(settings, f, allow_unicode=True, default_flow_style=False)


_settings = _load_settings()

# All user-facing settings come from agent_settings.yaml (edited via the API
# Settings panel in the web UI). Env vars are no longer consulted for these —
# only TTS_SERVICE_URL stays env-driven because it's deployment infra.
LLM_API_KEY = _settings.get("llm_api_key", "")
LLM_MODEL = _settings.get("llm_model", "gpt-4")
LLM_BASE_URL = _settings.get("llm_base_url", "")
LLM_MODEL_2 = _settings.get("llm_model_2", "")
LLM_API_KEY_2 = _settings.get("llm_api_key_2", "")
LLM_BASE_URL_2 = _settings.get("llm_base_url_2", "")
LLM_MODEL_3 = _settings.get("llm_model_3", "")
LLM_API_KEY_3 = _settings.get("llm_api_key_3", "")
LLM_BASE_URL_3 = _settings.get("llm_base_url_3", "")
USE_LLM = str(_settings.get("use_llm", "false")).lower() == "true"

STT_MODEL = _settings.get("stt_model", "whisper-1")
TTS_MODEL = _settings.get("tts_model", "tts-1")
TTS_VOICE = _settings.get("tts_voice", "alloy")
TTS_PROVIDER = _settings.get("tts_provider", "openai")
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL") or _settings.get("tts_service_url", "http://tts:8003")

AGENT_LANGUAGE = _settings.get("language", "zh-TW")


def _is_zh() -> bool:
    """Check if current language is any Chinese variant."""
    return current_language in ("zh-TW", "zh-CN")


class GameConfig(BaseModel):
    """Game configuration."""
    robot_color: Literal["white", "black"] = "black"
    use_llm: bool = True
    simulation: bool = False
    # Mode: "battle", "teach", or "watch"
    game_mode: str = "battle"
    # Battle mode settings
    difficulty: str = "intermediate"  # beginner, intermediate, advanced, master
    character: str = ""  # free-text personality (empty = default)
    move_source: str = "stockfish"  # "stockfish" or "llm"
    # Teach mode settings
    lesson_id: Optional[str] = None
    # Watch mode settings (AI vs AI)
    white_engine: str = "stockfish"
    black_engine: str = "stockfish"
    white_character: str = ""
    black_character: str = ""
    white_difficulty: str = "intermediate"
    black_difficulty: str = "intermediate"
    move_delay: float = 3.0
    # Legacy (ignored, kept for backward compat)
    agent_mode: Optional[str] = None
    stockfish_depth: Optional[int] = None


class GameState(BaseModel):
    """Current game state."""
    game_id: str
    status: Literal["waiting", "playing", "ended"]
    fen: str
    whose_turn: Literal["human", "robot"]
    move_number: int
    robot_color: str
    agent_mode: str = "battle"  # legacy field name, actually game_mode
    game_mode: str = "battle"
    difficulty: str = "intermediate"
    move_source: str = "stockfish"  # "stockfish" or "llm"
    last_human_move: Optional[str] = None
    last_robot_move: Optional[str] = None
    game_result: Optional[str] = None
    ascii_board: Optional[str] = None
    legal_moves: Optional[list] = None
    is_check: bool = False
    simulation: bool = False
    # Teach mode state
    lesson_id: Optional[str] = None
    lesson_step: Optional[int] = None
    lesson_total_steps: Optional[int] = None


class MoveNotification(BaseModel):
    """Human move notification."""
    trigger_type: Literal["manual", "auto"] = "manual"


class AgentResponse(BaseModel):
    """Generic agent response."""
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None
    game_state: Optional[GameState] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    vision_service: str
    robot_service: str
    stockfish_available: bool
    llm_model: str = ""
    timestamp: float


class SpeakRequest(BaseModel):
    """TTS request."""
    text: str
    voice: str = "alloy"
    speed: float = 1.0
    format: str = "mp3"


class VoiceMoveRequest(BaseModel):
    """Voice move request."""
    audio_data: str = Field(..., description="Base64 encoded audio")
    audio_format: str = "wav"


class VoiceMoveResponse(BaseModel):
    """Voice move response."""
    success: bool
    transcribed_text: Optional[str] = None
    parsed_move: Optional[str] = None
    agent_response_text: Optional[str] = None
    game_state: Optional[GameState] = None
    error: Optional[str] = None


# Global game state
current_game: Optional[dict] = None
game_lock = asyncio.Lock()
_robot_lock = asyncio.Lock()  # Serializes ALL robot commands (prevents USB serial overlap)
http_client: Optional[httpx.AsyncClient] = None
stockfish_engine = None

# Voice services
stt_service: Optional[STTService] = None
tts_service: Optional[TTSService] = None
voice_handler: Optional[VoiceHandler] = None

# Conversation services
llm_service: Optional[LLMService] = None
llm_service_2: Optional[LLMService] = None  # Second LLM for Watch mode
llm_service_3: Optional[LLMService] = None  # Third LLM for Watch mode
intent_router: Optional[IntentRouter] = None
conversation_context: Optional[ConversationContext] = None
current_mode: GameMode = GameMode.BATTLE
current_language: str = AGENT_LANGUAGE  # "en" or "zh-TW"
current_character: str = ""  # free-text personality for battle mode

# Teach mode state
_teach_lesson: Optional[Lesson] = None
_teach_step_idx: int = 0
_teach_hint_idx: int = 0  # tracks how many hints have been given for current step

# Voice event system
voice_event_queue: Optional[VoiceEventQueue] = None
voice_event_processor: Optional[VoiceEventProcessor] = None
_voice_loop_task: Optional[asyncio.Task] = None

# Auto-detect background task
_auto_detect_task: Optional[asyncio.Task] = None
_watch_task: Optional[asyncio.Task] = None  # Watch mode auto-play task
_game_stopping: bool = False  # Set during stop_game to skip cleanup robot moves
_tts_enabled: bool = True  # Controlled by frontend voice-output-enabled checkbox

# Game state persistence
GAME_STATE_FILE = Path("/app/game_state.yaml") if Path("/app").exists() else Path("game_state.yaml")

# LLM generation tracking (for interruption)
_llm_generation_id: int = 0


def _get_max_tokens() -> int:
    """Get max_tokens for current mode."""
    if current_mode == GameMode.TEACH:
        return 500
    if current_mode == GameMode.WATCH:
        return 500
    return 300


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup."""
    global http_client, stockfish_engine, stt_service, tts_service, voice_handler
    global llm_service, llm_service_2, llm_service_3, intent_router, conversation_context, current_mode, current_language
    global voice_event_queue, voice_event_processor, _voice_loop_task

    print("[Agent Service] Initializing...")

    # Create HTTP client for service calls (with generous pool to prevent exhaustion)
    http_client = httpx.AsyncClient(
        timeout=30.0,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10, keepalive_expiry=30),
    )

    # Initialize Stockfish
    try:
        from stockfish import Stockfish
        stockfish_engine = Stockfish(STOCKFISH_PATH)
        stockfish_engine.set_depth(15)
        print("[Agent Service] Stockfish ready")
    except Exception as e:
        print(f"[Agent Service] Stockfish init failed: {e}")
        stockfish_engine = None

    # Initialize conversation services
    try:
        llm_service = LLMService(
            model=LLM_MODEL,
            api_key=LLM_API_KEY or None,
            base_url=LLM_BASE_URL or None,
        )
        print(f"[Agent Service] LLM model: {LLM_MODEL}" + (f" (base_url: {LLM_BASE_URL})" if LLM_BASE_URL else ""))

        # Initialize second LLM service if configured
        if LLM_MODEL_2:
            try:
                llm_service_2 = LLMService(
                    model=LLM_MODEL_2,
                    api_key=LLM_API_KEY_2 or None,
                    base_url=LLM_BASE_URL_2 or None,
                )
                print(f"[Agent Service] LLM model 2: {LLM_MODEL_2}" + (f" (base_url: {LLM_BASE_URL_2})" if LLM_BASE_URL_2 else ""))
            except Exception as e2:
                print(f"[Agent Service] LLM 2 init failed: {e2}")
                llm_service_2 = None

        if LLM_MODEL_3:
            try:
                llm_service_3 = LLMService(
                    model=LLM_MODEL_3,
                    api_key=LLM_API_KEY_3 or None,
                    base_url=LLM_BASE_URL_3 or None,
                )
                print(f"[Agent Service] LLM model 3: {LLM_MODEL_3}" + (f" (base_url: {LLM_BASE_URL_3})" if LLM_BASE_URL_3 else ""))
            except Exception as e3:
                print(f"[Agent Service] LLM 3 init failed: {e3}")
                llm_service_3 = None

        intent_router = IntentRouter()
        conversation_context = ConversationContext(max_messages=20)
        current_mode = GameMode.BATTLE
        current_language = AGENT_LANGUAGE
        print(f"[Agent Service] Language: {current_language}")
        print("[Agent Service] Conversation services ready")
    except Exception as e:
        print(f"[Agent Service] Conversation services init failed: {e}")
        llm_service = None
        intent_router = None
        conversation_context = None

    # Initialize voice event system
    voice_event_queue = VoiceEventQueue()
    voice_event_processor = VoiceEventProcessor(
        on_game_command=handle_game_command,
        on_mode_switch=lambda mode, text: handle_mode_switch(mode, text),
        on_language_switch=lambda lang, text: handle_language_switch(lang, text),
        on_move=handle_move,
        on_conversation=handle_conversation,
    )

    # Initialize voice services
    try:
        stt_service = STTService(api_key=LLM_API_KEY or None, model=STT_MODEL)

        # TTS provider: OpenAI (cloud) or CosyVoice (local/remote)
        if TTS_PROVIDER == "cosyvoice":
            tts_service = CosyVoiceTTSProvider(
                service_url=TTS_SERVICE_URL,
                default_voice=TTS_VOICE,
                language=AGENT_LANGUAGE,
            )
            print(f"[Agent Service] TTS: CosyVoice at {TTS_SERVICE_URL}")
        else:
            tts_service = TTSService(api_key=LLM_API_KEY or None, default_voice=TTS_VOICE, model=TTS_MODEL)
            print(f"[Agent Service] TTS: OpenAI ({TTS_MODEL})")

        voice_handler = VoiceHandler(
            stt_service=stt_service,
            tts_service=tts_service,
            game_command_handler=process_voice_command,
            voice_event_queue=voice_event_queue,
            get_board_fn=_get_current_board,
            intent_router=intent_router,
            language=current_language,
        )
        print(f"[Agent Service] Voice services ready (STT: {STT_MODEL}, TTS: {TTS_PROVIDER}, voice: {TTS_VOICE})")
    except Exception as e:
        print(f"[Agent Service] Voice services init failed: {e}")
        stt_service = None
        tts_service = None
        voice_handler = None

    # Start voice event processing loop
    _voice_loop_task = asyncio.create_task(_voice_event_loop())
    print("[Agent Service] Voice event loop started")

    yield

    # Cancel auto-detect loop
    if _auto_detect_task and not _auto_detect_task.done():
        _auto_detect_task.cancel()
        try:
            await _auto_detect_task
        except asyncio.CancelledError:
            pass

    # Cancel voice event loop
    if _voice_loop_task:
        _voice_loop_task.cancel()
        try:
            await _voice_loop_task
        except asyncio.CancelledError:
            pass

    # Cleanup
    await http_client.aclose()
    print("[Agent Service] Shutdown complete")


app = FastAPI(
    title="Agent Service",
    description="Chess AI agent orchestration API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for control panel
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the control panel."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "RX-200 Chess Robot Agent API", "docs": "/docs"}


async def check_service(url: str) -> str:
    """Check if a service is healthy."""
    try:
        response = await http_client.get(f"{url}/health")
        if response.status_code == 200:
            return "healthy"
        return "unhealthy"
    except:
        return "unreachable"


async def _robot_post(path: str, timeout: float = 10.0, retries: int = 2) -> dict:
    """POST to robot service with retry on connection errors."""
    async with _robot_lock:
        for attempt in range(retries + 1):
            try:
                response = await http_client.post(f"{ROBOT_SERVICE_URL}{path}", timeout=timeout)
                return response.json()
            except Exception as e:
                if attempt < retries:
                    print(f"[Agent] Robot call {path} failed (attempt {attempt+1}), retrying: {e}")
                    await asyncio.sleep(0.5)
                else:
                    print(f"[Agent] Robot call {path} failed after {retries+1} attempts: {e}")
                    return {"success": False, "error": str(e)}


async def move_robot_for_vision() -> bool:
    """Move robot to vision position so it doesn't block the camera. Returns True on success."""
    result = await _robot_post("/arm/vision", timeout=10.0)
    if not result.get("success"):
        print(f"[Agent] Vision pos move failed: {result.get('error')}")
    return result.get("success", False)


async def move_robot_to_work() -> bool:
    """Move robot back to work position after vision capture. Returns True on success."""
    result = await _robot_post("/arm/work", timeout=10.0)
    if not result.get("success"):
        print(f"[Agent] Work pos move failed: {result.get('error')}")
    return result.get("success", False)


async def hover_robot_above_square(square: str) -> bool:
    """Move robot to hover above a square (for teach mode demonstrations)."""
    try:
        async with _robot_lock:
            response = await http_client.post(
                f"{ROBOT_SERVICE_URL}/arm/hover_square",
                json={"square": square},
                timeout=10.0,
            )
            result = response.json()
            if not result.get("success"):
                print(f"[Agent] Hover above {square} failed: {result.get('error')}")
            return result.get("success", False)
    except Exception as e:
        print(f"[Agent] Hover above {square} error: {e}")
        return False


async def capture_board() -> dict:
    """
    Call vision service to capture board state.

    Moves robot to vision position first (so arm doesn't block camera),
    captures the board, then moves robot back to work position.
    """
    try:
        # Move robot out of camera view
        await move_robot_for_vision()

        response = await http_client.post(f"{VISION_SERVICE_URL}/capture", timeout=15.0)
        result = response.json()

        # Move robot back to work position
        await move_robot_to_work()

        return result
    except Exception as e:
        # Try to move back to work even on error
        await move_robot_to_work()
        return {"success": False, "error": str(e)}


async def capture_enemy_piece(square: str, piece_type: str = "pawn") -> dict:
    """Call robot service /capture to remove enemy piece to capture zone."""
    async with _robot_lock:
        try:
            response = await http_client.post(
                f"{ROBOT_SERVICE_URL}/capture",
                json={"square": square, "piece_type": piece_type},
                timeout=45.0,
            )
            return response.json()
        except Exception as e:
            print(f"[Agent] capture_enemy_piece exception: {e}")
            return {"success": False, "error": str(e)}


async def robot_manual_pick(square: str, piece_type: str = "pawn",
                           skip_return_to_work: bool = False) -> dict:
    """Call robot service /manual_pick to pick up a piece."""
    async with _robot_lock:
        try:
            response = await http_client.post(
                f"{ROBOT_SERVICE_URL}/manual_pick",
                json={"square": square, "piece_type": piece_type,
                      "skip_return_to_work": skip_return_to_work},
                timeout=45.0,
            )
            return response.json()
        except Exception as e:
            print(f"[Agent] robot_manual_pick exception: {e}")
            return {"success": False, "error": str(e)}


async def robot_manual_place(square: str, piece_type: str = "pawn") -> dict:
    """Call robot service /manual_place to place a held piece."""
    async with _robot_lock:
        try:
            response = await http_client.post(
                f"{ROBOT_SERVICE_URL}/manual_place",
                json={"square": square, "piece_type": piece_type},
                timeout=45.0,
            )
            return response.json()
        except Exception as e:
            print(f"[Agent] robot_manual_place exception: {e}")
            return {"success": False, "error": str(e)}


async def pick_from_promotion_queen_pos(piece_type: str = "queen") -> dict:
    """Pick up a spare queen from the taught promotion_queen_position.

    Uses /arm/promotion_queen to move above, then custom XYZ pick sequence
    via /pickup_from_promotion_queen on the robot service.
    Falls back to: move to pos -> manual grasp sequence.
    """
    async with _robot_lock:
        try:
            response = await http_client.post(
                f"{ROBOT_SERVICE_URL}/pickup_from_promotion_queen",
                json={"piece_type": piece_type},
                timeout=30.0,
            )
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}


async def execute_robot_move(uci_move: str, fen: str) -> dict:
    """
    Execute a chess move using smart endpoints.

    Handles:
    - Captures: /capture to remove enemy piece first
    - Regular moves: /manual_pick + /manual_place
    - Castling: also moves the rook via /manual_pick + /manual_place
    - En passant: captures the pawn on the correct square
    - Promotion: remove pawn to capture zone, pick spare queen from
      promotion_queen_pos, place it on destination square
    """
    try:
        # Ensure robot is at work position before pick/place — separates
        # the big arm movement (from vision) from gripper commands,
        # preventing Dynamixel bus overload.
        await move_robot_to_work()

        board = chess.Board(fen)
        move = chess.Move.from_uci(uci_move)

        from_sq = uci_move[0:2]
        to_sq = uci_move[2:4]

        # Get the moving piece type
        piece = board.piece_at(move.from_square)
        piece_type = chess.piece_name(piece.piece_type) if piece else "pawn"

        is_capture = board.is_capture(move)
        is_castling = board.is_castling(move)
        is_en_passant = board.is_en_passant(move)
        is_promotion = move.promotion is not None

        if is_promotion:
            promotion_piece_type = chess.piece_name(move.promotion)
            print(f"[Agent] Promotion: {piece_type} promotes to {promotion_piece_type}")
        else:
            promotion_piece_type = None

        # --- Step 1: Handle capture (remove enemy piece) ---
        if is_capture:
            if is_en_passant:
                # En passant: captured pawn is on a different square
                ep_file = chess.square_file(move.to_square)
                ep_rank = chess.square_rank(move.from_square)
                captured_square = chess.square_name(chess.square(ep_file, ep_rank))
                captured_type = "pawn"
            else:
                captured_square = to_sq
                captured_piece = board.piece_at(move.to_square)
                captured_type = chess.piece_name(captured_piece.piece_type) if captured_piece else "pawn"

            print(f"[Agent] Step 1/3: Capturing {captured_type} on {captured_square}")
            cap_result = await capture_enemy_piece(captured_square, captured_type)
            print(f"[Agent] Capture result: {cap_result}")
            if not cap_result.get("success"):
                return {"success": False, "error": f"Capture failed: {cap_result.get('error')}"}
            print(f"[Agent] Step 1/3 done. Now moving {piece_type} {from_sq} -> {to_sq}")

        # --- Step 2: Handle promotion (swap pawn for queen) ---
        if is_promotion:
            # 2a. Remove the pawn from the board to capture zone
            print(f"[Agent] Promotion: removing pawn from {from_sq} to capture zone")
            pawn_remove = await capture_enemy_piece(from_sq, "pawn")
            if not pawn_remove.get("success"):
                return {"success": False, "error": f"Pawn removal failed: {pawn_remove.get('error')}"}

            # 2b. Pick up spare queen from promotion_queen_position
            print(f"[Agent] Promotion: picking up spare {promotion_piece_type} from promotion pos")
            queen_pick = await pick_from_promotion_queen_pos(promotion_piece_type)
            if not queen_pick.get("success"):
                return {"success": False, "error": f"Promotion queen pick failed: {queen_pick.get('error')}"}

            # 2c. Place the queen on the destination square
            print(f"[Agent] Promotion: placing {promotion_piece_type} on {to_sq}")
            queen_place = await robot_manual_place(to_sq, promotion_piece_type)
            if not queen_place.get("success"):
                return {"success": False, "error": f"Promotion queen place failed: {queen_place.get('error')}"}

        else:
            # --- Normal move: pick from source, place on destination ---
            print(f"[Agent] Step 2/3: Pick {piece_type} from {from_sq}")
            pick_result = await robot_manual_pick(from_sq, piece_type, skip_return_to_work=True)
            print(f"[Agent] Pick result: {pick_result}")
            if not pick_result.get("success"):
                return {"success": False, "error": f"Pick failed: {pick_result.get('error')}"}

            print(f"[Agent] Step 3/3: Place {piece_type} on {to_sq}")
            place_result = await robot_manual_place(to_sq, piece_type)
            print(f"[Agent] Place result: {place_result}")
            if not place_result.get("success"):
                return {"success": False, "error": f"Place failed: {place_result.get('error')}"}

        # --- Step 3: Handle castling (also move the rook) ---
        if is_castling:
            from_file = chess.square_file(move.from_square)
            to_file = chess.square_file(move.to_square)
            rank = chess.square_rank(move.from_square)

            if to_file > from_file:  # Kingside
                rook_from = chess.square_name(chess.square(7, rank))
                rook_to = chess.square_name(chess.square(5, rank))
            else:  # Queenside
                rook_from = chess.square_name(chess.square(0, rank))
                rook_to = chess.square_name(chess.square(3, rank))

            print(f"[Agent] Castling: moving rook {rook_from} to {rook_to}")
            rook_pick = await robot_manual_pick(rook_from, "rook", skip_return_to_work=True)
            if not rook_pick.get("success"):
                return {"success": False, "error": f"Rook pick failed: {rook_pick.get('error')}"}

            rook_place = await robot_manual_place(rook_to, "rook")
            if not rook_place.get("success"):
                return {"success": False, "error": f"Rook place failed: {rook_place.get('error')}"}

        return {
            "success": True,
            "move": uci_move,
            "from_square": from_sq,
            "to_square": to_sq,
            "was_capture": is_capture,
            "was_promotion": is_promotion,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


STOCKFISH_MAX_RETRIES = 2


def _restart_stockfish() -> bool:
    """Attempt to restart the Stockfish engine after a crash."""
    global stockfish_engine
    try:
        from stockfish import Stockfish
        stockfish_engine = Stockfish(STOCKFISH_PATH)
        stockfish_engine.set_depth(15)
        print("[Agent] Stockfish restarted successfully")
        return True
    except Exception as e:
        print(f"[Agent] Stockfish restart failed: {e}")
        stockfish_engine = None
        return False


def get_stockfish_move(fen: str, depth: int = 15) -> Optional[str]:
    """
    Get best move from Stockfish with retry and engine restart on failure.

    Returns the UCI move string, or None only if no legal moves exist.
    Raises RuntimeError if Stockfish is completely unavailable.
    """
    global stockfish_engine

    if stockfish_engine is None:
        if not _restart_stockfish():
            raise RuntimeError("Stockfish engine unavailable and restart failed")

    last_error = None
    for attempt in range(1 + STOCKFISH_MAX_RETRIES):
        try:
            stockfish_engine.set_fen_position(fen)
            move = stockfish_engine.get_best_move()
            if move is None:
                # Stockfish returns None when no legal moves exist
                return None
            return move
        except Exception as e:
            last_error = e
            print(f"[Agent] Stockfish error (attempt {attempt+1}): {e}")
            if _restart_stockfish():
                continue
            break

    raise RuntimeError(f"Stockfish failed after {1 + STOCKFISH_MAX_RETRIES} attempts: {last_error}")


async def get_llm_move(fen: str, difficulty: str = "intermediate", character: str = "", service_override=None) -> Optional[str]:
    """Ask the LLM to choose a chess move instead of Stockfish.

    The LLM receives the board position, legal moves, and a difficulty/personality
    hint so it can play in-character.  Falls back to Stockfish on any failure.

    Args:
        service_override: Use a specific LLMService instance instead of the global one.

    Returns a legal UCI move string, or None if no legal moves exist.
    """
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    legal_uci = [m.uci() for m in legal_moves]

    # Build a difficulty hint
    diff_hints = {
        "beginner": "Play at a beginner level. Make natural-looking but imperfect moves — occasional mistakes are fine.",
        "intermediate": "Play at an intermediate club level. Solid but not perfect.",
        "advanced": "Play strong, aggressive chess. Find tactical opportunities.",
        "master": "Play at the highest level you can. Find the strongest move.",
    }
    diff_hint = diff_hints.get(difficulty, diff_hints["intermediate"])

    char_hint = ""
    if character:
        char_hint = f"\nYour chess personality: {character}. Let this influence your style (aggressive, defensive, tricky, etc.) but always pick a legal move."

    prompt = (
        f"You are playing chess.\n"
        f"Position (FEN): {fen}\n"
        f"Legal moves: {', '.join(legal_uci)}\n"
        f"{diff_hint}{char_hint}\n"
        f"Choose one move from the legal moves list above.\n"
        f"Reply with ONLY the 4-character UCI move string. Example: e2e4\n"
        f"Your move:"
    )

    svc = service_override or llm_service
    if not svc:
        print("[Agent] LLM move requested but no LLM service — falling back to Stockfish")
        depth = DIFFICULTY_DEPTH.get(difficulty, 5)
        return get_stockfish_move(fen, depth)

    try:
        response = await svc.chat_async(
            user_message=prompt,
            system_prompt="You are a chess engine. Output exactly one legal UCI move from the list (4-5 characters like e2e4 or e7e8q). Nothing else.",
            max_tokens=100,
        )
        if not response:
            # Retry with minimal prompt (safety filter or empty response)
            print(f"[Agent] LLM returned empty, retrying with simple prompt...")
            response = await svc.chat_async(
                user_message=f"Legal chess moves: {', '.join(legal_uci)}\nPick one move. Reply with only the move:",
                system_prompt="Reply with one chess move from the list. Nothing else.",
                max_tokens=100,
            )
        if not response:
            print(f"[Agent] LLM returned empty response twice — falling back to Stockfish")
            depth = DIFFICULTY_DEPTH.get(difficulty, 5)
            return get_stockfish_move(fen, depth)
        chosen = response.strip().lower().replace(" ", "")
        # Direct match
        if chosen in legal_uci:
            print(f"[Agent] LLM chose move: {chosen}")
            return chosen
        # Extract any legal move from the response (LLM might add explanation)
        import re
        for m in sorted(legal_uci, key=len, reverse=True):  # Try longer moves first
            if m in chosen:
                print(f"[Agent] LLM chose move (extracted): {m} from '{chosen}'")
                return m
        # Regex: find any 4-5 char pattern matching a legal move
        candidates = re.findall(r'[a-h][1-8][a-h][1-8][qrbn]?', chosen)
        for c in candidates:
            if c in legal_uci:
                print(f"[Agent] LLM chose move (regex): {c} from '{chosen}'")
                return c
        print(f"[Agent] LLM returned invalid move '{chosen}' — falling back to Stockfish")
        depth = DIFFICULTY_DEPTH.get(difficulty, 5)
        return get_stockfish_move(fen, depth)
    except Exception as e:
        print(f"[Agent] LLM move failed: {e} — falling back to Stockfish")
        depth = DIFFICULTY_DEPTH.get(difficulty, 5)
        return get_stockfish_move(fen, depth)


def analyze_battle_move_context(uci_move: str, fen_before: str) -> dict:
    """Analyze a robot move to determine gesture and emotion for battle mode.

    Emotion is resolved from the character personality via resolve_character_emotion(),
    so a "zen master" character gets calm emotions while a "trash-talking pirate" gets
    angry/happy emotions for the same game events.

    Returns dict with 'gesture' and 'emotion' keys.
    """
    board = chess.Board(fen_before)
    move = chess.Move.from_uci(uci_move)

    is_capture = board.is_capture(move)

    # Push the move to check resulting position
    board.push(move)
    gives_check = board.is_check()
    is_checkmate = board.is_checkmate()
    board.pop()

    if is_checkmate:
        event = "checkmate"
        gesture = "celebrate"
    elif gives_check and is_capture:
        event = "check_capture"
        gesture = "celebrate"
    elif gives_check:
        event = "check"
        gesture = "nod"
    elif is_capture:
        event = "capture"
        gesture = "celebrate"
    else:
        event = "normal"
        gesture = "think"

    emotion = resolve_character_emotion(current_character, event)
    return {"gesture": gesture, "emotion": emotion}


def get_game_status(fen: str, board_with_history: Optional[chess.Board] = None) -> str:
    """Get game status from FEN or a board with full move history.

    Args:
        fen: Current position FEN (used for checkmate/stalemate/material checks)
        board_with_history: Optional board with move stack (needed for repetition/50-move detection)
    """
    board = board_with_history or chess.Board(fen)
    if board.is_checkmate():
        return "checkmate"
    elif board.is_stalemate():
        return "stalemate"
    elif board.is_insufficient_material():
        return "draw"
    elif board.is_fivefold_repetition() or board.is_seventyfive_moves():
        return "draw"
    elif board.can_claim_draw():
        return "draw"
    return "playing"


def fen_to_ascii(fen: str) -> str:
    """Convert FEN to ASCII board."""
    board = chess.Board(fen)
    return str(board)


def create_game_state() -> GameState:
    """Create GameState from current game."""
    if current_game is None:
        return None

    game_mode_str = current_game.get("game_mode", current_mode.value)
    gs = GameState(
        game_id=current_game["game_id"],
        status=current_game["status"],
        fen=current_game["fen"],
        whose_turn=current_game["whose_turn"],
        move_number=current_game["move_number"],
        robot_color=current_game["robot_color"],
        agent_mode=game_mode_str,
        game_mode=game_mode_str,
        difficulty=current_game.get("difficulty", "intermediate"),
        move_source=current_game.get("move_source", "stockfish"),
        last_human_move=current_game.get("last_human_move"),
        last_robot_move=current_game.get("last_robot_move"),
        game_result=current_game.get("game_result"),
        ascii_board=fen_to_ascii(current_game["fen"]),
        simulation=current_game.get("simulation", False),
        lesson_id=current_game.get("lesson_id"),
        lesson_step=current_game.get("lesson_step"),
        lesson_total_steps=current_game.get("lesson_total_steps"),
    )
    # Include legal moves and check status in simulation mode when it's human's turn
    if current_game.get("simulation") and current_game.get("whose_turn") == "human" and current_game.get("status") == "playing":
        board = chess.Board(current_game["fen"])
        gs.legal_moves = [m.uci() for m in board.legal_moves]
        gs.is_check = board.is_check()
    return gs


def get_game_context() -> Optional[GameContext]:
    """Create GameContext for LLM from current game."""
    if current_game is None:
        return None

    return GameContext(
        fen=current_game["fen"],
        move_number=current_game["move_number"],
        whose_turn=current_game["whose_turn"],
        last_human_move=current_game.get("last_human_move"),
        last_robot_move=current_game.get("last_robot_move"),
        robot_color=current_game["robot_color"],
        game_status=current_game["status"],
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check including dependent services."""
    vision_status = await check_service(VISION_SERVICE_URL)
    robot_status = await check_service(ROBOT_SERVICE_URL)

    return HealthResponse(
        status="healthy",
        vision_service=vision_status,
        robot_service=robot_status,
        stockfish_available=stockfish_engine is not None,
        llm_model=LLM_MODEL,
        timestamp=time.time(),
    )


async def _move_commentary(uci_move: str, fen_before: str):
    """Generate and broadcast speech about the robot's move + random chitchat.

    Streams text **and** TTS audio sentence-by-sentence so they stay in sync.
    Coach mode gets longer commentary; other modes get 1 short sentence.
    """
    try:
        move_desc = describe_move(uci_move, fen_before)
        is_teach = current_mode == GameMode.TEACH
        is_battle = current_mode == GameMode.BATTLE
        gen_id = _llm_generation_id

        # Detect check/checkmate after the move
        board_after = chess.Board(fen_before)
        board_after.push(chess.Move.from_uci(uci_move))
        is_check = board_after.is_check()
        is_checkmate = board_after.is_checkmate()
        check_hint = ""
        if is_checkmate:
            check_hint = "將殺！" if _is_zh() else "Checkmate! "
        elif is_check:
            check_hint = "將軍！" if _is_zh() else "Check! "

        # Determine gesture & emotion for battle mode
        battle_gesture = ""
        battle_emotion = ""
        if is_battle:
            ctx = analyze_battle_move_context(uci_move, fen_before)
            battle_gesture = ctx["gesture"]
            battle_emotion = ctx["emotion"]
            # Override emotion/gesture for check
            if is_checkmate:
                battle_emotion = "happy"
                battle_gesture = "celebrate"
            elif is_check:
                battle_emotion = "serious"

        is_watch = current_mode == GameMode.WATCH
        if llm_service:
            if _is_zh():
                if is_teach:
                    prompt = (
                        f"你剛下了一步棋：{move_desc}（{uci_move}）。{check_hint}"
                        f"簡短說明你的走法，並解釋背後的策略。"
                        f"2-3句話。"
                    )
                elif is_watch:
                    # Watch mode: identify which side and model
                    board_pre = chess.Board(fen_before)
                    side = "白方" if board_pre.turn == chess.WHITE else "黑方"
                    prompt = (
                        f"你是{side}。你剛下了{move_desc}。{check_hint}"
                        f"{'一定要說「將軍」！' if is_check else ''}"
                        f"用你的個性風格，以{side}身份說一句話。要簡短！"
                    )
                else:
                    prompt = (
                        f"你剛下了{move_desc}。{check_hint}"
                        f"{'一定要明確告訴對手「將軍」！' if is_check else ''}"
                        "用你的個性風格說一句話回應。最多1句，要簡短！"
                    )
            else:
                if is_teach:
                    prompt = (
                        f"You just played: {move_desc} ({uci_move}). {check_hint}"
                        f"Announce your move and briefly explain the strategy behind it. "
                        f"2-3 sentences max."
                    )
                elif is_watch:
                    board_pre = chess.Board(fen_before)
                    side = "White" if board_pre.turn == chess.WHITE else "Black"
                    check_instr = 'You MUST say "Check!" ' if is_check else ''
                    prompt = (
                        f"You are {side}. You just played {move_desc}. {check_hint}"
                        f"{check_instr}"
                        f"React as {side} in 1 short sentence. Be brief and in character!"
                    )
                else:
                    check_instr = 'You MUST clearly say "Check!" ' if is_check else ''
                    prompt = (
                        f"You just played {move_desc}. {check_hint}"
                        f"{check_instr}"
                        f"React in character in 1 short sentence only. Be very brief!"
                    )
            # Stream text + TTS in sync (NO gesture here — handled by make_robot_move)
            full_text = await _stream_text_with_tts(
                prompt, gen_id,
                emotion=battle_emotion,
            )
            commentary = full_text if full_text else f"I play {move_desc}."
        else:
            commentary = f"I play {move_desc}."
            await broadcast_game_event("voice_text_chunk", {"chunk": commentary})
            await _speak_text_only(commentary, emotion=battle_emotion)

        # Signal text is complete (tts_streamed flag tells frontend not to re-fetch TTS)
        await broadcast_game_event("voice_text_done", {"text": commentary, "tts_streamed": True})

        if conversation_context:
            conversation_context.add_exchange(f"(robot played {uci_move})", commentary)

        print(f"[Agent] Move commentary: {commentary}")
    except Exception as e:
        print(f"[Agent] Move commentary error: {e}")


async def _welcome_speech(robot_color: str):
    """Generate and broadcast a welcome speech when a game starts.

    Streams text **and** TTS audio sentence-by-sentence so they stay in sync.
    """
    try:
        mode_greeting = get_mode_greeting(current_mode, current_language)
        is_teach = current_mode == GameMode.TEACH
        gen_id = _llm_generation_id

        is_watch = current_mode == GameMode.WATCH
        if llm_service:
            if _is_zh():
                if is_watch:
                    w_model = current_game.get("white_engine", "stockfish") if current_game else "?"
                    b_model = current_game.get("black_engine", "stockfish") if current_game else "?"
                    if w_model in ("llm", "llm2"):
                        w_svc = llm_service_2 if w_model == "llm2" and llm_service_2 else llm_service
                        w_model = w_svc.model if w_svc else w_model
                    if b_model in ("llm", "llm2"):
                        b_svc = llm_service_2 if b_model == "llm2" and llm_service_2 else llm_service
                        b_model = b_svc.model if b_svc else b_model
                    prompt = f"AI 對戰開始！白方：{w_model} vs 黑方：{b_model}。用1句話宣布比賽開始，像體育播報員一樣興奮！"
                elif is_teach and _teach_lesson:
                    lesson_title = _teach_lesson.title_zh or _teach_lesson.title
                    prompt = (
                        f"歡迎來到課程「{lesson_title}」！"
                        f"用1句話簡短歡迎學生，要親切！不要說明練習內容。"
                    )
                else:
                    prompt = (
                        f"新遊戲開始！你是{robot_color}方。"
                        f"用1句話打個招呼，要簡短有個性！"
                    )
            else:
                if is_watch:
                    w_model = current_game.get("white_engine", "stockfish") if current_game else "?"
                    b_model = current_game.get("black_engine", "stockfish") if current_game else "?"
                    if w_model in ("llm", "llm2"):
                        w_svc = llm_service_2 if w_model == "llm2" and llm_service_2 else llm_service
                        w_model = w_svc.model if w_svc else w_model
                    if b_model in ("llm", "llm2"):
                        b_svc = llm_service_2 if b_model == "llm2" and llm_service_2 else llm_service
                        b_model = b_svc.model if b_svc else b_model
                    prompt = f"AI vs AI match! White: {w_model} vs Black: {b_model}. Announce the match in 1 excited sentence like a sports commentator!"
                elif is_teach and _teach_lesson:
                    prompt = (
                        f"Welcome to the lesson '{_teach_lesson.title}'! "
                        f"1 short welcoming sentence only. Don't explain the exercise."
                    )
                else:
                    prompt = (
                        f"New game! You play {robot_color}. "
                        f"Say hi in 1 short sentence only. Stay in character, be very brief!"
                    )
            # Stream text + TTS in sync
            welcome_emotion = "encouraging" if is_teach else resolve_character_emotion(current_character, "welcome")
            full_text = await _stream_text_with_tts(prompt, gen_id, emotion=welcome_emotion)
            welcome_text = full_text if full_text else mode_greeting
        else:
            welcome_emotion = "encouraging" if is_teach else resolve_character_emotion(current_character, "welcome")
            welcome_text = mode_greeting
            await broadcast_game_event("voice_text_chunk", {"chunk": welcome_text})
            await _speak_text_only(welcome_text, emotion=welcome_emotion)

        # Signal text is complete
        await broadcast_game_event("voice_text_done", {"text": welcome_text, "tts_streamed": True})

        if conversation_context:
            conversation_context.add_exchange("(game started)", welcome_text)

        print(f"[Agent] Welcome speech: {welcome_text}")
    except Exception as e:
        print(f"[Agent] Welcome speech error: {e}")


# =============================================================================
# Game state persistence
# =============================================================================

def _save_game_state():
    """Persist current game state to disk for crash recovery."""
    if current_game is None:
        if GAME_STATE_FILE.exists():
            GAME_STATE_FILE.unlink()
        return

    import yaml
    state = {
        "game_id": current_game["game_id"],
        "status": current_game["status"],
        "fen": current_game["fen"],
        "whose_turn": current_game["whose_turn"],
        "move_number": current_game["move_number"],
        "robot_color": current_game["robot_color"],
        "use_llm": current_game.get("use_llm", True),
        "stockfish_depth": current_game.get("stockfish_depth", 15),
        "game_mode": current_game.get("game_mode", "battle"),
        "difficulty": current_game.get("difficulty", "intermediate"),
        "character": current_game.get("character", ""),
        "move_source": current_game.get("move_source", "stockfish"),
        "last_human_move": current_game.get("last_human_move"),
        "last_robot_move": current_game.get("last_robot_move"),
        "game_result": current_game.get("game_result"),
        "simulation": current_game.get("simulation", False),
        "lesson_id": current_game.get("lesson_id"),
        "lesson_step": current_game.get("lesson_step"),
        "lesson_total_steps": current_game.get("lesson_total_steps"),
    }
    tracker = current_game.get("tracker")
    if tracker:
        state["tracker"] = tracker.to_dict()

    try:
        with open(GAME_STATE_FILE, "w") as f:
            yaml.safe_dump(state, f, default_flow_style=False)
    except Exception as e:
        print(f"[Agent] Failed to save game state: {e}")


def _load_game_state() -> Optional[dict]:
    """Load persisted game state from disk. Returns None if no saved state."""
    import yaml

    if not GAME_STATE_FILE.exists():
        return None

    try:
        with open(GAME_STATE_FILE, "r") as f:
            state = yaml.safe_load(f)

        if not state or state.get("status") != "playing":
            GAME_STATE_FILE.unlink(missing_ok=True)
            return None

        tracker = BoardTracker()
        if state.get("tracker"):
            tracker = BoardTracker.from_dict(state["tracker"])

        game = {
            "game_id": state["game_id"],
            "status": state["status"],
            "fen": state["fen"],
            "whose_turn": state["whose_turn"],
            "move_number": state["move_number"],
            "robot_color": state["robot_color"],
            "use_llm": state.get("use_llm", True),
            "stockfish_depth": state.get("stockfish_depth", 15),
            "move_source": state.get("move_source", "stockfish"),
            "last_human_move": state.get("last_human_move"),
            "last_robot_move": state.get("last_robot_move"),
            "game_result": state.get("game_result"),
            "simulation": state.get("simulation", False),
            "tracker": tracker,
        }
        print(f"[Agent] Restored game {game['game_id']} at move {game['move_number']}")
        return game

    except Exception as e:
        print(f"[Agent] Failed to load game state: {e}")
        return None


# =============================================================================
# Pre-game validation
# =============================================================================

async def validate_robot_readiness() -> dict:
    """
    Check that the robot service has all required taught positions
    loaded before starting a game.

    Returns {"ready": bool, "warnings": [...], "errors": [...]}
    """
    warnings = []
    errors = []

    try:
        robot_health = await check_service(ROBOT_SERVICE_URL)
        if robot_health != "healthy":
            errors.append(f"Robot service is {robot_health}")
            return {"ready": False, "warnings": warnings, "errors": errors}

        vision_health = await check_service(VISION_SERVICE_URL)
        if vision_health != "healthy":
            errors.append(f"Vision service is {vision_health}")
            return {"ready": False, "warnings": warnings, "errors": errors}

        checks = await asyncio.gather(
            http_client.get(f"{ROBOT_SERVICE_URL}/board_surface_z", timeout=5.0),
            http_client.get(f"{ROBOT_SERVICE_URL}/square_positions", timeout=5.0),
            http_client.get(f"{ROBOT_SERVICE_URL}/capture_zone", timeout=5.0),
            http_client.get(f"{ROBOT_SERVICE_URL}/connection", timeout=5.0),
            http_client.get(f"{ROBOT_SERVICE_URL}/calibration/status", timeout=5.0),
            return_exceptions=True,
        )

        # board_surface_z
        if isinstance(checks[0], Exception):
            warnings.append(f"Could not check board_surface_z: {checks[0]}")
        else:
            bsz = checks[0].json()
            if bsz.get("board_surface_z") is None:
                warnings.append("board_surface_z not taught — grasp height may be inaccurate")

        # square_positions
        if isinstance(checks[1], Exception):
            warnings.append(f"Could not check square_positions: {checks[1]}")
        else:
            sp = checks[1].json()
            positions = sp.get("positions", sp) if isinstance(sp, dict) else {}
            count = len(positions) if isinstance(positions, dict) else 0
            if count == 0:
                warnings.append("No square positions taught — using hardcoded/vision fallback")
            elif count < 64:
                warnings.append(f"Only {count}/64 square positions taught")

        # capture_zone
        if isinstance(checks[2], Exception):
            warnings.append(f"Could not check capture_zone: {checks[2]}")
        else:
            cz = checks[2].json()
            if not cz.get("x") and not cz.get("capture_zone"):
                warnings.append("Capture zone not taught — captures will fail")

        # connection
        if isinstance(checks[3], Exception):
            warnings.append(f"Could not check robot connection: {checks[3]}")
        else:
            conn = checks[3].json()
            if not conn.get("connected", False):
                errors.append("Robot arm not connected")

        # calibration
        if isinstance(checks[4], Exception):
            warnings.append(f"Could not check calibration: {checks[4]}")
        else:
            cal = checks[4].json()
            is_calibrated = cal.get("calibrated", False) or cal.get("loaded", False)
            if not is_calibrated:
                warnings.append(
                    "Camera-robot calibration not loaded — "
                    "vision+calibration position lookup will fall back to hardcoded grid"
                )

    except Exception as e:
        errors.append(f"Pre-game validation error: {e}")

    ready = len(errors) == 0
    if warnings:
        print(f"[Agent] Pre-game warnings: {warnings}")
    if errors:
        print(f"[Agent] Pre-game errors: {errors}")
    else:
        print("[Agent] Pre-game validation passed")

    return {"ready": ready, "warnings": warnings, "errors": errors}


# =============================================================================
# Board re-sync
# =============================================================================

async def resync_tracker_from_vision() -> dict:
    """
    Re-synchronize the board tracker with the physical board by finding
    the legal move that explains the occupancy difference.

    Returns {"success": bool, "action": str, "details": str}
    """
    if current_game is None:
        return {"success": False, "action": "none", "details": "No active game"}

    tracker: BoardTracker = current_game.get("tracker")
    if tracker is None:
        return {"success": False, "action": "none", "details": "No tracker"}

    vision_result = await _capture_occupancy()
    if not vision_result.get("success"):
        return {"success": False, "action": "none",
                "details": f"Vision failed: {vision_result.get('error')}"}

    expected_occ = tracker.get_occupancy()
    current_occ = set(vision_result.get("occupied_squares", []))

    vacated = expected_occ - current_occ
    appeared = current_occ - expected_occ

    if not vacated and not appeared:
        return {"success": True, "action": "none", "details": "Board matches tracker"}

    print(f"[Resync] Discrepancy: vacated={sorted(vacated)}, appeared={sorted(appeared)}")

    board = tracker.board.copy()
    matching_moves = []
    for legal_move in board.legal_moves:
        test_board = board.copy()
        test_board.push(legal_move)
        simulated_occ = set()
        for sq in chess.SQUARES:
            if test_board.piece_at(sq) is not None:
                simulated_occ.add(chess.square_name(sq))
        if simulated_occ == current_occ:
            matching_moves.append(legal_move)

    if len(matching_moves) == 1:
        move = matching_moves[0]
        uci = move.uci()
        tracker.push_uci(uci)
        new_fen = tracker.fen

        is_human_turn = current_game["whose_turn"] == "human"
        current_game["fen"] = new_fen
        if is_human_turn:
            current_game["last_human_move"] = uci
            current_game["whose_turn"] = "robot"
        else:
            current_game["last_robot_move"] = uci
            current_game["whose_turn"] = "human"

        _save_game_state()
        print(f"[Resync] Auto-applied move: {uci}")
        await broadcast_game_event("resync_applied", {
            "move": uci,
            "game_state": create_game_state().dict(),
        })
        return {"success": True, "action": "auto_applied",
                "details": f"Applied move {uci} to match physical board"}

    elif len(matching_moves) > 1:
        candidates = [m.uci() for m in matching_moves]
        return {"success": False, "action": "ambiguous",
                "details": f"Multiple moves match: {candidates}. Manual resolution needed."}
    else:
        return {"success": False, "action": "unresolvable",
                "details": f"vacated={sorted(vacated)}, appeared={sorted(appeared)} — no legal move matches"}


@app.post("/game/resync")
async def resync_board():
    """Re-synchronize the board tracker with the physical board."""
    if current_game is None:
        raise HTTPException(status_code=400, detail="No active game")
    async with game_lock:
        return await resync_tracker_from_vision()


# =============================================================================
# Post-move verification
# =============================================================================

async def verify_robot_move(expected_move: str) -> dict:
    """
    Verify a robot move by capturing the board and comparing occupancy.

    Returns {"verified": bool, "discrepancies": [...], "error": str|None}
    """
    try:
        vision_result = await _capture_occupancy()
        if not vision_result.get("success"):
            return {"verified": False, "discrepancies": [],
                    "error": f"Vision failed: {vision_result.get('error')}"}

        tracker: BoardTracker = current_game.get("tracker")
        if tracker is None:
            return {"verified": False, "discrepancies": [], "error": "No tracker"}

        expected_occupancy = tracker.get_occupancy()
        vision_occupancy = set(vision_result.get("occupied_squares", []))

        missing = expected_occupancy - vision_occupancy
        extra = vision_occupancy - expected_occupancy

        discrepancies = []
        if missing:
            discrepancies.append(f"Expected but not seen: {', '.join(sorted(missing))}")
        if extra:
            discrepancies.append(f"Unexpected pieces on: {', '.join(sorted(extra))}")

        # Allow up to 2 squares of discrepancy (vision noise)
        verified = (len(missing) + len(extra)) <= 2

        if not verified:
            print(f"[Agent] Move verification FAILED for {expected_move}: {discrepancies}")
            await broadcast_game_event("move_verification_failed", {
                "move": expected_move,
                "discrepancies": discrepancies,
            })
        else:
            if discrepancies:
                print(f"[Agent] Move verified with minor discrepancies: {discrepancies}")
            else:
                print(f"[Agent] Move verified: {expected_move}")

        return {"verified": verified, "discrepancies": discrepancies, "error": None}
    except Exception as e:
        print(f"[Agent] Move verification error: {e}")
        return {"verified": False, "discrepancies": [], "error": str(e)}


# =============================================================================
# Game recovery endpoint
# =============================================================================

@app.post("/game/recover", response_model=AgentResponse)
async def recover_game():
    """Recover a game from persisted state after a service crash."""
    global current_game

    async with game_lock:
        if current_game is not None:
            return AgentResponse(success=False, error="A game is already active. Stop it first.")

        restored = _load_game_state()
        if restored is None:
            return AgentResponse(success=False, error="No saved game to recover.")

        current_game = restored

        if current_game["whose_turn"] == "human":
            await _start_auto_detect()

        return AgentResponse(
            success=True,
            message=f"Recovered game {current_game['game_id']} at move {current_game['move_number']}. "
                    f"It's {current_game['whose_turn']}'s turn.",
            game_state=create_game_state(),
        )


@app.post("/game/start", response_model=AgentResponse)
async def start_game(config: GameConfig):
    """Start a new game in Battle or Teach mode."""
    global current_game, current_mode, current_character
    global _teach_lesson, _teach_step_idx, _teach_hint_idx

    async with game_lock:
        if not config.simulation:
            # Connect to robot first (so readiness check sees it as connected)
            conn_data = await _robot_post("/connect", timeout=15.0)
            if conn_data.get("connected"):
                print(f"[Agent] Robot connected: {conn_data.get('message')}")
            else:
                print(f"[Agent] Robot connect failed: {conn_data.get('error')}")

        # --- Pre-game validation ---
        if not config.simulation:
            readiness = await validate_robot_readiness()
            if not readiness["ready"]:
                return AgentResponse(
                    success=False,
                    error=f"Robot not ready: {'; '.join(readiness['errors'])}",
                )
        else:
            readiness = {"ready": True, "warnings": [], "errors": []}

        # Resolve game mode
        game_mode = config.game_mode.lower()
        if game_mode == "teach":
            current_mode = GameMode.TEACH
        elif game_mode == "watch":
            current_mode = GameMode.WATCH
        else:
            current_mode = GameMode.BATTLE

        # Resolve difficulty → stockfish depth
        difficulty = config.difficulty.lower()
        stockfish_depth = DIFFICULTY_DEPTH.get(difficulty, config.stockfish_depth or 5)

        # Store character for battle mode
        current_character = config.character or ""

        # --- Teach mode: load lesson ---
        if current_mode == GameMode.TEACH:
            if not config.lesson_id:
                return AgentResponse(success=False, error="Teach mode requires a lesson_id.")
            lesson = load_lesson(config.lesson_id)
            if lesson is None:
                return AgentResponse(success=False, error=f"Lesson '{config.lesson_id}' not found.")
            if not lesson.steps:
                return AgentResponse(success=False, error=f"Lesson '{config.lesson_id}' has no steps.")

            _teach_lesson = lesson
            _teach_step_idx = 0
            _teach_hint_idx = 0
            initial_fen = lesson.steps[0].fen
        else:
            _teach_lesson = None
            _teach_step_idx = 0
            _teach_hint_idx = 0
            initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        # Initialize game state with board tracker
        tracker = BoardTracker()
        if current_mode == GameMode.TEACH:
            # Set tracker to lesson's starting FEN
            tracker = BoardTracker(fen=initial_fen)

        # Determine whose turn
        if current_mode == GameMode.TEACH:
            first_turn = "human"
        elif current_mode == GameMode.WATCH:
            first_turn = "robot"  # Both sides are AI in watch mode
        else:
            first_turn = "human" if config.robot_color == "black" else "robot"

        current_game = {
            "game_id": f"game_{int(time.time())}",
            "status": "playing",
            "fen": initial_fen,
            "whose_turn": first_turn,
            "move_number": 1,
            "robot_color": config.robot_color,
            "use_llm": config.use_llm,
            "stockfish_depth": stockfish_depth,
            "difficulty": difficulty,
            "move_source": config.move_source.lower() if current_mode == GameMode.BATTLE else "stockfish",
            "game_mode": current_mode.value,
            "character": current_character,
            "last_human_move": None,
            "last_robot_move": None,
            "game_result": None,
            "tracker": tracker,
            "simulation": config.simulation if current_mode != GameMode.WATCH else True,
            "pregame_warnings": readiness["warnings"] if not config.simulation else [],
        }

        # Watch mode extras
        if current_mode == GameMode.WATCH:
            current_game["white_engine"] = config.white_engine
            current_game["black_engine"] = config.black_engine
            current_game["white_character"] = config.white_character
            current_game["black_character"] = config.black_character
            current_game["white_difficulty"] = config.white_difficulty
            current_game["black_difficulty"] = config.black_difficulty
            current_game["move_delay"] = config.move_delay

        # Teach mode extras
        if current_mode == GameMode.TEACH:
            current_game["lesson_id"] = _teach_lesson.lesson_id
            current_game["lesson_step"] = 0
            current_game["lesson_total_steps"] = len(_teach_lesson.steps)

        # Persist initial state
        _save_game_state()

        print(f"[Agent] Game started: {current_game['game_id']}")
        print(f"[Agent] Mode: {current_mode.value}, difficulty: {difficulty}"
              + (f", lesson: {config.lesson_id}" if config.lesson_id else "")
              + (", simulation=True" if config.simulation else ""))

        if not config.simulation:
            # Move robot to home position (arm stands up to greet)
            try:
                await _robot_post("/arm/home", timeout=10.0)
                print("[Agent] Robot moved to home position")
            except Exception as e:
                print(f"[Agent] Home position error: {e}")

        # Swap llm_service for battle mode if using LLM2/LLM3
        print(f"[Agent] Game start: mode={current_mode.value}, move_source={config.move_source}")
        if current_mode == GameMode.BATTLE and config.move_source in ("llm2", "llm3"):
            if config.move_source == "llm3" and llm_service_3:
                globals()['llm_service'] = llm_service_3
            elif config.move_source == "llm2" and llm_service_2:
                globals()['llm_service'] = llm_service_2

        # Fire welcome speech (skip in Watch mode — just start playing)
        if current_mode != GameMode.WATCH:
            asyncio.ensure_future(_welcome_speech(config.robot_color))

        if current_mode == GameMode.WATCH:
            # Watch mode: start AI vs AI auto-play loop
            global _watch_task
            _watch_task = asyncio.create_task(_watch_auto_play())
        elif current_mode == GameMode.TEACH:
            # Teach mode: set up the first position and send instruction
            if not config.simulation:
                standard_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                current_game["fen"] = standard_fen
                await _setup_board_position(initial_fen)
            await broadcast_game_event("teach_input_lock", {"locked": True})
            asyncio.ensure_future(_teach_speak_first_instruction())
        elif current_game["whose_turn"] == "robot":
            # Battle mode: robot moves first
            await make_robot_move()
        else:
            # Battle mode: human moves first — start auto-detect loop (skip in simulation)
            if not config.simulation:
                await _start_auto_detect()

        return AgentResponse(
            success=True,
            message=f"Game started in {current_mode.value} mode."
                   + (f" Warnings: {'; '.join(readiness['warnings'])}" if readiness["warnings"] else ""),
            game_state=create_game_state(),
        )


async def _capture_occupancy() -> dict:
    """
    Capture board occupancy using 5-shot union from vision service.

    Moves robot to vision position first, calls /capture/occupancy,
    and stays at vision position (robot only moves to work when it
    needs to execute its own move).
    """
    try:
        await move_robot_for_vision()
        response = await http_client.post(
            f"{VISION_SERVICE_URL}/capture/occupancy", timeout=15.0,
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


async def _capture_occupancy_in_place() -> dict:
    """
    Capture board occupancy WITHOUT moving the robot.

    Assumes the robot is already at the vision position.
    Used during auto-detect polling to avoid cycling vision<->work every 5s.
    """
    try:
        response = await http_client.post(
            f"{VISION_SERVICE_URL}/capture/occupancy", timeout=15.0,
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


async def _detect_and_respond() -> dict:
    """
    Shared logic: capture board, detect human move, make robot response.

    Returns dict with success, message/error, game_state.
    """
    global current_game

    # Capture OUTSIDE game_lock — this involves robot movement (seconds)
    vision_result = await _capture_occupancy()

    if not vision_result.get("success"):
        return {
            "success": False,
            "error": f"Vision failed: {vision_result.get('error')}",
            "game_state": create_game_state() if current_game else None,
        }

    async with game_lock:
        # Re-check game is still active after releasing/reacquiring lock
        if current_game is None:
            return {
                "success": False,
                "error": "Game ended during capture",
            }

        # Use tracker for occupancy-based move detection
        tracker: BoardTracker = current_game.get("tracker")
        if tracker is None:
            return {
                "success": False,
                "error": "BoardTracker not initialized",
                "game_state": create_game_state(),
            }

        prev_occupancy = tracker.get_occupancy()
        curr_occupancy = set(vision_result.get("occupied_squares", []))

        # Build vision_pieces for capture tiebreaking
        vision_pieces = vision_result.get("piece_positions", {})

        detection = tracker.detect_human_move(
            prev_occupancy=prev_occupancy,
            curr_occupancy=curr_occupancy,
            vision_pieces=vision_pieces if vision_pieces else None,
        )

        if not detection.success:
            return {
                "success": False,
                "error": f"Could not detect valid move: {detection.error}",
                "game_state": create_game_state(),
            }

        detected_move = detection.uci_move
        print(f"[Agent] Human played: {detected_move}")

        # Apply move to tracker
        fen_before_move = tracker.fen
        tracker.push_uci(detected_move)
        new_fen = tracker.fen

        # Update game state
        current_game["fen"] = new_fen
        current_game["last_human_move"] = detected_move
        current_game["whose_turn"] = "robot"

        # Check game status
        status = get_game_status(new_fen)
        if status != "playing":
            current_game["status"] = "ended"
            current_game["game_result"] = status
            asyncio.ensure_future(_game_over_speech(status, detected_move, fen_before_move, who_won="human"))
            return {
                "success": True,
                "message": f"Game over: {status}",
                "game_state": create_game_state(),
            }

        # Make robot's move
        await make_robot_move()

        return {
            "success": True,
            "message": f"Detected move: {detected_move}. Robot responded.",
            "game_state": create_game_state(),
        }


@app.post("/game/human_moved", response_model=AgentResponse)
async def human_moved(notification: MoveNotification, background_tasks: BackgroundTasks):
    """
    Notify that human has made a move.

    The agent will:
    1. Cancel auto-detect polling (manual trigger takes priority)
    2. Capture the board to detect the new position
    3. Calculate and execute the robot's response
    """
    if current_game is None:
        raise HTTPException(status_code=400, detail="No active game")

    if current_game["whose_turn"] != "human":
        raise HTTPException(status_code=400, detail="Not human's turn")

    await _cancel_auto_detect()

    result = await _detect_and_respond()

    return AgentResponse(
        success=result["success"],
        message=result.get("message"),
        error=result.get("error"),
        game_state=result.get("game_state") or create_game_state(),
    )


class ManualMoveRequest(BaseModel):
    """Manual move submission (UCI notation like e2e4, e7e8q)."""
    move: str = Field(..., description="UCI move string")


@app.post("/game/submit_move", response_model=AgentResponse)
async def submit_move(req: ManualMoveRequest):
    """
    Manually submit a human move in UCI notation (e.g. e2e4, e7e8q).

    Validates the move is legal, applies it to the board tracker,
    and triggers the robot's response.
    """
    global current_game

    if current_game is None:
        raise HTTPException(status_code=400, detail="No active game")

    if current_game["whose_turn"] != "human":
        raise HTTPException(status_code=400, detail="Not human's turn")

    uci_move = req.move.strip().lower()

    # Validate move is legal
    board = chess.Board(current_game["fen"])
    try:
        move = chess.Move.from_uci(uci_move)
    except ValueError:
        return AgentResponse(success=False, error=f"Invalid UCI format: '{uci_move}'")

    if move not in board.legal_moves:
        legal_uci = [m.uci() for m in board.legal_moves]
        return AgentResponse(
            success=False,
            error=f"Illegal move: {uci_move}. Legal moves: {', '.join(sorted(legal_uci))}",
        )

    # Cancel auto-detect — manual submission takes priority
    await _cancel_auto_detect()

    print(f"[Agent] Manual move submitted: {uci_move}")

    async with game_lock:
        # Re-validate inside lock (state may have changed since initial check)
        if current_game.get("whose_turn") != "human":
            return AgentResponse(success=False, error="Not human's turn (move already processed)")

        tracker: BoardTracker = current_game.get("tracker")
        if tracker:
            # Verify move is still legal in current position
            current_board = chess.Board(tracker.fen)
            try:
                m = chess.Move.from_uci(uci_move)
            except ValueError:
                return AgentResponse(success=False, error=f"Invalid move: {uci_move}")
            if m not in current_board.legal_moves:
                return AgentResponse(success=False, error=f"Move {uci_move} is no longer legal")
            tracker.push_uci(uci_move)
            new_fen = tracker.fen
        else:
            board = chess.Board(current_game["fen"])
            move = chess.Move.from_uci(uci_move)
            if move not in board.legal_moves:
                return AgentResponse(success=False, error=f"Move {uci_move} is no longer legal")
            board.push(move)
            new_fen = board.fen()

        current_game["fen"] = new_fen
        current_game["last_human_move"] = uci_move
        current_game["whose_turn"] = "robot"
        _save_game_state()

        await broadcast_game_event("move_detected", {
            "move": uci_move,
            "source": "manual",
            "game_state": create_game_state().dict(),
        })

        # Check game status
        status = get_game_status(new_fen)
        if status != "playing":
            current_game["status"] = "ended"
            current_game["game_result"] = status
            _save_game_state()
            # Fire game-over speech in background
            asyncio.ensure_future(_game_over_speech(status, uci_move, new_fen, who_won="human"))
            return AgentResponse(
                success=True,
                message=f"Move {uci_move} applied. Game over: {status}",
                game_state=create_game_state(),
            )

        # Make robot's move (starts new auto-detect loop)
        await make_robot_move()

        return AgentResponse(
            success=True,
            message=f"Move {uci_move} applied. Robot responded.",
            game_state=create_game_state(),
        )


@app.post("/game/validate_board")
async def validate_board():
    """
    Capture the board and validate it against the expected state.

    Checks:
    - Vision can see the board
    - Piece count is reasonable
    - If a game is active, compares vision occupancy with tracker state
    - Reports discrepancies (extra/missing/wrong pieces)
    """
    vision_result = await capture_board()

    if not vision_result.get("success"):
        return {
            "success": False,
            "valid": False,
            "error": f"Vision failed: {vision_result.get('error')}",
        }

    vision_pieces = vision_result.get("piece_positions", {})
    is_valid = vision_result.get("is_valid", True)
    warnings = vision_result.get("warnings", [])
    piece_count = len(vision_pieces)

    result = {
        "success": True,
        "valid": is_valid and len(warnings) == 0,
        "piece_count": piece_count,
        "vision_warnings": warnings,
        "issues": [],
    }

    # If a game is active, compare with tracker
    if current_game and current_game.get("tracker"):
        tracker = current_game["tracker"]
        expected_occupancy = tracker.get_occupancy()
        vision_occupancy = set(vision_pieces.keys())

        missing = expected_occupancy - vision_occupancy
        extra = vision_occupancy - expected_occupancy

        if missing:
            result["issues"].append(f"Missing pieces on: {', '.join(sorted(missing))}")
            result["valid"] = False
        if extra:
            result["issues"].append(f"Unexpected pieces on: {', '.join(sorted(extra))}")
            result["valid"] = False

        if not missing and not extra:
            result["issues"].append("Board matches expected state")
            result["valid"] = result["valid"]  # keep existing validity
    else:
        # No game — just check starting position sanity
        if piece_count < 28:
            result["issues"].append(f"Only {piece_count} pieces detected (expected ~32)")
            result["valid"] = False
        elif piece_count == 32:
            result["issues"].append("All 32 pieces detected")

    return result


# =============================================================================
# Teach mode
# =============================================================================

async def _setup_board_position(target_fen: str):
    """Physically rearrange the board to match a target FEN position.

    Uses compute_setup_moves() to find the minimal set of pick/place/remove
    operations, then executes them via the robot service.
    """
    if current_game is None:
        return

    current_fen = current_game.get("fen", "8/8/8/8/8/8/8/8 w - - 0 1")
    moves = compute_setup_moves(current_fen, target_fen)

    if not moves:
        print("[Teach] Board already matches target position")
        return

    print(f"[Teach] Setting up board: {len(moves)} operations")
    await move_robot_to_work()

    for op in moves:
        action = op["action"]
        piece_type = op.get("piece_type", "pawn")

        if action == "remove":
            print(f"[Teach] Remove {piece_type} from {op['from']} to capture zone")
            result = await capture_enemy_piece(op["from"], piece_type)
            if not result.get("success"):
                print(f"[Teach] Remove failed: {result.get('error')}")

        elif action == "move":
            print(f"[Teach] Move {piece_type} from {op['from']} to {op['to']}")
            pick_result = await robot_manual_pick(op["from"], piece_type, skip_return_to_work=True)
            if pick_result.get("success"):
                place_result = await robot_manual_place(op["to"], piece_type)
                if not place_result.get("success"):
                    print(f"[Teach] Place failed: {place_result.get('error')}")
            else:
                print(f"[Teach] Pick failed: {pick_result.get('error')}")

        elif action == "need_piece":
            print(f"[Teach] Need {piece_type} on {op['to']} — requires spare piece")
            # TODO: pick from spare piece pool when available
            await broadcast_game_event("teach_need_piece", {
                "square": op["to"],
                "piece_type": piece_type,
                "piece_symbol": op.get("piece_symbol", "?"),
            })


def _get_teach_step() -> Optional[LessonStep]:
    """Get the current teach mode lesson step."""
    if _teach_lesson is None or _teach_step_idx >= len(_teach_lesson.steps):
        return None
    return _teach_lesson.steps[_teach_step_idx]


@app.get("/teach/lessons")
async def get_lessons():
    """List all available lessons."""
    lessons = list_lessons()
    return {"lessons": lessons}


@app.get("/teach/lesson/{lesson_id}")
async def get_lesson(lesson_id: str):
    """Get full lesson details including all steps."""
    lesson = load_lesson(lesson_id)
    if lesson is None:
        raise HTTPException(status_code=404, detail=f"Lesson '{lesson_id}' not found")
    return lesson.dict()


@app.get("/teach/state")
async def get_teach_state():
    """Get current teach mode state."""
    if current_mode != GameMode.TEACH or _teach_lesson is None:
        return {"active": False}

    step = _get_teach_step()
    zh = _is_zh()

    return {
        "active": True,
        "lesson_id": _teach_lesson.lesson_id,
        "lesson_title": _teach_lesson.title_zh if zh and _teach_lesson.title_zh else _teach_lesson.title,
        "step_index": _teach_step_idx,
        "total_steps": len(_teach_lesson.steps),
        "hints_given": _teach_hint_idx,
        "current_step": {
            "fen": step.fen,
            "instruction": step.instruction_zh if zh and step.instruction_zh else step.instruction,
            "has_expected_move": step.expected_move is not None,
            "hints_available": len(step.hints_zh if zh and step.hints_zh else step.hints) - _teach_hint_idx,
        } if step else None,
    }


class TeachMoveRequest(BaseModel):
    """Student's move submission in teach mode."""
    move: str


@app.post("/teach/check_move")
async def teach_check_move(req: TeachMoveRequest):
    """Check if the student's move matches the expected move for the current step."""
    global _teach_step_idx, _teach_hint_idx

    if current_mode != GameMode.TEACH or _teach_lesson is None:
        raise HTTPException(status_code=400, detail="Not in teach mode")

    step = _get_teach_step()
    if step is None:
        raise HTTPException(status_code=400, detail="No current step")

    uci_move = req.move.strip().lower()
    zh = _is_zh()

    # Validate move is legal in the current position
    board = chess.Board(step.fen)
    try:
        move_obj = chess.Move.from_uci(uci_move)
    except ValueError:
        return {"correct": False, "error": f"Invalid move format: {uci_move}"}

    if move_obj not in board.legal_moves:
        return {"correct": False, "error": f"Illegal move: {uci_move}"}

    # If no expected move, any legal move is accepted (exploration step)
    if step.expected_move is None or uci_move == step.expected_move:
        explanation = step.explanation_zh if zh and step.explanation_zh else step.explanation
        # Update game FEN to reflect the student's move
        if current_game:
            board.push(move_obj)
            current_game["fen"] = board.fen()

        # Wrap up: congrats → opponent move + why → ready?
        commentary = await _teach_wrap_up(step, uci_move)

        return {
            "correct": True,
            "message": commentary,
            "explanation": explanation,
        }
    else:
        # Wrong move (gesture: shake — gentle "try again")
        move_desc = describe_move(uci_move, step.fen)
        if not (current_game and current_game.get("simulation", False)):
            asyncio.create_task(_play_robot_gesture("shake", return_to_vision=True))
        if llm_service:
            prompt = (
                f"學生走了{move_desc}，不對。用1句話提示再試。最多10個字。不要說答案。" if zh else
                f"Student played {move_desc}, wrong. Hint to retry in 1 sentence, max 10 words. Don't reveal answer."
            )
            commentary = await get_llm_response(prompt)
        else:
            commentary = "That's not quite right. Think about it and try again!" if not zh else "不太對喔，再想想看！"

        return {
            "correct": False,
            "message": commentary,
        }


@app.post("/teach/human_moved")
async def teach_human_moved():
    """
    Detect the student's physical move via vision, then check against lesson.

    Flow: capture board → detect move via BoardTracker → check against expected.
    This is the teach-mode equivalent of /game/human_moved.
    """
    global _teach_step_idx, _teach_hint_idx

    if current_mode != GameMode.TEACH or _teach_lesson is None:
        raise HTTPException(status_code=400, detail="Not in teach mode")
    if current_game is None:
        raise HTTPException(status_code=400, detail="No active game")

    step = _get_teach_step()
    if step is None:
        raise HTTPException(status_code=400, detail="No current step")

    # Capture board via vision
    vision_result = await _capture_occupancy()
    if not vision_result.get("success"):
        return {
            "success": False,
            "error": f"Vision failed: {vision_result.get('error')}",
        }

    # Detect move using BoardTracker (same as battle mode detection)
    tracker: BoardTracker = current_game.get("tracker")
    if tracker is None:
        return {"success": False, "error": "BoardTracker not initialized"}

    prev_occupancy = tracker.get_occupancy()
    curr_occupancy = set(vision_result.get("occupied_squares", []))
    vision_pieces = vision_result.get("piece_positions", {})

    detection = tracker.detect_human_move(
        prev_occupancy=prev_occupancy,
        curr_occupancy=curr_occupancy,
        vision_pieces=vision_pieces if vision_pieces else None,
    )

    if not detection.success:
        return {
            "success": False,
            "error": f"Could not detect move: {detection.error}",
            "message": "Could not detect your move. Make sure you moved a piece, then try again.",
        }

    detected_move = detection.uci_move
    print(f"[Teach] Detected physical move: {detected_move}")

    # Now check the detected move against the lesson (reuse check_move logic)
    zh = _is_zh()
    board = chess.Board(step.fen)
    try:
        move_obj = chess.Move.from_uci(detected_move)
    except ValueError:
        return {"success": False, "correct": False, "error": f"Invalid detected move: {detected_move}"}

    if move_obj not in board.legal_moves:
        return {"success": False, "correct": False, "error": f"Detected illegal move: {detected_move}"}

    # Check if move is correct
    if step.expected_move is None or detected_move == step.expected_move:
        # Apply move to tracker and update FEN
        tracker.push_uci(detected_move)
        current_game["fen"] = tracker.fen

        # Wrap up: congrats → opponent move + why → ready?
        explanation = step.explanation_zh if zh and step.explanation_zh else step.explanation
        commentary = await _teach_wrap_up(step, detected_move)

        return {"success": True, "correct": True, "detected_move": detected_move,
                "message": commentary, "explanation": explanation}
    else:
        # Wrong move — but it's already physically on the board (gesture: shake)
        move_desc = describe_move(detected_move, step.fen)
        if not (current_game and current_game.get("simulation", False)):
            asyncio.create_task(_play_robot_gesture("shake", return_to_vision=True))
        if llm_service:
            prompt = (
                f"學生走了{move_desc}，不對。用1句話提示再試，提醒放回棋子。最多10個字。不要說答案。" if zh else
                f"Student played {move_desc}, wrong. Hint to retry, remind to put piece back. Max 10 words. Don't reveal answer."
            )
            commentary = await get_llm_response(prompt)
        else:
            commentary = ("That's not the right move. Put the piece back and try again!"
                          if not zh else "不太對喔，把棋子放回去再試一次！")
        return {"success": True, "correct": False, "detected_move": detected_move,
                "message": commentary}


@app.post("/teach/hint")
async def teach_hint():
    """Get the next hint for the current step."""
    global _teach_hint_idx

    if current_mode != GameMode.TEACH or _teach_lesson is None:
        raise HTTPException(status_code=400, detail="Not in teach mode")

    step = _get_teach_step()
    if step is None:
        raise HTTPException(status_code=400, detail="No current step")

    zh = _is_zh()
    hints = step.hints_zh if zh and step.hints_zh else step.hints

    if _teach_hint_idx >= len(hints):
        return {
            "hint": None,
            "message": "沒有更多提示了！" if zh else "No more hints available!",
            "hints_remaining": 0,
        }

    hint = hints[_teach_hint_idx]
    _teach_hint_idx += 1

    # Optionally enhance hint with LLM
    if llm_service:
        prompt = (
            f"你正在給學生一個提示：「{hint}」。用你教練的方式自然地表達這個提示。" if zh else
            f"You're giving the student a hint: '{hint}'. Express this hint naturally in your coaching style."
        )
        enhanced = await get_llm_response(prompt)
        message = enhanced
    else:
        message = hint

    return {
        "hint": hint,
        "message": message,
        "hints_remaining": len(hints) - _teach_hint_idx,
    }


@app.post("/teach/next_step")
async def teach_next_step():
    """Advance to the next lesson step."""
    global _teach_step_idx, _teach_hint_idx

    if current_mode != GameMode.TEACH or _teach_lesson is None:
        raise HTTPException(status_code=400, detail="Not in teach mode")

    _teach_step_idx += 1
    _teach_hint_idx = 0

    # Cancel auto-detect from previous step BEFORE demo move
    # (otherwise it sees robot arm movement as a board change)
    await _cancel_auto_detect()

    if _teach_step_idx >= len(_teach_lesson.steps):
        # Lesson complete (gesture: celebrate!)
        zh = _is_zh()
        if not (current_game and current_game.get("simulation", False)):
            asyncio.create_task(_play_robot_gesture("celebrate", return_to_vision=True))
        if llm_service:
            title = _teach_lesson.title_zh if zh and _teach_lesson.title_zh else _teach_lesson.title
            prompt = (
                f"學生完成了「{title}」課程的所有步驟！恭喜他們並總結學到的內容。" if zh else
                f"The student completed all steps of the '{title}' lesson! Congratulate them and summarize what they learned."
            )
            message = await get_llm_response(prompt)
        else:
            message = "Lesson complete! Great job!" if not zh else "課程完成！做得好！"

        return {
            "complete": True,
            "message": message,
            "game_state": create_game_state().dict() if current_game else None,
        }

    step = _get_teach_step()
    zh = _is_zh()

    # FEN and tracker are already set to this step's position by
    # _teach_wrap_up() (called after the previous step's correct move).
    # Just update lesson_step index and save.
    if current_game:
        current_game["lesson_step"] = _teach_step_idx
        _save_game_state()

    # Ensure robot at vision position before 3-phase instruction
    is_physical = current_game and not current_game.get("simulation", False)
    if is_physical:
        await move_robot_for_vision()

    instruction = step.instruction_zh if zh and step.instruction_zh else step.instruction

    # Run 3-phase teaching flow: illustrate → demo → ending
    await _teach_3phase_instruction(step)

    # Restart auto-detect — all 3 phases done
    if is_physical:
        await _start_auto_detect(already_at_vision=True)

    return {
        "complete": False,
        "step_index": _teach_step_idx,
        "total_steps": len(_teach_lesson.steps),
        "instruction": instruction,
        "message": instruction,
        "fen": step.fen,
        "game_state": create_game_state().dict() if current_game else None,
    }


@app.post("/teach/stop")
async def teach_stop():
    """Stop the current teach session and return to idle."""
    global _teach_lesson, _teach_step_idx, _teach_hint_idx, current_game

    await _cancel_auto_detect()

    _teach_lesson = None
    _teach_step_idx = 0
    _teach_hint_idx = 0

    if current_game:
        current_game["status"] = "ended"
        current_game["game_result"] = "lesson_stopped"
        gs = create_game_state()
        current_game = None
        _save_game_state()
        return {"success": True, "message": "Lesson stopped.", "game_state": gs.dict()}

    return {"success": True, "message": "No active lesson."}


@app.post("/board/setup")
async def setup_board(target_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
    """Manually trigger board setup to a target FEN (for testing)."""
    if current_game is None:
        raise HTTPException(status_code=400, detail="No active game")

    current_fen = current_game["fen"]
    moves = compute_setup_moves(current_fen, target_fen)

    if not current_game.get("simulation", False):
        await _setup_board_position(target_fen)

    # Update game FEN
    current_game["fen"] = target_fen
    current_game["tracker"] = BoardTracker(fen=target_fen)
    _save_game_state()

    return {
        "success": True,
        "operations": len(moves),
        "moves": moves,
        "game_state": create_game_state().dict(),
    }


async def make_robot_move():
    """Calculate and execute robot's move."""
    global current_game

    if current_game is None:
        return

    is_simulation = current_game.get("simulation", False)
    fen = current_game["fen"]
    depth = current_game.get("stockfish_depth", 15)
    move_source = current_game.get("move_source", "stockfish")

    # Get best move — from LLM or Stockfish
    try:
        if move_source in ("llm", "llm2", "llm3"):
            difficulty = current_game.get("difficulty", "intermediate")
            character = current_game.get("character", "")
            # Pick the right LLM service
            if move_source == "llm3" and llm_service_3:
                svc = llm_service_3
            elif move_source == "llm2" and llm_service_2:
                svc = llm_service_2
            else:
                svc = llm_service
            print(f"[Agent] Requesting move from {svc.model if svc else 'LLM'} (difficulty={difficulty})")
            best_move = await get_llm_move(fen, difficulty, character, service_override=svc)
        else:
            best_move = get_stockfish_move(fen, depth)
    except RuntimeError as e:
        print(f"[Agent] Move engine unavailable: {e}")
        current_game["last_robot_error"] = f"Move engine failed: {e}"
        await broadcast_game_event("robot_error", {
            "move": None, "error": f"Move engine failed: {e}",
        })
        return

    if best_move is None:
        print("[Agent] No legal moves available (confirmed by Stockfish)")
        current_game["status"] = "ended"
        current_game["game_result"] = "no_moves"
        _save_game_state()
        return

    print(f"[Agent] Playing: {best_move}")

    # Analyze move context for gesture/emotion (battle mode)
    battle_ctx = None
    if current_mode == GameMode.BATTLE:
        battle_ctx = analyze_battle_move_context(best_move, fen)
        # Override for check/checkmate
        board_check = chess.Board(fen)
        board_check.push(chess.Move.from_uci(best_move))
        if board_check.is_checkmate():
            battle_ctx["gesture"] = "celebrate"
        elif board_check.is_check():
            battle_ctx["gesture"] = "nod"

    # PRE-MOVE gesture: "think" plays BEFORE the physical move
    if not is_simulation and battle_ctx and battle_ctx["gesture"] == "think":
        await _play_robot_gesture("think", max_duration=2.0)

    # Swap llm_service to the battle engine's service for commentary
    _battle_svc = None
    if move_source in ("llm2", "llm3"):
        _battle_svc = llm_service
        if move_source == "llm3" and llm_service_3:
            globals()['llm_service'] = llm_service_3
        elif move_source == "llm2" and llm_service_2:
            globals()['llm_service'] = llm_service_2

    # Start commentary streaming (LLM + TTS run in background)
    commentary_task = asyncio.create_task(_move_commentary(best_move, fen))

    if not is_simulation:
        # Execute move with robot WHILE commentary streams
        print(f"[Agent] Calling execute_robot_move({best_move}, ...)")
        robot_result = await execute_robot_move(best_move, fen)
        print(f"[Agent] execute_robot_move returned: success={robot_result.get('success')}, error={robot_result.get('error')}")

        if not robot_result.get("success"):
            print(f"[Agent] Robot move FAILED: {robot_result.get('error')}")
            commentary_task.cancel()
            current_game["last_robot_error"] = robot_result.get("error")
            await broadcast_game_event("robot_error", {
                "move": best_move, "error": robot_result.get("error"),
            })
            return

        # POST-MOVE gesture: celebrate/nod plays AFTER the physical move
        if battle_ctx and battle_ctx["gesture"] in ("celebrate", "nod"):
            await _play_robot_gesture(battle_ctx["gesture"], max_duration=2.0)

    # Update game state and tracker only on success
    board = chess.Board(fen)
    move = chess.Move.from_uci(best_move)
    board.push(move)
    new_fen = board.fen()

    tracker: BoardTracker = current_game.get("tracker")
    if tracker:
        tracker.push_uci(best_move)
        # Use tracker FEN as authoritative
        new_fen = tracker.fen

    current_game["fen"] = new_fen
    current_game["last_robot_move"] = best_move
    current_game["last_robot_error"] = None
    current_game["whose_turn"] = "human"
    current_game["move_number"] += 1

    if not is_simulation:
        # --- Post-move verification ---
        verification = await verify_robot_move(best_move)
        if not verification["verified"] and verification["error"] is None:
            print(f"[Agent] WARNING: Move {best_move} verification failed — board may need correction")
            await broadcast_game_event("move_verification_warning", {
                "move": best_move, "discrepancies": verification["discrepancies"],
            })

    # Persist state after every robot move
    _save_game_state()

    # Check game status
    status = get_game_status(new_fen)
    if status != "playing":
        current_game["status"] = "ended"
        current_game["game_result"] = status
        _save_game_state()
        if not is_simulation:
            # Physical: wait for commentary before game-over speech
            try:
                await commentary_task
            except Exception:
                pass
        asyncio.ensure_future(_game_over_speech(status, best_move, fen, who_won="robot"))
    else:
        # Broadcast robot_done immediately (don't block on TTS in simulation)
        await broadcast_game_event("robot_done", {
            "last_robot_move": best_move,
            "game_state": create_game_state().dict(),
        })

        if not is_simulation:
            # Physical: wait for commentary before starting auto-detect
            try:
                await commentary_task
            except Exception:
                pass
            await _start_auto_detect(already_at_vision=True)

    # Restore llm_service if swapped for battle mode
    if _battle_svc is not None:
        globals()['llm_service'] = _battle_svc


async def _watch_auto_play():
    """Auto-play loop for Watch mode (AI vs AI)."""
    global current_game, current_character, _llm_generation_id

    try:
        await asyncio.sleep(2.0)  # Let welcome speech finish

        while current_game and current_game.get("status") == "playing" and current_game.get("game_mode") == "watch":
            fen = current_game["fen"]
            board = chess.Board(fen)
            is_white_turn = board.turn == chess.WHITE

            # Pick engine/character/difficulty for the current side
            engine = current_game["white_engine"] if is_white_turn else current_game["black_engine"]
            difficulty = current_game["white_difficulty"] if is_white_turn else current_game["black_difficulty"]
            character = current_game["white_character"] if is_white_turn else current_game["black_character"]
            depth = DIFFICULTY_DEPTH.get(difficulty, 5)
            side_label = "White" if is_white_turn else "Black"

            # Pick LLM service: "llm" = service 1, "llm2" = service 2, "llm3" = service 3
            svc = None
            svc_label = ""
            if engine in ("llm", "llm2", "llm3"):
                if engine == "llm3" and llm_service_3:
                    svc = llm_service_3
                elif engine == "llm2" and llm_service_2:
                    svc = llm_service_2
                else:
                    svc = llm_service
                svc_label = f", model={svc.model}" if svc else ""
            print(f"[Watch] {side_label} turn: engine={engine}, difficulty={difficulty}{svc_label}")

            # Get move
            try:
                if engine in ("llm", "llm2", "llm3"):
                    best_move = await get_llm_move(fen, difficulty, character, service_override=svc)
                else:
                    best_move = get_stockfish_move(fen, depth)
            except Exception as e:
                print(f"[Watch] Engine error: {e}")
                break

            if best_move is None:
                current_game["status"] = "ended"
                current_game["game_result"] = "no_moves"
                _save_game_state()
                break

            print(f"[Watch] {side_label} plays: {best_move}")

            # Broadcast which side is speaking (so frontend labels the chat correctly)
            model_name = svc.model if svc else "Stockfish"
            await broadcast_game_event("watch_side", {
                "side": side_label,
                "model": model_name,
                "engine": engine,
            })

            # Generate commentary with this side's character AND LLM service
            old_character = current_character
            old_llm = llm_service
            current_character = character or old_character
            # Swap llm_service so commentary uses the correct side's model
            if svc and svc is not llm_service:
                globals()['llm_service'] = svc
            _llm_generation_id += 1
            commentary_task = asyncio.create_task(_move_commentary(best_move, fen))

            # Apply move
            tracker = current_game.get("tracker")
            if tracker:
                tracker.push_uci(best_move)
                new_fen = tracker.fen
            else:
                board.push(chess.Move.from_uci(best_move))
                new_fen = board.fen()

            # Update state (reuse human/robot fields for white/black)
            if is_white_turn:
                current_game["last_human_move"] = best_move
            else:
                current_game["last_robot_move"] = best_move
            current_game["fen"] = new_fen
            current_game["move_number"] += 1
            _save_game_state()

            # Broadcast
            await broadcast_game_event("robot_done", {
                "last_robot_move": best_move,
                "game_state": create_game_state().dict(),
            })

            # Always wait for commentary to finish before restoring character + service
            try:
                await commentary_task
            except Exception:
                pass
            current_character = old_character
            globals()['llm_service'] = old_llm

            # Check game over (pass tracker board for repetition/50-move detection)
            tracker_board = tracker._board if tracker else None
            status = get_game_status(new_fen, board_with_history=tracker_board)
            if status != "playing":
                current_game["status"] = "ended"
                current_game["game_result"] = status
                _save_game_state()
                # The side that just moved is the winner — set their character for speech
                who_won = "human" if is_white_turn else "robot"  # human=white, robot=black
                winner_char = current_game.get("white_character" if is_white_turn else "black_character", "")
                current_character = winner_char or current_character
                asyncio.ensure_future(_game_over_speech(status, best_move, fen, who_won=who_won))
                break

            move_delay = current_game.get("move_delay", 3.0)
            await asyncio.sleep(move_delay)

    except asyncio.CancelledError:
        print("[Watch] Auto-play cancelled")
    except Exception as e:
        import traceback
        print(f"[Watch] Auto-play error: {e}")
        traceback.print_exc()


async def _teach_demo_move(expected_move: str):
    """Robot traces the expected move path: hover source → hover destination.

    Gives the student a visual hint of which piece to move and where.
    For castling (e1g1/e1c1), also shows the rook path.
    Called sequentially during the demo phase (no initial delay).
    """
    if not expected_move or len(expected_move) < 4:
        return

    is_simulation = current_game and current_game.get("simulation", False)
    source_sq = expected_move[:2]
    dest_sq = expected_move[2:4]

    # Build highlight squares list (source + destination, plus rook for castling)
    highlight_squares = [source_sq, dest_sq]
    if expected_move == "e1g1":
        highlight_squares.extend(["h1", "f1"])
    elif expected_move == "e1c1":
        highlight_squares.extend(["a1", "d1"])

    if is_simulation:
        # Simulation: animate the demo move on the GUI chessboard
        # Step 1: highlight source square (piece to move)
        await broadcast_game_event("teach_demo_anim", {
            "phase": "source",
            "from": source_sq,
            "to": dest_sq,
            "squares": highlight_squares,
        })
        await asyncio.sleep(1.5)
        # Step 2: animate piece sliding from source to destination
        await broadcast_game_event("teach_demo_anim", {
            "phase": "move",
            "from": source_sq,
            "to": dest_sq,
            "squares": highlight_squares,
        })
        await asyncio.sleep(1.5)
        # Step 3: for castling, also animate the rook
        if expected_move in ("e1g1", "e1c1"):
            rook_from = "h1" if expected_move == "e1g1" else "a1"
            rook_to = "f1" if expected_move == "e1g1" else "d1"
            await broadcast_game_event("teach_demo_anim", {
                "phase": "move",
                "from": rook_from,
                "to": rook_to,
                "squares": highlight_squares,
            })
            await asyncio.sleep(1.0)
        # Step 4: reset board back (demo is conceptual, not actual)
        await broadcast_game_event("teach_demo_anim", {
            "phase": "reset",
            "from": source_sq,
            "to": dest_sq,
            "squares": highlight_squares,
        })
        # Keep highlight squares visible for student reference
        await broadcast_game_event("teach_highlight", {
            "squares": highlight_squares,
            "from": source_sq,
            "to": dest_sq,
        })
    else:
        # Physical: robot hovers + GUI highlights simultaneously
        print(f"[Teach] Demo move: {source_sq} → {dest_sq} (calling hover_robot_above_square)")

        # GUI: highlight source square
        await broadcast_game_event("teach_demo_anim", {
            "phase": "source", "from": source_sq, "to": dest_sq, "squares": highlight_squares,
        })
        result1 = await hover_robot_above_square(source_sq)
        print(f"[Teach] Hover {source_sq} result: {result1}")
        await asyncio.sleep(1.5)

        # GUI: animate piece slide + robot hover destination
        await broadcast_game_event("teach_demo_anim", {
            "phase": "move", "from": source_sq, "to": dest_sq, "squares": highlight_squares,
        })
        result2 = await hover_robot_above_square(dest_sq)
        print(f"[Teach] Hover {dest_sq} result: {result2}")
        await asyncio.sleep(1.5)

        # For castling, also show the rook's path
        if expected_move == "e1g1":
            await broadcast_game_event("teach_demo_anim", {
                "phase": "move", "from": "h1", "to": "f1", "squares": highlight_squares,
            })
            await hover_robot_above_square("h1")
            await asyncio.sleep(1.0)
            await hover_robot_above_square("f1")
            await asyncio.sleep(1.0)
        elif expected_move == "e1c1":
            await broadcast_game_event("teach_demo_anim", {
                "phase": "move", "from": "a1", "to": "d1", "squares": highlight_squares,
            })
            await hover_robot_above_square("a1")
            await asyncio.sleep(1.0)
            await hover_robot_above_square("d1")
            await asyncio.sleep(1.0)

        # GUI: reset animation, keep highlight for reference
        await broadcast_game_event("teach_demo_anim", {
            "phase": "reset", "from": source_sq, "to": dest_sq, "squares": highlight_squares,
        })
        await broadcast_game_event("teach_highlight", {
            "squares": highlight_squares, "from": source_sq, "to": dest_sq,
        })

        # Return to vision position (out of the way for camera + student)
        await move_robot_for_vision()


async def _teach_speak_phase(text: str, emotion: str = "encouraging", gesture: str = ""):
    """Speak a single teaching phase paragraph via LLM streaming or plain TTS."""
    gen_id = _llm_generation_id
    if llm_service:
        result = await _stream_text_with_tts(text, gen_id, emotion=emotion, gesture=gesture)
        if result:
            await broadcast_game_event("voice_text_done", {"text": result, "tts_streamed": True})
            return result
    # Fallback: speak the raw text (still fire gesture)
    if gesture and not (current_game and current_game.get("simulation", False)):
        asyncio.create_task(_play_robot_gesture(gesture))
    await broadcast_game_event("voice_text_chunk", {"chunk": text})
    await _speak_text_only(text, emotion=emotion)
    await broadcast_game_event("voice_text_done", {"text": text, "tts_streamed": True})
    return text


async def _teach_3phase_instruction(step, is_first: bool = False):
    """Run the opening phase for a step: one short instruction + demo move.

    Single LLM call that explains what to do + demo runs concurrently.
    Must complete under ~10 seconds so the full step stays under 20s.
    """
    zh = _is_zh()
    is_physical = current_game and not current_game.get("simulation", False)
    instruction = step.instruction_zh if zh and step.instruction_zh else step.instruction

    print(f"[Teach] Opening: step={_teach_step_idx}, expected_move={step.expected_move}")

    # Lock student input during opening & demo
    await broadcast_game_event("teach_input_lock", {"locked": True})

    # Start robot demo concurrently (4s delay built into _teach_demo_move)
    demo_task = None
    if step.expected_move:
        if is_physical:
            demo_task = asyncio.create_task(_teach_demo_move(step.expected_move))
        else:
            await _teach_demo_move(step.expected_move)

    # Single short speech: explain + invite student to try
    gen_id = _llm_generation_id
    if llm_service:
        prompt = (
            f"你是棋藝教練。引導學生這一步：「{instruction}」。"
            f"用3-5句話：先解釋這步棋的概念和意義，再告訴學生該怎麼做，最後鼓勵他們試試看。"
            f"用自然口語，不要使用棋譜符號。" if zh else
            f"You are a chess coach. Guide the student: '{instruction}'. "
            f"In 3-5 sentences: explain the concept and why this move matters, "
            f"tell them what to do, then encourage them to try. "
            f"Use natural spoken language, no chess notation."
        )
        # No gesture during opening — demo move is the physical action
        message = await _stream_text_with_tts(prompt, gen_id, emotion="encouraging")
        if not message:
            message = instruction
    else:
        message = instruction
        await broadcast_game_event("voice_text_chunk", {"chunk": message})
        await _speak_text_only(message, emotion="encouraging")

    await broadcast_game_event("voice_text_done", {"text": message, "tts_streamed": True})

    # Wait for robot demo to finish
    if demo_task:
        await demo_task

    # Signal unlock — in physical mode, JS waits for TTS audio to finish
    # In simulation mode, unlock immediately (no robot arm to wait for)
    is_sim = current_game and current_game.get("simulation", False)
    await broadcast_game_event("teach_input_lock", {"locked": False, "after_audio": not is_sim})


async def _teach_speak_first_instruction():
    """Speak the first step's instruction using the 3-phase flow."""
    try:
        # Small delay to let welcome speech finish
        await asyncio.sleep(2)

        step = _get_teach_step()
        if step is None:
            print("[Teach] _teach_speak_first_instruction: no step found, returning")
            await broadcast_game_event("teach_input_lock", {"locked": False})
            return

        # Ensure robot starts at vision position
        is_physical = current_game and not current_game.get("simulation", False)
        if is_physical:
            await move_robot_for_vision()

        print(f"[Teach] First instruction: expected_move={step.expected_move}")
        await _teach_3phase_instruction(step, is_first=True)

        # Now start auto-detect — all 3 phases done, watch for student's move
        if is_physical:
            await _start_auto_detect()
    except Exception as e:
        import traceback
        print(f"[Teach] _teach_speak_first_instruction CRASHED: {e}")
        traceback.print_exc()
        # Always unlock on crash so student isn't stuck
        await broadcast_game_event("teach_input_lock", {"locked": False})


async def _teach_wrap_up(step: LessonStep, uci_move: str) -> str:
    """After correct move: congrats → opponent move + explanation → ready for next?

    Single flow that handles conclusion + opponent reply in natural order.
    Returns the full commentary text.
    """
    zh = _is_zh()
    explanation = step.explanation_zh if zh and step.explanation_zh else step.explanation
    gen_id = _llm_generation_id
    is_simulation = current_game.get("simulation", False)
    move_desc = describe_move(uci_move, step.fen)

    # --- Determine opponent's reply move (if any) ---
    black_move_uci = None
    black_move_desc = ""
    next_idx = _teach_step_idx + 1
    has_next = _teach_lesson is not None and next_idx < len(_teach_lesson.steps)
    next_step = _teach_lesson.steps[next_idx] if has_next else None

    if next_step:
        current_fen = current_game.get("fen", "")
        try:
            board = chess.Board(current_fen)
            target_board = chess.Board(next_step.fen)
            for move in board.legal_moves:
                board.push(move)
                if board.board_fen() == target_board.board_fen():
                    black_move_uci = move.uci()
                    board.pop()
                    break
                board.pop()
        except Exception as e:
            print(f"[Teach] Could not determine Black's reply: {e}")

        if black_move_uci:
            black_move_desc = describe_move(black_move_uci, current_fen)

    # --- Single LLM call: congrats + opponent move + ready? ---
    if llm_service:
        if black_move_uci:
            prompt = (
                f"學生走對了：{move_desc}。{explanation} "
                f"接下來對手走了{black_move_desc}。"
                f"用3-4句話，按順序：1)簡短恭喜學生 2)說明現在對手走了什麼、為什麼 3)問學生準備好下一步了嗎。"
                f"用自然口語，不要棋譜符號。" if zh else
                f"Correct: {move_desc}. {explanation} "
                f"The opponent replies with {black_move_desc}. "
                f"In 3-4 sentences in order: 1) briefly congratulate 2) explain the opponent's move and why "
                f"3) ask if ready for next step. Natural spoken language, no notation."
            )
        else:
            # Last step — no opponent reply
            prompt = (
                f"學生走對了：{move_desc}。{explanation} "
                f"用2句話：簡短恭喜、解釋這步學到了什麼。"
                f"用自然口語，不要棋譜符號。" if zh else
                f"Correct: {move_desc}. {explanation} "
                f"In 2 sentences: briefly congratulate and explain what they learned. "
                f"Natural spoken language, no notation."
            )
        # No gesture during teach wrap-up — physical opponent move follows immediately
        commentary = await _stream_text_with_tts(prompt, gen_id, emotion="happy")
        if not commentary:
            commentary = explanation
    else:
        commentary = explanation
        await broadcast_game_event("voice_text_chunk", {"chunk": commentary})
        await _speak_text_only(commentary, emotion="happy")

    await broadcast_game_event("voice_text_done", {"text": commentary, "tts_streamed": True})

    if conversation_context:
        conversation_context.add_exchange(f"(student played {uci_move})", commentary)

    # --- Execute opponent's physical/sim move ---
    if next_step:
        if not is_simulation:
            await broadcast_game_event("teach_robot_moving", {
                "message": "Teacher is making the opponent's move...",
            })
            await _setup_board_position(next_step.fen)
            await move_robot_for_vision()

        # Update FEN and tracker to next step's position
        current_game["fen"] = next_step.fen
        current_game["tracker"] = BoardTracker(fen=next_step.fen)

    return commentary


async def _handle_teach_auto_detect(detected_move: str, tracker):
    """Handle a detected move during teach mode auto-detect.

    Checks the move against the current lesson step's expected move.
    Flow: correct → robot plays Black reply → broadcast conclusion.
    """
    step = _get_teach_step()
    if step is None:
        return

    zh = _is_zh()
    board = chess.Board(step.fen)

    try:
        move_obj = chess.Move.from_uci(detected_move)
    except ValueError:
        await broadcast_game_event("teach_move_wrong", {
            "detected_move": detected_move,
            "message": f"Invalid move detected: {detected_move}",
        })
        return

    if move_obj not in board.legal_moves:
        await broadcast_game_event("teach_move_wrong", {
            "detected_move": detected_move,
            "message": "Detected move is not legal. Make sure pieces are placed correctly.",
        })
        return

    # Check against expected
    if step.expected_move is None or detected_move == step.expected_move:
        # Apply student's move to tracker and update FEN
        tracker.push_uci(detected_move)
        current_game["fen"] = tracker.fen

        # Wrap up: congrats → opponent move + why → ready?
        explanation = step.explanation_zh if zh and step.explanation_zh else step.explanation
        commentary = await _teach_wrap_up(step, detected_move)

        await broadcast_game_event("teach_step_done", {
            "detected_move": detected_move,
            "message": commentary,
            "explanation": explanation,
            "game_state": create_game_state().dict(),
        })
    else:
        # Wrong move — guide student, DON'T update tracker (gesture: shake)
        move_desc = describe_move(detected_move, step.fen)
        if not (current_game and current_game.get("simulation", False)):
            asyncio.create_task(_play_robot_gesture("shake", return_to_vision=True))
        if llm_service:
            prompt = (
                f"學生走了{move_desc}，不對。用1句話提示再試，提醒放回棋子。最多10個字。不要說答案。" if zh else
                f"Student played {move_desc}, wrong. Hint to retry, remind to put piece back. Max 10 words. Don't reveal answer."
            )
            commentary = await get_llm_response(prompt)
        else:
            commentary = ("Not quite right. Put the piece back and try again!"
                          if not zh else "不太對喔，把棋子放回去再試一次！")

        await broadcast_game_event("teach_move_wrong", {
            "detected_move": detected_move,
            "message": commentary,
        })
        # Restart auto-detect to keep watching
        await _start_auto_detect(already_at_vision=True)


async def _start_auto_detect(already_at_vision: bool = False):
    """Start the auto-detect background loop (replaces old 15s timer).

    Awaits full cleanup of the old task (including its finally block robot
    movement) before launching a new one — prevents TOCTOU race.
    """
    global _auto_detect_task
    # Await full cleanup of existing task (including finally block)
    if _auto_detect_task and not _auto_detect_task.done():
        _auto_detect_task.cancel()
        try:
            await _auto_detect_task
        except asyncio.CancelledError:
            pass
    _auto_detect_task = asyncio.create_task(_auto_detect_loop(already_at_vision))
    print("[Auto-detect] Started")


async def _cancel_auto_detect():
    """Cancel the running auto-detect loop."""
    global _auto_detect_task
    if _auto_detect_task and not _auto_detect_task.done():
        _auto_detect_task.cancel()
        try:
            await _auto_detect_task
        except asyncio.CancelledError:
            pass
        print("[Auto-detect] Cancelled")
    _auto_detect_task = None


async def _auto_detect_loop(already_at_vision: bool = False):
    """
    Background loop: poll vision every ~0.5s to auto-detect the human's move.

    Flow:
    - Initial 1s delay (commentary already streaming concurrently)
    - Move robot to vision position ONCE (skipped if already_at_vision)
    - Then every 2s: capture occupancy in-place (no robot movement), compare with tracker baseline
    - Stability check: if same diff detected 2 consecutive polls → trigger
    - On move detected: move robot back to work, execute robot move
    - On exit/cancel: ensure robot returns to work position
    - Cancelled by: human_moved button, voice "your_turn", game stop/pause
    """
    global current_game

    _at_vision = already_at_vision  # Track whether we moved to vision (for cleanup)

    try:
        # Initial delay — give human time to process
        await asyncio.sleep(1)

        # Move robot to vision position ONCE at the start (skip if verify already did it)
        if not already_at_vision:
            await move_robot_for_vision()
        _at_vision = True

        prev_diff = None  # Track diff from previous poll for stability
        poll_count = 0

        while True:
            # Check preconditions
            if current_game is None or current_game.get("status") != "playing":
                print("[Auto-detect] Game not active, stopping")
                return
            if current_game.get("whose_turn") != "human":
                print("[Auto-detect] Not human's turn, stopping")
                return

            poll_count += 1
            is_teach = current_mode == GameMode.TEACH
            print(f"[Auto-detect] Poll #{poll_count}" + (" (teach)" if is_teach else ""))

            # Broadcast detecting event
            event_type = "teach_detecting" if is_teach else "detecting_move"
            await broadcast_game_event(event_type, {"poll": poll_count})

            try:
                vision_result = await _capture_occupancy_in_place()
            except Exception as e:
                print(f"[Auto-detect] Vision error: {e}")
                await asyncio.sleep(0.5)
                continue

            if not vision_result.get("success"):
                print(f"[Auto-detect] Vision failed: {vision_result.get('error')}")
                await asyncio.sleep(0.5)
                continue

            tracker: BoardTracker = current_game.get("tracker")
            if tracker is None:
                print("[Auto-detect] No tracker, stopping")
                return

            baseline = tracker.get_occupancy()
            curr_occupancy = set(vision_result.get("occupied_squares", []))

            # Compute diff
            curr_diff = (baseline - curr_occupancy, curr_occupancy - baseline)
            has_change = curr_diff[0] or curr_diff[1]

            if not has_change:
                # No change yet — keep polling
                prev_diff = None
                await asyncio.sleep(0.5)
                continue

            # There is a change — check stability (same diff as last poll)
            if prev_diff is not None and curr_diff == prev_diff:
                # Stable change detected — try to detect the move
                print(f"[Auto-detect] Stable change detected: vacated={curr_diff[0]}, new={curr_diff[1]}")

                # Build vision_pieces for capture tiebreaking
                vision_pieces = vision_result.get("piece_positions", {})

                detection = tracker.detect_human_move(
                    prev_occupancy=baseline,
                    curr_occupancy=curr_occupancy,
                    vision_pieces=vision_pieces if vision_pieces else None,
                )

                if detection.success:
                    detected_move = detection.uci_move
                    print(f"[Auto-detect] Human played: {detected_move}")

                    if is_teach:
                        # --- Teach mode: check against lesson expected move ---
                        await _handle_teach_auto_detect(detected_move, tracker)
                        # Stop polling — student clicks Next Step to continue
                        return

                    # --- Battle mode: apply move and make robot respond ---
                    # Robot stays at vision — execute_robot_move handles
                    # its own work positioning via manual_pick/place
                    _at_vision = False

                    # Apply move
                    async with game_lock:
                        fen_before_move = tracker.fen
                        tracker.push_uci(detected_move)
                        new_fen = tracker.fen
                        current_game["fen"] = new_fen
                        current_game["last_human_move"] = detected_move
                        current_game["whose_turn"] = "robot"

                        # Persist after human move
                        _save_game_state()

                        # Broadcast move detected
                        await broadcast_game_event("move_detected", {
                            "move": detected_move,
                            "game_state": create_game_state().dict(),
                        })

                        # Check game status
                        status = get_game_status(new_fen)
                        if status != "playing":
                            current_game["status"] = "ended"
                            current_game["game_result"] = status
                            _save_game_state()
                            asyncio.ensure_future(_game_over_speech(status, detected_move, fen_before_move, who_won="human"))
                            return

                        # Make robot's move (this will start a new auto-detect loop)
                        await make_robot_move()

                    # make_robot_move starts a new loop, so exit this one
                    return
                else:
                    print(f"[Auto-detect] Detection failed: {detection.error}")
                    await broadcast_game_event("detection_failed", {
                        "error": detection.error,
                        "candidates": detection.candidates or [],
                    })
                    # Reset stability — maybe human is still moving
                    prev_diff = None
                    await asyncio.sleep(0.5)
                    continue
            else:
                # First time seeing this change — record and wait for confirmation
                prev_diff = curr_diff
                print(f"[Auto-detect] Change seen, waiting for stability confirmation")
                await asyncio.sleep(0.5)
                continue

    except asyncio.CancelledError:
        print("[Auto-detect] Loop cancelled")
        # Robot stays at vision — no need to move to work.
        # stop_game goodbye handles home→sleep; human_moved/your_turn
        # will take over from vision position.
        raise
    finally:
        pass  # Robot stays at vision position


@app.get("/game/state", response_model=AgentResponse)
async def get_game_state():
    """Get current game state."""
    if current_game is None:
        return AgentResponse(
            success=False,
            error="No active game",
        )

    return AgentResponse(
        success=True,
        game_state=create_game_state(),
    )


@app.post("/game/stop", response_model=AgentResponse)
async def stop_game():
    """Stop current game, say goodbye, and send robot to sleep."""
    global current_game, _game_stopping
    global _teach_lesson, _teach_step_idx, _teach_hint_idx

    # Clean up teach state
    _teach_lesson = None
    _teach_step_idx = 0
    _teach_hint_idx = 0

    # Cancel watch mode auto-play
    global _watch_task
    if _watch_task and not _watch_task.done():
        _watch_task.cancel()
        try:
            await _watch_task
        except asyncio.CancelledError:
            pass
    _watch_task = None

    # Signal auto-detect to skip cleanup robot movement (goodbye handles it)
    _game_stopping = True
    await _cancel_auto_detect()
    _game_stopping = False

    async with game_lock:
        if current_game is None:
            return AgentResponse(success=False, error="No active game")

        was_simulation = current_game.get("simulation", False)
        current_game["status"] = "ended"
        current_game["game_result"] = "stopped"

        game_state = create_game_state()
        current_game = None

        # Clear persisted state
        _save_game_state()

    # Goodbye speech + robot shutdown in background (don't block the response)
    asyncio.ensure_future(_goodbye_and_sleep(simulation=was_simulation))

    return AgentResponse(
        success=True,
        message="Game stopped",
        game_state=game_state,
    )


async def _game_over_speech(status: str, last_move: str, fen_before: str, who_won: str = "human"):
    """Generate dramatic game-over commentary (5-6 sentences, in character)."""
    global _llm_generation_id
    try:
        # Wait briefly for any ongoing commentary to finish
        await asyncio.sleep(1.0)
        # Claim a fresh generation ID so game-over speech is never skipped
        _llm_generation_id += 1
        is_simulation = current_game and current_game.get("simulation", False)
        move_desc = describe_move(last_move, fen_before)
        gen_id = _llm_generation_id
        zh = _is_zh()

        # Determine result description
        is_watch = current_game and current_game.get("game_mode") == "watch"
        if status == "checkmate":
            if is_watch:
                winner_side = "白方" if who_won == "human" else "黑方"
                winner_side_en = "White" if who_won == "human" else "Black"
                result_desc = f"{winner_side}將殺了" if zh else f"{winner_side_en} checkmated"
                result_desc_full = (
                    f"{winner_side}走了{move_desc}，將殺了對手！" if zh else
                    f"{winner_side_en} played {move_desc} and delivered checkmate!"
                )
            elif who_won == "human":
                result_desc = "被將殺了" if zh else "got checkmated"
                result_desc_full = f"對手走了{move_desc}，把你將殺了" if zh else f"The opponent played {move_desc} and checkmated you"
            else:
                result_desc = "將殺對手了" if zh else "checkmated the opponent"
                result_desc_full = f"你走了{move_desc}，將殺了對手" if zh else f"You played {move_desc} and checkmated the opponent"
        elif status == "stalemate":
            result_desc = "和棋（逼和）" if zh else "stalemate (draw)"
            result_desc_full = f"走了{move_desc}之後，局面變成逼和" if zh else f"After {move_desc}, the position is a stalemate"
        elif status == "draw":
            result_desc = "和棋" if zh else "draw"
            result_desc_full = f"走了{move_desc}之後，比賽和棋" if zh else f"After {move_desc}, the game is a draw"
        else:
            result_desc = status
            result_desc_full = f"遊戲結束：{status}" if zh else f"Game over: {status}"

        # Determine emotion and gesture
        if is_watch:
            emotion = "happy"
            gesture = "celebrate"
        elif who_won == "human":
            emotion = "sad"
            gesture = "sad"
        else:
            emotion = "happy"
            gesture = "celebrate"

        if llm_service:
            if zh:
                if is_watch:
                    winner_side_zh = "白方" if who_won == "human" else "黑方"
                    loser_side_zh = "黑方" if who_won == "human" else "白方"
                    if who_won == "human":
                        # White won — speak as the winner (white's character)
                        prompt = (
                            f"{result_desc_full}！你是{winner_side_zh}。"
                            f"用你的個性風格，3-4句話慶祝勝利。"
                            f"回顧你的精彩棋步，嘲笑或讚美對手，展現你的態度！"
                            f"要有感情、有個性！用自然口語。"
                        )
                    else:
                        prompt = (
                            f"{result_desc_full}！你是{winner_side_zh}。"
                            f"用你的個性風格，3-4句話慶祝勝利。"
                            f"回顧你的精彩棋步，嘲笑或讚美對手，展現你的態度！"
                            f"要有感情、有個性！用自然口語。"
                        )
                else:
                    prompt = (
                        f"{result_desc_full}！"
                        f"用你的個性風格，5-6句話回應這場比賽的結束。"
                        f"{'承認對手贏了，表達你的感受，回顧這場比賽的精彩之處，讚美對手，說你想再來一局。' if who_won == 'human' else '慶祝勝利，回顧精彩的棋步，讚美對手的表現，鼓勵他們繼續挑戰。'}"
                        f"要有感情、有個性！用自然口語。"
                    )
            else:
                if is_watch:
                    winner_side_en = "White" if who_won == "human" else "Black"
                    prompt = (
                        f"{result_desc_full}! You are {winner_side_en}. "
                        f"In 3-4 sentences, celebrate your victory in character. "
                        f"Reflect on your brilliant moves, taunt or compliment the opponent, show your attitude! "
                        f"Be emotional and in character!"
                    )
                else:
                    prompt = (
                        f"{result_desc_full}! "
                        f"React to the end of the game in 5-6 sentences, staying in character. "
                        f"{'Acknowledge the loss, express your feelings, reflect on highlights, compliment the opponent, suggest a rematch.' if who_won == 'human' else 'Celebrate the win, reflect on key moments, compliment the opponent, encourage them to try again.'} "
                        f"Be emotional and in character! Use natural spoken language."
                    )
            commentary = await _stream_text_with_tts(prompt, gen_id, emotion=emotion, gesture=gesture)
            if not commentary:
                commentary = result_desc_full
        else:
            commentary = "好棋！再來一局吧！" if zh else "Good game! Let's play again!"
            if not is_simulation:
                asyncio.create_task(_play_robot_gesture(gesture, return_to_vision=True))
            await broadcast_game_event("voice_text_chunk", {"chunk": commentary})
            await _speak_text_only(commentary, emotion=emotion)

        await broadcast_game_event("voice_text_done", {"text": commentary, "tts_streamed": True})

        if conversation_context:
            conversation_context.add_exchange(f"(game over: {status})", commentary)

        print(f"[Agent] Game over speech: {commentary}")
    except Exception as e:
        import traceback
        print(f"[Agent] Game over speech error: {e}")
        traceback.print_exc()


async def _goodbye_and_sleep(simulation: bool = False):
    """Say goodbye, move robot to home then sleep."""
    try:
        # Generate and stream goodbye speech + TTS in sync
        is_teach = current_mode == GameMode.TEACH
        gen_id = _llm_generation_id

        if llm_service:
            if _is_zh():
                if is_teach:
                    prompt = "遊戲結束了。用你的教練個性說一段簡短的告別詞，鼓勵對方。1-2句話。"
                else:
                    prompt = "遊戲結束了。用你的個性風格說一句告別的話，要簡短有趣！最多1句。"
            else:
                if is_teach:
                    prompt = "The game is over. Say a brief goodbye in character, encourage the player. 1-2 sentences."
                else:
                    prompt = "Game over. Say goodbye in 1 short sentence. Stay in character, be brief!"

            goodbye_emotion = "encouraging" if is_teach else resolve_character_emotion(current_character, "goodbye")
            full_text = await _stream_text_with_tts(prompt, gen_id, emotion=goodbye_emotion)
            goodbye_text = full_text if full_text else "Good game! See you next time."
        else:
            goodbye_emotion = "encouraging" if is_teach else resolve_character_emotion(current_character, "goodbye")
            goodbye_text = "Good game! See you next time." if current_language == "en" else "好棋！下次再見！"
            await broadcast_game_event("voice_text_chunk", {"chunk": goodbye_text})
            await _speak_text_only(goodbye_text, emotion=goodbye_emotion)

        await broadcast_game_event("voice_text_done", {"text": goodbye_text, "tts_streamed": True})

        if conversation_context:
            conversation_context.add_exchange("(game ended)", goodbye_text)

        print(f"[Agent] Goodbye speech: {goodbye_text}")
    except Exception as e:
        print(f"[Agent] Goodbye speech error: {e}")

    # Move robot to home then sleep (skip in simulation mode)
    if not simulation:
        try:
            await _robot_post("/arm/home", timeout=10.0)
            print("[Agent] Robot moved to home")
            await asyncio.sleep(1.5)
            await _robot_post("/arm/sleep", timeout=10.0)
            print("[Agent] Robot moved to sleep")
        except Exception as e:
            print(f"[Agent] Robot shutdown error: {e}")


# =============================================================================
# Pause / Resume (for LangGraph agent state machine)
# =============================================================================

# The LangGraph agent can pause (instead of blocking on input()).
# These globals mirror the agent's pause state for the REST API.
_agent_paused = False
_agent_pause_reason = None  # type: Optional[str]
_agent_pause_options = None  # type: Optional[list]


@app.get("/game/pause_status")
async def get_pause_status():
    """Get current pause status of the LangGraph agent."""
    return {
        "paused": _agent_paused,
        "pause_reason": _agent_pause_reason,
        "pause_options": _agent_pause_options,
    }


class ResumeRequest(BaseModel):
    """Resume request with chosen action."""
    action: str = "continue"  # e.g. "retry", "skip", "abort", "continue"


@app.post("/game/resume")
async def resume_game(request: ResumeRequest):
    """
    Resume a paused LangGraph agent.

    The action string is passed into the agent state as user_action,
    then the graph is re-invoked from its paused phase.
    """
    global _agent_paused, _agent_pause_reason, _agent_pause_options

    if not _agent_paused:
        return {"success": False, "error": "Agent is not paused"}

    valid_actions = _agent_pause_options or ["continue"]
    if request.action not in valid_actions:
        return {
            "success": False,
            "error": f"Invalid action '{request.action}'. Valid: {valid_actions}",
        }

    _agent_paused = False
    _agent_pause_reason = None
    _agent_pause_options = None

    return {
        "success": True,
        "message": f"Resumed with action: {request.action}",
        "action": request.action,
    }


# =============================================================================
# WebSocket /ws/game — push game events to connected UI clients
# =============================================================================

import json as _json

_game_ws_clients: set = set()


async def broadcast_game_event(event_type: str, data: dict = None):
    """Push an event to all connected /ws/game clients."""
    msg = _json.dumps({"type": event_type, "data": data or {}})
    dead = set()
    for ws in _game_ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    _game_ws_clients.difference_update(dead)


@app.websocket("/ws/game")
async def game_websocket(websocket: WebSocket):
    """
    WebSocket for real-time game events.

    Pushes events:
        {"type": "game_state",   "data": {...}}
        {"type": "robot_status", "data": {"status": "moving"|"idle"|"error", ...}}
        {"type": "error",        "data": {"message": "...", "pause_reason": "..."}}
        {"type": "pause",        "data": {"paused": true, "reason": "...", "options": [...]}}
    """
    await websocket.accept()
    _game_ws_clients.add(websocket)
    try:
        # Send current state on connect
        gs = create_game_state()
        if gs:
            await websocket.send_text(_json.dumps({
                "type": "game_state",
                "data": gs.dict(),
            }))
        await websocket.send_text(_json.dumps({
            "type": "pause",
            "data": {
                "paused": _agent_paused,
                "reason": _agent_pause_reason,
                "options": _agent_pause_options,
            },
        }))
        # Keep connection alive — read pings/messages until disconnect
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        _game_ws_clients.discard(websocket)


@app.get("/game/tracker")
async def get_tracker_state():
    """Get the internal board tracker state for debugging."""
    if current_game is None:
        raise HTTPException(status_code=400, detail="No active game")

    tracker: BoardTracker = current_game.get("tracker")
    if tracker is None:
        return {"error": "No tracker available"}

    return {
        "fen": tracker.fen,
        "turn": tracker.turn,
        "piece_map": tracker.get_piece_map(),
        "occupied_squares": sorted(tracker.get_occupancy()),
        "captured_pieces": tracker._captured_pieces,
        "move_count": len(tracker.board.move_stack),
        "move_history": [m.uci() for m in tracker.board.move_stack],
        "serialized": tracker.to_dict(),
    }


@app.post("/game/robot_move", response_model=AgentResponse)
async def trigger_robot_move():
    """Manually trigger robot's move (for testing)."""
    global current_game

    if current_game is None:
        raise HTTPException(status_code=400, detail="No active game")

    if current_game["whose_turn"] != "robot":
        raise HTTPException(status_code=400, detail="Not robot's turn")

    async with game_lock:
        await make_robot_move()

        return AgentResponse(
            success=True,
            message=f"Robot played: {current_game.get('last_robot_move')}",
            game_state=create_game_state(),
        )


# =============================================================================
# Voice Endpoints
# =============================================================================


def parse_spoken_move(spoken: str, board: chess.Board) -> Optional[str]:
    """
    Parse natural language chess move to UCI format.

    Examples:
        "pawn to e4" -> "e2e4"
        "knight f3" -> "g1f3"
        "castle kingside" -> "e1g1"
    """
    import re

    spoken_lower = spoken.lower().strip()

    # Castling
    if "castle" in spoken_lower or "castling" in spoken_lower:
        if "king" in spoken_lower or "short" in spoken_lower:
            return "e1g1" if board.turn == chess.WHITE else "e8g8"
        elif "queen" in spoken_lower or "long" in spoken_lower:
            return "e1c1" if board.turn == chess.WHITE else "e8c8"

    # Extract squares mentioned
    squares = re.findall(r"[a-h][1-8]", spoken_lower)

    # Piece names
    piece_map = {
        "king": chess.KING,
        "queen": chess.QUEEN,
        "rook": chess.ROOK,
        "bishop": chess.BISHOP,
        "knight": chess.KNIGHT,
        "pawn": chess.PAWN,
        "horse": chess.KNIGHT,  # Common alternative
    }

    piece_type = None
    for name, ptype in piece_map.items():
        if name in spoken_lower:
            piece_type = ptype
            break

    # Try to find matching legal move
    for move in board.legal_moves:
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)
        piece = board.piece_at(move.from_square)

        # Check if move matches spoken description
        if len(squares) >= 1:
            if squares[-1] == to_sq:  # Destination matches
                if len(squares) == 2 and squares[0] != from_sq:
                    continue
                if piece_type and piece.piece_type != piece_type:
                    continue
                return move.uci()

        # Piece-only description (e.g., "knight f3")
        if piece_type and piece.piece_type == piece_type:
            if to_sq in spoken_lower:
                return move.uci()

    return None


def describe_move(uci_move: str, fen: str) -> str:
    """Generate natural language description of a move."""
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
    to_sq = chess.square_name(move.to_square)

    if board.is_capture(move):
        return f"{piece_name} takes on {to_sq}"
    elif board.is_castling(move):
        return "castle kingside" if move.to_square > move.from_square else "castle queenside"
    else:
        return f"{piece_name} to {to_sq}"


def _get_current_board():
    """Get current chess.Board from game state (for intent router)."""
    if current_game is None:
        return None
    try:
        return chess.Board(current_game["fen"])
    except Exception:
        return None


async def _voice_event_loop():
    """Background task that drains the voice event queue and processes events."""
    while True:
        try:
            await asyncio.sleep(0.1)  # 100ms poll interval

            if voice_event_queue is None or voice_event_processor is None:
                continue

            if voice_event_queue.pending == 0:
                continue

            events = await voice_event_queue.get_all()

            for event in events:
                response = await voice_event_processor.process(event)

                if response and response.get("text"):
                    # Broadcast response to voice WebSocket + chat
                    await _broadcast_voice_response(event, response)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[Voice Event Loop] Error: {e}")
            await asyncio.sleep(1.0)


async def _broadcast_voice_response(event: VoiceEvent, response: dict):
    """Send response to active voice WebSocket and broadcast chat.

    Streams text chunks to game WS and TTS audio via *both* game WS
    (for text+audio sync) and voice WS (for echo suppression / headset).
    """
    import base64 as _b64

    ws = voice_handler.active_websocket if voice_handler else None
    response_voice = response.get("voice", voice_handler.current_voice if voice_handler else "alloy")
    response_text = response.get("text", "")

    # Broadcast transcription to game WS
    if event.text:
        await broadcast_game_event("voice_response", {
            "transcription": event.text,
            "intent": event.intent,
        })

    # Stream text chunks to game WS for incremental display
    if response_text:
        words = response_text.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == 0 else " " + word
            await broadcast_game_event("voice_text_chunk", {"chunk": chunk})

    gen_id = _llm_generation_id  # Capture for interruption check

    # Send agent_response to voice WS (for transcription panel)
    if ws:
        try:
            await ws.send_json({
                "type": "agent_response",
                "text": response_text,
                "move_detected": response.get("move"),
                "game_state": response.get("game_state"),
                "voice": response_voice,
            })
        except Exception as e:
            print(f"[Voice Event Loop] WebSocket send error: {e}")

    # Skip TTS for move responses — _move_commentary already handles TTS streaming
    is_move_response = response.get("move") is not None
    # Generate TTS and broadcast via game WS (unified audio channel)
    if tts_service and response_text and not is_move_response:
        if ws:
            try:
                await ws.send_json({"type": "voice_status", "status": "speaking"})
            except Exception:
                pass

        try:
            audio_bytes = await tts_service.synthesize_async(
                response_text, response_format="mp3"
            )
            if _llm_generation_id == gen_id and audio_bytes:
                audio_b64 = _b64.b64encode(audio_bytes).decode("utf-8")
                await broadcast_game_event("voice_audio_chunk", {
                    "data": audio_b64,
                    "format": "mp3",
                })
                await broadcast_game_event("voice_audio_done", {})
                print(f"[Voice] TTS audio sent: {len(audio_bytes)} bytes")
            else:
                print(f"[Voice] TTS skipped: gen_id mismatch or empty audio")
        except Exception as e:
            import traceback
            print(f"[Voice Event Loop] TTS error: {e}")
            traceback.print_exc()
        finally:
            if ws:
                try:
                    await ws.send_json({"type": "voice_status", "status": "listening"})
                except Exception:
                    pass

    # Signal text done (with tts_streamed flag so frontend doesn't re-fetch)
    if response_text:
        await broadcast_game_event("voice_text_done", {"text": response_text, "tts_streamed": True})


async def process_voice_command(spoken_text: str) -> dict:
    """
    Process a spoken voice command using intent routing and LLM.

    Routes to:
    - Mode switch: Changes agent personality
    - Move: Executes chess move + LLM comment
    - Game command: Status, help, etc.
    - Conversation: General LLM chat
    """
    global current_game, current_mode, current_language, conversation_context

    # Get current board for move detection
    board = chess.Board(current_game["fen"]) if current_game else None

    # Classify intent
    intent_result = intent_router.classify(spoken_text, board)

    # Handle language switch
    if intent_result.intent == Intent.LANGUAGE_SWITCH:
        return await handle_language_switch(intent_result.data["language"], spoken_text)

    # Handle mode switch
    if intent_result.intent == Intent.MODE_SWITCH:
        return await handle_mode_switch(intent_result.data["mode"], spoken_text)

    # Handle game commands
    if intent_result.intent == Intent.GAME_COMMAND:
        return await handle_game_command(intent_result.data["command"], spoken_text)

    # Handle chess move
    if intent_result.intent == Intent.MOVE:
        return await handle_move(intent_result.data, spoken_text)

    # Handle conversation (default)
    return await handle_conversation(spoken_text)


async def handle_mode_switch(new_mode: AgentMode, spoken_text: str) -> dict:
    """Handle mode switch request."""
    global current_mode

    current_mode = new_mode
    new_voice = MODE_VOICES.get(new_mode, "alloy")

    # Update TTS voice for new mode
    if tts_service:
        tts_service.default_voice = new_voice

    greeting = get_mode_greeting(new_mode, current_language)

    # Add to conversation history
    if conversation_context:
        conversation_context.add_exchange(spoken_text, greeting)

    return {
        "text": greeting,
        "mode_changed": new_mode.value,
        "voice": new_voice,
        "language": current_language,
        "game_state": create_game_state().model_dump() if current_game else None,
    }


async def handle_language_switch(new_language: str, spoken_text: str) -> dict:
    """Handle language switch request."""
    global current_language

    if not new_language or len(new_language) > 30:
        return {"text": f"Invalid language: {new_language}"}

    current_language = new_language

    # Update TTS voice language (CosyVoice switches zh_default/en_default)
    if tts_service and hasattr(tts_service, 'language'):
        tts_service.language = new_language

    # Update voice handler language for STT hints
    if voice_handler:
        voice_handler.set_language(new_language)

    # Update intent router language
    if intent_router:
        intent_router.set_language(new_language)

    greeting = get_language_greeting(new_language)

    # Add to conversation history
    if conversation_context:
        conversation_context.add_exchange(spoken_text, greeting)

    return {
        "text": greeting,
        "language_changed": new_language,
        "game_state": create_game_state().model_dump() if current_game else None,
    }


async def handle_game_command(command: str, spoken_text: str) -> dict:
    """Handle game control commands (dual-language)."""
    global current_game, _agent_paused, _agent_pause_reason, _agent_pause_options

    zh = _is_zh()

    if command == "status":
        if current_game is None:
            return {"text": "目前沒有進行中的棋局。" if zh else "No game is currently active."}

        fen = current_game["fen"]
        board = chess.Board(fen)
        move_num = current_game["move_number"]
        whose_turn = current_game["whose_turn"]
        mode_name = get_mode_name(current_mode, current_language)

        if zh:
            turn_str = "人類" if whose_turn == "human" else "機器人"
            check_str = "將軍！" if board.is_check() else ""
            status_text = f"第 {move_num} 手。輪到{turn_str}。目前是{mode_name}。{check_str}"
        else:
            status_text = (
                f"Move {move_num}. It is {whose_turn}'s turn. "
                f"I'm in {mode_name}. "
                f"{'Check!' if board.is_check() else ''}"
            )
        return {
            "text": status_text,
            "game_state": create_game_state().model_dump(),
        }

    elif command == "help":
        help_text = intent_router.get_help_text(current_language) if intent_router else (
            "問我任何事！" if zh else "Ask me anything!"
        )
        if zh:
            return {"text": "你可以：說出棋步如「兵到e4」、切換模式、切換語言，或直接跟我聊天！"}
        return {"text": "Here's what you can do: say a move like 'pawn to e4', switch modes, switch language, or just chat with me!"}

    elif command == "resign":
        if current_game:
            current_game["status"] = "ended"
            current_game["game_result"] = "resigned"
            return {
                "text": "你認輸了。好棋！要再來一局嗎？" if zh else "You've resigned. Good game! Want to play again?",
                "game_state": create_game_state().model_dump(),
            }
        return {"text": "目前沒有棋局可以認輸。" if zh else "No active game to resign from."}

    elif command == "stop":
        if current_game:
            await _cancel_auto_detect()
            current_game["status"] = "ended"
            current_game["game_result"] = "stopped"
            gs = create_game_state()
            current_game = None
            return {
                "text": "棋局已結束。" if zh else "Game stopped.",
                "game_state": gs.model_dump() if gs else None,
            }
        return {"text": "目前沒有棋局可以停止。" if zh else "No active game to stop."}

    elif command == "new_game":
        if current_game and current_game["status"] == "playing":
            if zh:
                return {"text": "目前已有進行中的棋局。先說「停止」結束，再說「新局」。"}
            return {"text": "There's already an active game. Say 'stop' first to end it, then 'start a game'."}
        if zh:
            return {"text": "說「開始遊戲」就可以開始！我準備好了。"}
        return {"text": "Say 'start a game' to begin! I'm ready when you are."}

    elif command == "your_turn":
        if current_game is None:
            return {"text": "目前沒有棋局。" if zh else "No active game."}
        if current_game["whose_turn"] != "human":
            return {"text": "現在是我的回合，請稍等。" if zh else "It's already my turn, hold on."}

        # Cancel auto-detect and trigger immediate single detection
        await _cancel_auto_detect()
        try:
            result = await _detect_and_respond()
            if result.get("success"):
                gs = result.get("game_state")
                return {
                    "text": "好的，我看到你的棋步了！讓我想想…" if zh else "Got it, I see your move! Let me think...",
                    "game_state": gs.model_dump() if hasattr(gs, 'model_dump') else gs,
                }
            else:
                error = result.get("error", "Unknown error")
                return {
                    "text": f"我看不到你的棋步：{error}" if zh else f"I can't detect your move: {error}",
                }
        except Exception as e:
            return {"text": f"偵測失敗：{e}" if zh else f"Detection failed: {e}"}

    return {"text": f"指令 '{command}' 已收到。" if zh else f"Command '{command}' acknowledged."}


async def handle_move(move_data: dict, spoken_text: str) -> dict:
    """Handle chess move with LLM commentary."""
    global current_game

    zh = _is_zh()

    if current_game is None:
        prompt = "目前沒有棋局。請先開始新局。" if zh else "No active game. Please start a game first."
        response = await get_llm_response(prompt)
        return {"text": response}

    if current_game["whose_turn"] != "human":
        prompt = "還沒輪到你，請等我走完。" if zh else "It's not your turn yet. Please wait for me to finish my move."
        response = await get_llm_response(prompt)
        return {"text": response}

    uci_move = move_data.get("move")

    if uci_move is None:
        # Could not parse move - ask LLM for help
        if zh:
            prompt = f"人類說了「{spoken_text}」，但我無法理解成棋步。請用你目前的個性幫助他們。"
        else:
            prompt = f"The human said '{spoken_text}' but I couldn't understand it as a chess move. Help them in my current personality."
        response = await get_llm_response(prompt)
        return {"text": response, "error": "Could not parse move"}

    # Validate move is legal
    board = chess.Board(current_game["fen"])
    try:
        move_obj = chess.Move.from_uci(uci_move)
    except ValueError:
        return {"text": f"'{uci_move}' doesn't look like a valid move. Try something like 'pawn to e4'.", "error": "Invalid move format"}

    if move_obj not in board.legal_moves:
        if zh:
            prompt = f"人類說了「{spoken_text}」，但 {uci_move} 不是合法棋步。請幫助他們。"
        else:
            prompt = f"The human said '{spoken_text}' but {uci_move} is not a legal move. Help them."
        response = await get_llm_response(prompt)
        return {"text": response, "error": f"Illegal move: {uci_move}"}

    # Cancel auto-detect — voice/chat move takes priority
    await _cancel_auto_detect()

    # Execute the move
    move_desc = describe_move(uci_move, current_game["fen"])

    async with game_lock:
        # Update game state and tracker
        board.push(move_obj)
        new_fen = board.fen()

        tracker: BoardTracker = current_game.get("tracker")
        if tracker:
            tracker.push_uci(uci_move)
            new_fen = tracker.fen

        current_game["fen"] = new_fen
        current_game["last_human_move"] = uci_move
        current_game["whose_turn"] = "robot"
        _save_game_state()

        await broadcast_game_event("move_detected", {
            "move": uci_move,
            "source": "voice",
            "game_state": create_game_state().dict(),
        })

        # Check game status
        status = get_game_status(new_fen)
        if status != "playing":
            current_game["status"] = "ended"
            current_game["game_result"] = status
            if zh:
                prompt = f"人類走了 {move_desc}，棋局結束：{status}。對此做出反應。"
            else:
                prompt = f"The human played {move_desc} and the game ended: {status}. React to this."
            response = await get_llm_response(prompt)
            return {
                "text": response,
                "move": uci_move,
                "game_state": create_game_state().model_dump(),
            }

        # Make robot's move
        await make_robot_move()

        robot_move = current_game.get("last_robot_move")

        # Get LLM commentary on the exchange
        if robot_move:
            robot_desc = describe_move(robot_move, new_fen)
            if zh:
                prompt = f"人類走了 {move_desc}。我回了 {robot_desc}。評論這個交換。"
            else:
                prompt = f"The human played {move_desc}. I responded with {robot_desc}. Comment on this exchange."
        else:
            if zh:
                prompt = f"人類走了 {move_desc}。評論他們的棋步。"
            else:
                prompt = f"The human played {move_desc}. Comment on their move."

        response = await get_llm_response(prompt)

    # Add to conversation
    if conversation_context:
        conversation_context.add_exchange(spoken_text, response)

    return {
        "text": response,
        "move": uci_move,
        "game_state": create_game_state().model_dump() if current_game else None,
    }


async def handle_conversation(spoken_text: str) -> dict:
    """Handle general conversation with LLM.

    After responding, if the game is active and it's the human's turn,
    ensures the auto-detect loop is running so vision can detect their move.
    """
    response = await get_llm_response(spoken_text)

    # Add to conversation history
    if conversation_context:
        conversation_context.add_exchange(spoken_text, response)

    # If it's the human's turn, ensure auto-detect is running.
    # The user may be implying they've moved (e.g. "I just moved my pawn")
    # without using exact trigger phrases like "your turn".
    if (current_game
            and current_game.get("status") == "playing"
            and current_game.get("whose_turn") == "human"):
        if _auto_detect_task is None or _auto_detect_task.done():
            print("[Agent] Chat during human turn — restarting auto-detect")
            await _start_auto_detect()

    return {
        "text": response,
        "game_state": create_game_state().model_dump() if current_game else None,
    }


def _get_system_prompt() -> str:
    """Build the system prompt for the current mode and language."""
    if current_mode == GameMode.TEACH:
        step = _get_teach_step()
        lesson_ctx = ""
        if step:
            zh = _is_zh()
            instr = step.instruction_zh if zh and step.instruction_zh else step.instruction
            lesson_ctx = f"Current exercise: {instr}"
        return build_teach_prompt(current_language, lesson_ctx)
    else:
        return build_battle_prompt(current_character, current_language)


async def get_llm_response(user_message: str) -> str:
    """Get LLM response with current mode, language, and game context."""
    if llm_service is None:
        if _is_zh():
            return "我現在有點問題，請稍後再試。"
        return "I'm having trouble thinking right now. Try again in a moment."

    system_prompt = _get_system_prompt()
    game_context = get_game_context()
    history = conversation_context.get_history() if conversation_context else []

    try:
        response = await llm_service.chat_async(
            user_message=user_message,
            system_prompt=system_prompt,
            conversation_history=history,
            game_context=game_context,
            max_tokens=_get_max_tokens(),
        )
        return response
    except Exception as e:
        print(f"[Agent] LLM error: {e}")
        if _is_zh():
            return "嗯，讓我想想...其實我剛才有點恍神。你剛說什麼？"
        return "Hmm, let me think... Actually, I'm having a moment. What were you saying?"


async def get_llm_response_streaming(user_message: str):
    """
    Async generator that streams LLM response chunks.

    For providers that support reliable streaming (OpenAI), streams chunks.
    For others (Gemini, Claude), falls back to non-streaming to avoid truncation.
    """
    global _llm_generation_id
    gen_id = _llm_generation_id

    if llm_service is None:
        if _is_zh():
            yield "我現在有點問題，請稍後再試。"
        else:
            yield "I'm having trouble thinking right now. Try again in a moment."
        return

    system_prompt = _get_system_prompt()
    game_ctx = get_game_context()
    # Skip conversation history for Watch mode — each commentary is independent
    # (history confuses Gemini/Claude into echoing the format)
    is_watch = current_mode == GameMode.WATCH
    history = [] if is_watch else (conversation_context.get_history() if conversation_context else [])

    try:
        provider = llm_service.provider if llm_service else "none"
        if provider == "openai":
            # OpenAI: reliable streaming
            async for chunk in llm_service.stream_chat_async(
                user_message=user_message,
                system_prompt=system_prompt,
                conversation_history=history,
                game_context=game_ctx,
                max_tokens=_get_max_tokens(),
            ):
                if _llm_generation_id != gen_id:
                    print(f"[Agent] LLM generation interrupted")
                    return
                yield chunk
        else:
            # Gemini/Claude: use non-streaming to avoid truncation
            print(f"[Agent] Non-streaming path: provider={provider}, model={llm_service.model}")
            response = await llm_service.chat_async(
                user_message=user_message,
                system_prompt=system_prompt,
                conversation_history=history,
                game_context=game_ctx,
                max_tokens=_get_max_tokens(),
            )
            if _llm_generation_id != gen_id:
                return
            if response:
                # Yield word by word to simulate streaming for TTS
                words = response.split(" ")
                for i, word in enumerate(words):
                    if _llm_generation_id != gen_id:
                        return
                    yield word if i == 0 else " " + word
    except Exception as e:
        import traceback
        print(f"[Agent] LLM streaming error: {e}")
        traceback.print_exc()
        if _is_zh():
            yield "嗯，讓我想想...其實我剛才有點恍神。"
        else:
            yield "Hmm, let me think... Actually, I'm having a moment."


# ---------------------------------------------------------------------------
# Sentence-level TTS pipelining (text + audio sync)
# ---------------------------------------------------------------------------

# Sentence boundary regex: split after . ! ? and CJK equivalents
_SENTENCE_END_RE = re.compile(r'(?<=[.!?。！？])\s*')


def _split_sentences(text: str):
    """Split *text* at sentence boundaries.  Returns a list; the last element
    may be an incomplete sentence (no terminator yet)."""
    parts = _SENTENCE_END_RE.split(text)
    return [p for p in parts if p]


async def _play_robot_gesture(gesture_name: str, return_to_vision: bool = False, max_duration: float = 2.0):
    """Fire-and-forget: ask the robot service to play a gesture.

    Args:
        return_to_vision: If True, move to vision position after gesture.
        max_duration: Cap gesture to this many seconds (auto-speeds up if longer).
    """
    try:
        payload = {"name": gesture_name, "speed": 1.0}
        if max_duration:
            payload["max_duration"] = max_duration
        resp = await http_client.post(
            f"{ROBOT_SERVICE_URL}/arm/gesture/play",
            json=payload,
            timeout=30.0,
        )
        data = resp.json()
        if not data.get("success"):
            print(f"[Agent] Gesture '{gesture_name}' failed: {data.get('error')}")
        elif return_to_vision:
            await move_robot_for_vision()
    except Exception as e:
        print(f"[Agent] Gesture '{gesture_name}' error: {e}")


async def _stream_text_with_tts(prompt: str, gen_id: int, emotion: str = "", gesture: str = "") -> str:
    """Stream LLM text to UI **and** TTS audio sentence-by-sentence via game WS.

    Text chunks are broadcast immediately (``voice_text_chunk``).
    As soon as a full sentence is detected, its TTS is queued.
    A background worker synthesises one sentence at a time and broadcasts
    ``voice_audio_chunk`` events so the frontend can play audio in order.

    Args:
        emotion: Optional emotion preset for CosyVoice (e.g. "happy", "serious").
                 Ignored by OpenAI TTS.
        gesture: Optional gesture name to play on the robot arm while speaking
                 (e.g. "nod", "celebrate"). Fire-and-forget.

    Returns the full generated text.
    """
    full_text = ""
    sentence_buffer = ""
    tts_queue: asyncio.Queue = asyncio.Queue()

    # Fire off gesture on robot arm (non-blocking, physical mode only)
    if gesture and not (current_game and current_game.get("simulation", False)):
        asyncio.create_task(_play_robot_gesture(gesture))

    # --- TTS worker: sequential sentence synthesis -------------------------
    async def _tts_worker():
        while True:
            sentence = await tts_queue.get()
            if sentence is None:  # sentinel → done
                break
            if _llm_generation_id != gen_id:
                break
            if not _tts_enabled or not tts_service:
                continue  # Skip TTS synthesis when disabled
            try:
                audio_bytes = await tts_service.synthesize_async(
                    sentence,
                    response_format="mp3",
                    emotion=emotion,
                )
                audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
                await broadcast_game_event("voice_audio_chunk", {
                    "data": audio_b64,
                    "format": "mp3",
                })
            except Exception as e:
                print(f"[Agent] TTS sentence error: {type(e).__name__}: {e}")

    worker: Optional[asyncio.Task] = None
    if tts_service:
        worker = asyncio.create_task(_tts_worker())

    # --- Stream text chunks ------------------------------------------------
    async for chunk in get_llm_response_streaming(prompt):
        if _llm_generation_id != gen_id:
            break
        full_text += chunk
        sentence_buffer += chunk
        await broadcast_game_event("voice_text_chunk", {"chunk": chunk})

        # Check for sentence boundaries
        parts = _split_sentences(sentence_buffer)
        if len(parts) > 1:
            for complete_sentence in parts[:-1]:
                s = complete_sentence.strip()
                if s and tts_service:
                    await tts_queue.put(s)
            sentence_buffer = parts[-1]

    # Flush remaining fragment as final sentence
    if sentence_buffer.strip() and tts_service:
        await tts_queue.put(sentence_buffer.strip())

    # Signal TTS worker to stop and wait for it
    if worker:
        await tts_queue.put(None)
        await worker
        # Tell frontend all audio has been sent
        await broadcast_game_event("voice_audio_done", {})

    return full_text.strip()


async def _speak_text_only(text: str, emotion: str = ""):
    """Synthesise and broadcast TTS for a static (non-streamed) text string.

    Used when the LLM is unavailable and we have a fallback string.
    Passes emotion to CosyVoice TTS (ignored by OpenAI TTS).
    """
    if not tts_service or not text:
        return
    try:
        audio_bytes = await tts_service.synthesize_async(text, response_format="mp3", emotion=emotion)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        await broadcast_game_event("voice_audio_chunk", {
            "data": audio_b64,
            "format": "mp3",
        })
        await broadcast_game_event("voice_audio_done", {})
    except Exception as e:
        print(f"[Agent] TTS static text error: {e}")


@app.post("/agent/tts_enabled")
async def set_tts_enabled(request: dict):
    """Enable/disable TTS synthesis on the server."""
    global _tts_enabled
    _tts_enabled = bool(request.get("enabled", True))
    print(f"[Agent] TTS {'enabled' if _tts_enabled else 'disabled'}")
    return {"success": True, "tts_enabled": _tts_enabled}


@app.post("/agent/interrupt")
async def interrupt_agent():
    """
    Interrupt current AI speech/generation.

    Increments the generation ID to cancel any in-progress streaming,
    and broadcasts a voice_stop event to all clients (game WS + voice WS).
    """
    global _llm_generation_id
    _llm_generation_id += 1
    # Stop on game WS
    await broadcast_game_event("voice_stop", {})
    # Stop on voice WS
    ws = voice_handler.active_websocket if voice_handler else None
    if ws:
        try:
            await ws.send_json({"type": "voice_stop"})
        except Exception:
            pass
    print(f"[Agent] Interrupted (gen_id={_llm_generation_id})")
    return {"success": True}


@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for bidirectional voice communication.

    Protocol:
    Client -> Server:
        {"type": "audio_chunk", "data": "<base64>", "format": "webm", "is_final": false}
        {"type": "end_stream"}

    Server -> Client:
        {"type": "transcription", "text": "...", "is_final": true}
        {"type": "agent_response", "text": "...", "move_detected": "e2e4"}
        {"type": "audio_chunk", "data": "<base64 mp3>", "format": "mp3"}
        {"type": "audio_end"}
    """
    if voice_handler is None:
        await websocket.close(code=1011, reason="Voice services not available")
        return

    await voice_handler.handle_connection(websocket)


@app.post("/voice/speak")
async def speak_text(request: SpeakRequest):
    """
    Convert text to speech and stream audio.

    Returns streaming audio response.
    """
    if tts_service is None:
        raise HTTPException(status_code=503, detail="TTS service not available")

    def audio_generator():
        for chunk in tts_service.stream(
            request.text,
            voice=request.voice,
            speed=request.speed,
            response_format=request.format,
        ):
            yield chunk

    media_type = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }.get(request.format, "audio/mpeg")

    return StreamingResponse(audio_generator(), media_type=media_type)


@app.post("/game/voice_move", response_model=VoiceMoveResponse)
async def voice_move(request: VoiceMoveRequest):
    """
    Process a voice move input.

    Alternative to /game/human_moved that accepts audio instead of
    relying on vision to detect the move.
    """
    if stt_service is None:
        raise HTTPException(status_code=503, detail="STT service not available")

    if current_game is None:
        return VoiceMoveResponse(
            success=False,
            error="No active game",
        )

    if current_game["whose_turn"] != "human":
        return VoiceMoveResponse(
            success=False,
            error="Not human's turn",
        )

    # Decode and transcribe audio
    try:
        audio_bytes = base64.b64decode(request.audio_data)
    except Exception:
        return VoiceMoveResponse(
            success=False,
            error="Invalid base64 audio data",
        )

    stt_result = await stt_service.transcribe_async(
        audio_bytes, audio_format=request.audio_format
    )

    if not stt_result["success"]:
        return VoiceMoveResponse(
            success=False,
            error=f"Transcription failed: {stt_result.get('error')}",
        )

    transcribed_text = stt_result["text"]

    # Process as game command
    result = await process_voice_command(transcribed_text)

    return VoiceMoveResponse(
        success=result.get("move") is not None,
        transcribed_text=transcribed_text,
        parsed_move=result.get("move"),
        agent_response_text=result.get("text"),
        game_state=create_game_state() if current_game else None,
        error=result.get("error"),
    )


@app.get("/voice/info")
async def voice_info():
    """Get information about voice capabilities."""
    return {
        "stt_available": stt_service is not None,
        "tts_available": tts_service is not None,
        "websocket_available": voice_handler is not None,
        "tts_voices": getattr(tts_service, 'VOICES', []) if tts_service else [],
        "supported_audio_formats": ["wav", "mp3", "webm", "m4a", "ogg", "flac"],
    }


class ListeningModeRequest(BaseModel):
    """Listening mode request."""
    mode: str = Field(..., description="'push_to_talk' or 'always_on'")


@app.get("/voice/listening")
async def get_listening_mode():
    """Get current voice listening mode."""
    mode = voice_handler.listening_mode if voice_handler else "push_to_talk"
    return {"mode": mode}


@app.post("/voice/listening")
async def set_listening_mode(request: ListeningModeRequest):
    """Set voice listening mode."""
    if voice_handler is None:
        raise HTTPException(status_code=503, detail="Voice services not available")

    if request.mode not in ("push_to_talk", "always_on"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {request.mode}. Use 'push_to_talk' or 'always_on'.",
        )

    voice_handler.set_listening_mode(request.mode)
    return {"success": True, "mode": request.mode}


# =============================================================================
# Mode Endpoints
# =============================================================================


class ModeRequest(BaseModel):
    """Mode switch request."""
    mode: str = Field(..., description="Mode name: battle or teach")
    character: Optional[str] = None  # free-text personality for battle mode


class ModeResponse(BaseModel):
    """Mode response."""
    success: bool
    current_mode: str
    mode_name: str
    greeting: Optional[str] = None
    error: Optional[str] = None


@app.get("/agent/modes")
async def list_modes():
    """Get all available game modes."""
    modes = []
    for mode in GameMode:
        modes.append({
            "id": mode.value,
            "name": get_mode_name(mode, current_language),
            "voice": MODE_VOICES.get(mode, "alloy"),
            "is_current": mode == current_mode,
        })
    return {
        "modes": modes,
        "current_mode": current_mode.value,
        "language": current_language,
        "difficulties": list(DIFFICULTY_DEPTH.keys()),
        "move_sources": ["stockfish", "llm"],
    }


@app.get("/agent/mode")
async def get_current_mode():
    """Get current game mode."""
    return {
        "mode": current_mode.value,
        "name": get_mode_name(current_mode, current_language),
        "voice": MODE_VOICES.get(current_mode, "alloy"),
        "language": current_language,
        "character": current_character,
        "difficulty": current_game.get("difficulty", "intermediate") if current_game else "intermediate",
    }


@app.post("/agent/mode", response_model=ModeResponse)
async def set_mode(request: ModeRequest):
    """Switch game mode."""
    global current_mode, current_character

    # Parse mode string to enum
    try:
        new_mode = GameMode(request.mode.lower())
    except ValueError:
        return ModeResponse(
            success=False,
            current_mode=current_mode.value,
            mode_name=get_mode_name(current_mode, current_language),
            error=f"Unknown mode: {request.mode}. Valid modes: {[m.value for m in GameMode]}",
        )

    current_mode = new_mode
    if request.character is not None:
        current_character = request.character

    # Update TTS voice
    if tts_service:
        tts_service.default_voice = MODE_VOICES.get(new_mode, "alloy")

    greeting = get_mode_greeting(new_mode, current_language)

    return ModeResponse(
        success=True,
        current_mode=new_mode.value,
        mode_name=get_mode_name(new_mode, current_language),
        greeting=greeting,
    )


# =============================================================================
# Language Endpoints
# =============================================================================


class LanguageRequest(BaseModel):
    """Language switch request."""
    language: str = Field(..., description="Language code: en or zh-TW")


@app.get("/agent/language")
async def get_current_language():
    """Get current language setting."""
    return {
        "language": current_language,
        "available": ["en", "zh-TW"],
    }


@app.post("/agent/language")
async def set_language(request: LanguageRequest):
    """Switch agent language."""
    global current_language

    if not request.language or len(request.language) > 30:
        raise HTTPException(status_code=400, detail=f"Invalid language: {request.language}")

    current_language = request.language

    # Persist language to settings file
    settings = _load_settings()
    settings["language"] = request.language
    _save_settings(settings)

    # Update TTS voice language
    if tts_service and hasattr(tts_service, 'language'):
        tts_service.language = request.language

    # Update voice handler language for STT
    if voice_handler:
        voice_handler.set_language(request.language)

    # Update intent router language
    if intent_router:
        intent_router.set_language(request.language)

    greeting = get_language_greeting(request.language)

    return {
        "success": True,
        "language": current_language,
        "greeting": greeting,
    }


class ChatRequest(BaseModel):
    """Chat request body."""
    message: str


# ── Character Management ─────────────────────────────────────────────
_CHARACTERS_FILE = Path("/app/characters.yaml") if Path("/app").exists() else Path(__file__).parent.parent.parent / "characters.yaml"
_characters: list = []


def _load_characters():
    global _characters
    import yaml
    if _CHARACTERS_FILE.exists():
        with open(_CHARACTERS_FILE) as f:
            _characters = yaml.safe_load(f) or []
    else:
        _characters = []


def _save_characters():
    import yaml
    with open(_CHARACTERS_FILE, "w") as f:
        yaml.dump(_characters, f, allow_unicode=True, default_flow_style=False)


# Load on import
_load_characters()


@app.get("/agent/characters")
async def list_characters(language: str = ""):
    """List saved characters, optionally filtered by language."""
    if language:
        return {"characters": [c for c in _characters if c.get("language") == language]}
    return {"characters": _characters}


@app.post("/agent/characters/save")
async def save_character(request: dict):
    """Save a character (language + title + description). Updates if same language+title exists."""
    lang = request.get("language", "en")
    title = request.get("title", "").strip()
    desc = request.get("description", "").strip()[:200]
    if not title:
        return {"success": False, "error": "Title is required"}

    # Update existing or append
    for c in _characters:
        if c.get("language") == lang and c.get("title") == title:
            c["description"] = desc
            _save_characters()
            return {"success": True, "message": f"Updated '{title}'"}

    _characters.append({"language": lang, "title": title, "description": desc})
    _save_characters()
    return {"success": True, "message": f"Saved '{title}'"}


@app.delete("/agent/characters/{title}")
async def delete_character(title: str, language: str = ""):
    """Delete a character by title (and optionally language)."""
    global _characters
    before = len(_characters)
    if language:
        _characters = [c for c in _characters if not (c.get("title") == title and c.get("language") == language)]
    else:
        _characters = [c for c in _characters if c.get("title") != title]
    if len(_characters) < before:
        _save_characters()
        return {"success": True}
    return {"success": False, "error": "Character not found"}


@app.get("/agent/settings")
async def get_agent_settings():
    """Get current agent settings."""
    return {
        "llm_model": LLM_MODEL,
        "llm_api_key": "configured" if LLM_API_KEY else "",
        "llm_base_url": LLM_BASE_URL,
        "llm_model_2": LLM_MODEL_2,
        "llm_api_key_2": "configured" if LLM_API_KEY_2 else "",
        "llm_base_url_2": LLM_BASE_URL_2,
        "llm_model_3": LLM_MODEL_3,
        "llm_api_key_3": "configured" if LLM_API_KEY_3 else "",
        "llm_base_url_3": LLM_BASE_URL_3,
        "tts_provider": TTS_PROVIDER,
        "tts_model": TTS_MODEL,
        "tts_voice": TTS_VOICE,
        "tts_service_url": TTS_SERVICE_URL,
        "stt_model": STT_MODEL,
        "language": current_language,
    }


@app.post("/agent/settings")
async def update_agent_settings(request: dict):
    """Update agent settings and persist to file. Changes take effect on next restart."""
    global TTS_PROVIDER, TTS_MODEL, TTS_VOICE, TTS_SERVICE_URL
    global LLM_API_KEY, LLM_MODEL, LLM_BASE_URL
    global LLM_MODEL_2, LLM_API_KEY_2, LLM_BASE_URL_2
    global LLM_MODEL_3, LLM_API_KEY_3, LLM_BASE_URL_3, STT_MODEL
    global llm_service_2, llm_service_3

    settings = _load_settings()

    # Update each setting if provided
    if "llm_api_key" in request:
        LLM_API_KEY = request["llm_api_key"]
        settings["llm_api_key"] = LLM_API_KEY
    if "llm_model" in request:
        LLM_MODEL = request["llm_model"]
        settings["llm_model"] = LLM_MODEL
    if "llm_base_url" in request:
        LLM_BASE_URL = request["llm_base_url"]
        settings["llm_base_url"] = LLM_BASE_URL
    if "llm_model_2" in request:
        LLM_MODEL_2 = request["llm_model_2"]
        settings["llm_model_2"] = LLM_MODEL_2
    if "llm_api_key_2" in request:
        LLM_API_KEY_2 = request["llm_api_key_2"]
        settings["llm_api_key_2"] = LLM_API_KEY_2
    if "llm_base_url_2" in request:
        LLM_BASE_URL_2 = request["llm_base_url_2"]
        settings["llm_base_url_2"] = LLM_BASE_URL_2
    if "llm_model_3" in request:
        LLM_MODEL_3 = request["llm_model_3"]
        settings["llm_model_3"] = LLM_MODEL_3
    if "llm_api_key_3" in request:
        LLM_API_KEY_3 = request["llm_api_key_3"]
        settings["llm_api_key_3"] = LLM_API_KEY_3
    if "llm_base_url_3" in request:
        LLM_BASE_URL_3 = request["llm_base_url_3"]
        settings["llm_base_url_3"] = LLM_BASE_URL_3
    if "tts_provider" in request:
        TTS_PROVIDER = request["tts_provider"]
        settings["tts_provider"] = TTS_PROVIDER
    if "tts_model" in request:
        TTS_MODEL = request["tts_model"]
        settings["tts_model"] = TTS_MODEL
    if "tts_voice" in request:
        TTS_VOICE = request["tts_voice"]
        settings["tts_voice"] = TTS_VOICE
    if "tts_service_url" in request:
        TTS_SERVICE_URL = request["tts_service_url"]
        settings["tts_service_url"] = TTS_SERVICE_URL
    if "stt_model" in request:
        STT_MODEL = request["stt_model"]
        settings["stt_model"] = STT_MODEL

    _save_settings(settings)

    # Apply TTS changes to the live service instance
    provider_changed = "tts_provider" in request or "tts_service_url" in request
    if provider_changed:
        # Provider switch requires re-init
        try:
            global tts_service
            if TTS_PROVIDER == "cosyvoice":
                from audio.tts_cosyvoice import CosyVoiceTTSProvider
                tts_service = CosyVoiceTTSProvider(
                    service_url=TTS_SERVICE_URL,
                    default_voice=TTS_VOICE,
                    language=current_language,
                )
                print(f"[Settings] TTS switched to CosyVoice at {TTS_SERVICE_URL}")
            else:
                from audio.tts_service import TTSService
                tts_service = TTSService(api_key=LLM_API_KEY or None, default_voice=TTS_VOICE, model=TTS_MODEL)
                print(f"[Settings] TTS switched to OpenAI ({TTS_MODEL})")
            if voice_handler:
                voice_handler.tts_service = tts_service
        except Exception as e:
            print(f"[Settings] TTS re-init failed: {e}")
    else:
        # Update voice/model on existing service
        if tts_service:
            if "tts_voice" in request and hasattr(tts_service, 'default_voice'):
                tts_service.default_voice = TTS_VOICE
            if "tts_model" in request and hasattr(tts_service, 'model'):
                tts_service.model = TTS_MODEL

    # Apply LLM model/base_url change
    llm1_changed = any(k in request for k in ("llm_api_key", "llm_model", "llm_base_url"))
    if llm1_changed and llm_service:
        if "llm_api_key" in request or "llm_base_url" in request:
            from openai import OpenAI
            kwargs = {"api_key": LLM_API_KEY or None}
            if LLM_BASE_URL:
                kwargs["base_url"] = LLM_BASE_URL
            llm_service.client = OpenAI(**kwargs)
            llm_service.base_url = LLM_BASE_URL
        llm_service.model = LLM_MODEL
        print(f"[Settings] LLM 1: {LLM_MODEL}" + (f" @ {LLM_BASE_URL}" if LLM_BASE_URL else ""))

    # Apply LLM 2 changes — re-create service
    llm2_changed = any(k in request for k in ("llm_model_2", "llm_api_key_2", "llm_base_url_2"))
    if llm2_changed:
        if LLM_MODEL_2:
            try:
                llm_service_2 = LLMService(
                    model=LLM_MODEL_2,
                    api_key=LLM_API_KEY_2 or None,
                    base_url=LLM_BASE_URL_2 or None,
                )
                print(f"[Settings] LLM 2: {LLM_MODEL_2}" + (f" @ {LLM_BASE_URL_2}" if LLM_BASE_URL_2 else ""))
            except Exception as e:
                print(f"[Settings] LLM 2 init failed: {e}")
                llm_service_2 = None
        else:
            llm_service_2 = None
            print("[Settings] LLM 2 cleared")

    llm3_changed = any(k in request for k in ("llm_model_3", "llm_api_key_3", "llm_base_url_3"))
    if llm3_changed:
        if LLM_MODEL_3:
            try:
                llm_service_3 = LLMService(
                    model=LLM_MODEL_3,
                    api_key=LLM_API_KEY_3 or None,
                    base_url=LLM_BASE_URL_3 or None,
                )
                print(f"[Settings] LLM 3: {LLM_MODEL_3}" + (f" @ {LLM_BASE_URL_3}" if LLM_BASE_URL_3 else ""))
            except Exception as e:
                print(f"[Settings] LLM 3 init failed: {e}")
                llm_service_3 = None
        else:
            llm_service_3 = None
            print("[Settings] LLM 3 cleared")

    return {
        "success": True,
        "message": "Settings applied",
    }


@app.post("/agent/generate_lesson")
async def generate_lesson(request: dict):
    """Generate a lesson using LLM. Requires topic and difficulty."""
    import json as _json
    topic = request.get("topic", "").strip()
    difficulty = request.get("difficulty", "beginner")
    language = request.get("language", current_language)
    zh = "zh" in language

    if not llm_service:
        return {"success": False, "error": "LLM not available"}

    prompt = (
        f"Create a chess lesson YAML for topic: '{topic or 'random opening'}', difficulty: {difficulty}.\n"
        f"Return ONLY valid JSON (no markdown, no explanation) with this exact structure:\n"
        f'{{"lesson_id": "unique_snake_case_id",\n'
        f' "title": "English Title",\n'
        f' "title_zh": "中文標題",\n'
        f' "description": "English description",\n'
        f' "description_zh": "中文描述",\n'
        f' "difficulty": "{difficulty}",\n'
        f' "steps": [\n'
        f'   {{"fen": "valid FEN position",\n'
        f'     "instruction": "English instruction for this step",\n'
        f'     "instruction_zh": "中文指示",\n'
        f'     "expected_move": "UCI move like e2e4",\n'
        f'     "hints": ["hint1", "hint2"],\n'
        f'     "hints_zh": ["提示1", "提示2"],\n'
        f'     "explanation": "What the student learned",\n'
        f'     "explanation_zh": "學生學到了什麼"}}\n'
        f' ]}}\n\n'
        f"IMPORTANT:\n"
        f"- Each step's FEN must be the EXACT position BEFORE the expected_move.\n"
        f"- The expected_move must be a legal UCI move in that FEN position.\n"
        f"- Steps must follow a logical game sequence (each FEN follows from previous step + opponent reply).\n"
        f"- Include 4-7 steps for a complete lesson.\n"
        f"- Difficulty '{difficulty}': {'simple concepts, basic moves' if difficulty == 'beginner' else 'tactical patterns, combinations' if difficulty == 'intermediate' else 'advanced strategy, deep calculation'}.\n"
        f"{'- Focus on Chinese explanations.' if zh else '- Focus on English explanations.'}"
    )

    try:
        # Use higher max_tokens — lesson JSON needs ~3000 tokens
        if llm_service is None:
            return {"success": False, "error": "LLM not available"}
        response = await llm_service.chat_async(
            user_message=prompt,
            system_prompt="You are a chess lesson creator. Return ONLY valid JSON.",
            conversation_history=[],
            game_context=None,
            max_tokens=4000,
        )
        if not response:
            return {"success": False, "error": "LLM returned empty response"}
        print(f"[Lesson Gen] LLM response length: {len(response)}, first 200: {response[:200]}")
        # Strip markdown code fences if present
        clean = response.strip()
        if clean.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = clean.find('\n')
            if first_newline > 0:
                clean = clean[first_newline + 1:]
            # Remove closing fence
            if clean.rstrip().endswith("```"):
                clean = clean.rstrip()[:-3].rstrip()

        # Parse JSON
        start = clean.find('{')
        end = clean.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = clean[start:end]
            print(f"[Lesson Gen] Extracted JSON length: {len(json_str)}")
            lesson_data = _json.loads(json_str)
            return {"success": True, "lesson": lesson_data}
        return {"success": False, "error": "No JSON object found in LLM response", "raw": clean[:500]}
    except _json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON parse error: {e}", "raw": clean[:500] if 'clean' in dir() else response[:500]}
    except Exception as e:
        return {"success": False, "error": str(e), "raw": response[:500] if response else ""}


@app.post("/agent/save_lesson")
async def save_lesson_endpoint(request: dict):
    """Save a generated lesson to the lessons directory."""
    import yaml
    lesson_data = request.get("lesson")
    if not lesson_data or not lesson_data.get("lesson_id"):
        return {"success": False, "error": "Invalid lesson data"}

    lesson_id = lesson_data["lesson_id"]
    # Sanitize filename
    safe_id = "".join(c for c in lesson_id if c.isalnum() or c in "_-").strip()
    if not safe_id:
        return {"success": False, "error": "Invalid lesson_id"}

    lessons_dir = Path("/app/lessons") if Path("/app/lessons").exists() else Path(__file__).parent.parent.parent / "lessons"
    filepath = lessons_dir / f"{safe_id}.yaml"

    with open(filepath, "w") as f:
        yaml.dump(lesson_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    return {"success": True, "message": f"Lesson saved as {safe_id}.yaml", "lesson_id": safe_id}


@app.delete("/agent/lesson/{lesson_id}")
async def delete_lesson_endpoint(lesson_id: str):
    """Delete a lesson file."""
    lessons_dir = Path("/app/lessons") if Path("/app/lessons").exists() else Path(__file__).parent.parent.parent / "lessons"
    filepath = lessons_dir / f"{lesson_id}.yaml"
    if filepath.exists():
        filepath.unlink()
        return {"success": True}
    return {"success": False, "error": "Lesson not found"}


@app.post("/agent/generate_character")
async def generate_character(request: dict):
    """Generate a character description from a title using LLM."""
    title = request.get("title", "").strip()
    if not title:
        return {"description": ""}
    if llm_service:
        zh = _is_zh()
        if zh:
            prompt = (
                f"幫一個叫「{title}」的下棋機器人角色寫一段短短的個性描述。"
                f"最多200字。描述他們怎麼說話、對棋步的反應、態度。要有創意又有趣。只要描述，不要標題。"
            )
        else:
            prompt = (
                f"Create a short chess robot personality description for a character called '{title}'. "
                f"Max 200 characters. Describe how they speak, react to moves, and their attitude. "
                f"Be creative and fun. Just the description, no title."
            )
        desc = await llm_service.chat_async(
            user_message=prompt,
            system_prompt="You are a creative writer. Return ONLY the description text, nothing else.",
            conversation_history=[], game_context=None, max_tokens=300,
        )
        return {"description": desc[:200] if desc else ""}
    return {"description": f"A {title.lower()} who plays chess with attitude and style."}


@app.post("/agent/random_character")
async def random_character():
    """Generate a completely random character using LLM."""
    import random
    if llm_service:
        zh = _is_zh()
        seed = random.randint(1, 99999)
        if zh:
            themes = ["歷史人物", "動漫角色", "神話生物", "科幻角色", "美食家", "運動明星", "音樂家", "偵探", "古代武將", "太空人"]
            theme = random.choice(themes)
            prompt = (
                f"(隨機種子：{seed}) 發明一個獨特有創意的下棋機器人個性，風格靈感來自「{theme}」。"
                f"只回傳一個JSON物件，兩個欄位："
                f'"title"（2-3個字的角色名）和 "description"（最多200字，描述下棋時怎麼說話和行為）。'
                f"要有創意，每次都不一樣。只要JSON，不要其他東西。"
            )
        else:
            themes = ["historical figure", "anime character", "mythical creature", "sci-fi persona", "foodie", "sports star", "musician", "detective", "ancient warrior", "astronaut"]
            theme = random.choice(themes)
            prompt = (
                f"(Random seed: {seed}) Invent a unique, creative chess robot personality inspired by '{theme}'. "
                f"Return ONLY a JSON object with two fields: "
                f'"title" (2-3 words, the character name) and '
                f'"description" (max 200 chars, how they speak and behave during chess). '
                f"Be creative and DIFFERENT every time. Just the JSON, nothing else."
            )
        response = await llm_service.chat_async(
            user_message=prompt,
            system_prompt="Return ONLY valid JSON. Be wildly creative and never repeat previous ideas.",
            conversation_history=[], game_context=None, max_tokens=300,
        )
        try:
            import json
            clean = response.strip()
            if clean.startswith("```"):
                first_nl = clean.find('\n')
                if first_nl > 0: clean = clean[first_nl + 1:]
                if clean.rstrip().endswith("```"): clean = clean.rstrip()[:-3].rstrip()
            start = clean.find('{')
            end = clean.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(clean[start:end])
                return {
                    "title": data.get("title", "Mystery Character")[:50],
                    "description": data.get("description", "")[:200],
                }
        except Exception:
            pass
    return {"title": "", "description": ""}


@app.post("/agent/chat")
async def chat_with_agent(request: ChatRequest):
    """
    Chat with the agent (text input, voice + text output).
    """
    if llm_service is None:
        return {"text": None, "error": "LLM service not available"}

    result = await process_voice_command(request.message)

    # Broadcast response text (+ TTS if available)
    if result.get("text"):
        gen_id = _llm_generation_id
        asyncio.ensure_future(_speak_chat_response(result["text"], gen_id))

    return result


async def _speak_chat_response(text: str, gen_id: int):
    """Broadcast chat response text + TTS audio."""
    try:
        # Check if interrupted before speaking
        if _llm_generation_id != gen_id:
            print(f"[Agent] Chat TTS skipped: interrupted (gen_id {gen_id} vs {_llm_generation_id})")
            return
        await broadcast_game_event("voice_text_chunk", {"chunk": text})
        if tts_service and _llm_generation_id == gen_id:
            audio_bytes = await tts_service.synthesize_async(text, response_format="mp3")
            if _llm_generation_id == gen_id and audio_bytes:
                audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
                await broadcast_game_event("voice_audio_chunk", {"data": audio_b64, "format": "mp3"})
                await broadcast_game_event("voice_audio_done", {})
            else:
                print(f"[Agent] Chat TTS skipped after synthesis: gen_id mismatch")
        await broadcast_game_event("voice_text_done", {"text": text, "tts_streamed": bool(tts_service)})
    except Exception as e:
        print(f"[Agent] Chat TTS error: {e}")


@app.post("/conversation/clear")
async def clear_conversation():
    """Clear conversation history."""
    if conversation_context:
        conversation_context.clear()
    return {"success": True, "message": "Conversation history cleared."}


@app.get("/conversation/history")
async def get_conversation_history():
    """Get current conversation history."""
    if conversation_context is None:
        return {"messages": [], "count": 0}
    return conversation_context.to_dict()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
