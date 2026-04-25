"""
WebSocket handler for bidirectional voice communication.

Supports push-to-talk and always-on listening modes.
"""

import json
import base64
import asyncio
import time
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass

from fastapi import WebSocket, WebSocketDisconnect

from audio.stt_service import STTService
from audio.tts_service import TTSService  # also accepts CosyVoiceTTSProvider (same interface)
from audio.audio_buffer import AudioBuffer, AudioChunkValidator

from voice_events import VoiceEvent, VoiceEventQueue, VoiceEventPriority
from conversation.intent_router import IntentRouter, Intent


@dataclass
class VoiceMessage:
    """Represents a voice protocol message."""

    type: str
    data: Optional[str] = None
    text: Optional[str] = None
    format: Optional[str] = None
    is_final: bool = False
    move_detected: Optional[str] = None
    game_state: Optional[dict] = None
    error: Optional[str] = None
    voice: Optional[str] = None  # TTS voice used for response

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        result = {"type": self.type}
        if self.data is not None:
            result["data"] = self.data
        if self.text is not None:
            result["text"] = self.text
        if self.format is not None:
            result["format"] = self.format
        if self.is_final:
            result["is_final"] = self.is_final
        if self.move_detected is not None:
            result["move_detected"] = self.move_detected
        if self.game_state is not None:
            result["game_state"] = self.game_state
        if self.error is not None:
            result["error"] = self.error
        if self.voice is not None:
            result["voice"] = self.voice
        return result


# Type for game command handler callback
GameCommandHandler = Callable[[str], Awaitable[dict]]

# Type for board accessor
GetBoardFn = Callable[[], Optional[object]]  # Returns chess.Board or None

# Priority mapping from intent router string to enum
_PRIORITY_MAP = {
    "HIGH": VoiceEventPriority.HIGH,
    "NORMAL": VoiceEventPriority.NORMAL,
    "LOW": VoiceEventPriority.LOW,
    "IGNORE": VoiceEventPriority.IGNORE,
}


class VoiceHandler:
    """
    Handles bidirectional voice communication over WebSocket.

    Supports two listening modes:
    - push_to_talk: Audio buffered between start/end signals (default)
    - always_on: Continuous audio stream with server-side VAD

    Protocol:
    Client -> Server:
        {"type": "audio_chunk", "data": "<base64>", "format": "webm", "is_final": false}
        {"type": "end_stream"}
        {"type": "cancel"}
        {"type": "set_mode", "mode": "push_to_talk"|"always_on"}
        {"type": "echo_state", "speaking": true|false}

    Server -> Client:
        {"type": "transcription", "text": "...", "is_final": true}
        {"type": "agent_response", "text": "...", "move_detected": "e2e4"}
        {"type": "audio_chunk", "data": "<base64 mp3>", "format": "mp3", "is_final": false}
        {"type": "audio_end"}
        {"type": "listening_mode", "mode": "push_to_talk"|"always_on"}
        {"type": "voice_status", "status": "listening"|"processing"|"speaking"|"idle"}
        {"type": "error", "error": "..."}
    """

    def __init__(
        self,
        stt_service: STTService,
        tts_service: TTSService,
        game_command_handler: Optional[GameCommandHandler] = None,
        default_voice: str = "alloy",
        voice_event_queue: Optional[VoiceEventQueue] = None,
        get_board_fn: Optional[GetBoardFn] = None,
        intent_router: Optional[IntentRouter] = None,
        language: str = "en",
    ):
        self.stt = stt_service
        self.tts = tts_service
        self.game_command_handler = game_command_handler
        self.audio_buffer = AudioBuffer()
        self.current_voice = default_voice
        self.voice_event_queue = voice_event_queue
        self.get_board_fn = get_board_fn
        self.intent_router = intent_router
        self.language = language  # "en" or "zh-TW"

        # Listening mode
        self.listening_mode = "push_to_talk"

        # Echo suppression: skip STT while TTS is playing
        self._tts_playing = False

        # Always-on VAD state
        self._vad_buffer = AudioBuffer()
        self._silence_frames = 0
        self._speech_frames = 0
        self._vad_speech_threshold = 0.02   # RMS energy threshold for speech
        self._vad_silence_count = 8         # Consecutive silent chunks to end utterance
        self._vad_min_speech_frames = 3     # Minimum speech frames for valid utterance

        # Connected WebSocket (for broadcasting from event loop)
        self._active_ws: Optional[WebSocket] = None

    def set_voice(self, voice: str) -> None:
        """Set the TTS voice for responses."""
        self.current_voice = voice

    def set_language(self, language: str) -> None:
        """Set the language for STT hints ('en' or 'zh-TW')."""
        self.language = language

    @property
    def _stt_language(self) -> str:
        """Map language setting to Whisper language code."""
        if self.language == "zh-TW":
            return "zh"
        return "en"

    def set_listening_mode(self, mode: str) -> None:
        """Set listening mode: 'push_to_talk' or 'always_on'."""
        if mode in ("push_to_talk", "always_on"):
            self.listening_mode = mode

    @property
    def active_websocket(self) -> Optional[WebSocket]:
        """Get the active WebSocket connection."""
        return self._active_ws

    async def handle_connection(self, websocket: WebSocket) -> None:
        """Main WebSocket connection handler."""
        await websocket.accept()
        self._active_ws = websocket

        # Notify client of current mode
        await websocket.send_json({
            "type": "listening_mode",
            "mode": self.listening_mode,
        })

        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                await self._process_message(websocket, message)

        except WebSocketDisconnect:
            self.audio_buffer.clear()
            self._vad_buffer.clear()
        except Exception as e:
            await self._send_error(websocket, str(e))
        finally:
            if self._active_ws is websocket:
                self._active_ws = None

    async def _process_message(self, websocket: WebSocket, message: dict) -> None:
        """Process incoming WebSocket message."""
        msg_type = message.get("type")

        if msg_type == "audio_chunk":
            audio_data = base64.b64decode(message["data"])
            audio_format = message.get("format", "webm")

            # Unified path: both push-to-talk and always-on (client-side VAD)
            # send complete utterances with is_final=True
            self.audio_buffer.set_format(audio_format)
            self.audio_buffer.add_chunk(audio_data)

            if message.get("is_final"):
                await self._process_audio(websocket)

        elif msg_type == "end_stream":
            await self._process_audio(websocket)

        elif msg_type == "cancel":
            self.audio_buffer.clear()
            self._vad_buffer.clear()
            self._silence_frames = 0
            self._speech_frames = 0
            await websocket.send_json({"type": "cancelled"})

        elif msg_type == "set_mode":
            new_mode = message.get("mode", "push_to_talk")
            self.set_listening_mode(new_mode)
            # Reset VAD state on mode change
            self._vad_buffer.clear()
            self._silence_frames = 0
            self._speech_frames = 0
            await websocket.send_json({
                "type": "listening_mode",
                "mode": self.listening_mode,
            })

        elif msg_type == "set_language":
            new_lang = message.get("language", "en")
            self.set_language(new_lang)
            await websocket.send_json({
                "type": "language_changed",
                "language": self.language,
            })

        elif msg_type == "echo_state":
            self._tts_playing = message.get("speaking", False)

        elif msg_type == "ping":
            await websocket.send_json({"type": "pong"})

    async def _handle_always_on_chunk(
        self, websocket: WebSocket, audio_data: bytes, audio_format: str
    ) -> None:
        """Handle audio chunk in always-on mode with server-side VAD."""
        # Skip processing during TTS playback (echo suppression)
        if self._tts_playing:
            return

        energy = AudioChunkValidator.estimate_speech_energy(audio_data)

        if energy > self._vad_speech_threshold:
            # Speech detected
            self._speech_frames += 1
            self._silence_frames = 0
            self._vad_buffer.set_format(audio_format)
            self._vad_buffer.add_chunk(audio_data)
        else:
            # Silence
            if self._speech_frames > 0:
                self._silence_frames += 1
                # Still buffer a bit of trailing silence
                self._vad_buffer.add_chunk(audio_data)

                if self._silence_frames >= self._vad_silence_count:
                    # End of utterance detected
                    if self._speech_frames >= self._vad_min_speech_frames:
                        # Process the accumulated speech
                        await self._process_vad_audio(websocket)
                    else:
                        # Too short, discard
                        self._vad_buffer.clear()

                    self._speech_frames = 0
                    self._silence_frames = 0

    async def _process_vad_audio(self, websocket: WebSocket) -> None:
        """Process audio accumulated by VAD in always-on mode."""
        audio_bytes = self._vad_buffer.get_audio()
        audio_format = self._vad_buffer.format
        self._vad_buffer.clear()

        if not audio_bytes:
            return

        await self._send_status(websocket, "processing")
        await self._transcribe_and_queue(websocket, audio_bytes, audio_format)
        await self._send_status(websocket, "listening")

    async def _process_audio(self, websocket: WebSocket) -> None:
        """Process buffered audio (push-to-talk mode)."""
        audio_bytes = self.audio_buffer.get_audio()
        audio_format = self.audio_buffer.format
        self.audio_buffer.clear()

        if not audio_bytes:
            return

        await self._send_status(websocket, "processing")

        # If we have an event queue, route through it
        if self.voice_event_queue and self.intent_router:
            await self._transcribe_and_queue(websocket, audio_bytes, audio_format)
            await self._send_status(websocket, "idle")
        else:
            # Legacy path: direct processing (no queue)
            await self._transcribe_and_handle_direct(websocket, audio_bytes, audio_format)
            await self._send_status(websocket, "idle")

    async def _transcribe_and_queue(
        self, websocket: WebSocket, audio_bytes: bytes, audio_format: str
    ) -> None:
        """Transcribe audio and put classified event into queue."""
        # Transcribe with current language hint
        result = await self.stt.transcribe_async(
            audio_bytes, audio_format=audio_format, language=self._stt_language
        )

        if not result["success"]:
            await self._send_error(
                websocket, result.get("error", "Transcription failed")
            )
            return

        transcribed_text = result["text"]

        # Send transcription to client
        await websocket.send_json(
            VoiceMessage(
                type="transcription", text=transcribed_text, is_final=True
            ).to_dict()
        )

        # Classify intent
        board = self.get_board_fn() if self.get_board_fn else None
        intent_result = self.intent_router.classify(transcribed_text, board)

        # Map to priority
        priority_str = IntentRouter.INTENT_PRIORITY.get(intent_result.intent, "LOW")
        priority = _PRIORITY_MAP.get(priority_str, VoiceEventPriority.LOW)

        # Create and queue event
        event = VoiceEvent(
            text=transcribed_text,
            intent=intent_result.intent.value,
            priority=priority,
            data=intent_result.data,
        )
        await self.voice_event_queue.put(event)

    async def _transcribe_and_handle_direct(
        self, websocket: WebSocket, audio_bytes: bytes, audio_format: str
    ) -> None:
        """Legacy path: transcribe and handle directly without queue."""
        result = await self.stt.transcribe_async(
            audio_bytes, audio_format=audio_format, language=self._stt_language
        )

        if not result["success"]:
            await self._send_error(
                websocket, result.get("error", "Transcription failed")
            )
            return

        transcribed_text = result["text"]

        await websocket.send_json(
            VoiceMessage(
                type="transcription", text=transcribed_text, is_final=True
            ).to_dict()
        )

        if self.game_command_handler:
            response = await self.game_command_handler(transcribed_text)

            response_voice = response.get("voice", self.current_voice)
            if response.get("voice"):
                self.current_voice = response["voice"]

            await websocket.send_json(
                VoiceMessage(
                    type="agent_response",
                    text=response.get("text", ""),
                    move_detected=response.get("move"),
                    game_state=response.get("game_state"),
                    voice=response_voice,
                ).to_dict()
            )

            if response.get("text"):
                await self._stream_tts(websocket, response["text"], voice=response_voice)
        else:
            await self._stream_tts(websocket, f"You said: {transcribed_text}")

    async def _stream_tts(
        self, websocket: WebSocket, text: str, voice: Optional[str] = None
    ) -> None:
        """Stream TTS audio to client."""
        tts_voice = voice or self.current_voice
        self._tts_playing = True
        try:
            await self._send_status(websocket, "speaking")
            async for chunk in self.tts.stream_async(
                text, voice=tts_voice, response_format="mp3"
            ):
                await websocket.send_json(
                    VoiceMessage(
                        type="audio_chunk",
                        data=base64.b64encode(chunk).decode("utf-8"),
                        format="mp3",
                        is_final=False,
                    ).to_dict()
                )

            await websocket.send_json(VoiceMessage(type="audio_end").to_dict())

        except Exception as e:
            await self._send_error(websocket, f"TTS error: {e}")
        finally:
            self._tts_playing = False

    async def _send_status(self, websocket: WebSocket, status: str) -> None:
        """Send voice status update to client."""
        try:
            await websocket.send_json({
                "type": "voice_status",
                "status": status,
            })
        except Exception:
            pass

    async def _send_error(self, websocket: WebSocket, message: str) -> None:
        """Send error message to client."""
        try:
            await websocket.send_json(
                VoiceMessage(type="error", error=message).to_dict()
            )
        except Exception:
            pass  # Connection may be closed

    async def speak(
        self, websocket: WebSocket, text: str, voice: Optional[str] = None
    ) -> None:
        """Convenience method to speak text to client."""
        speak_voice = voice or self.current_voice
        await websocket.send_json(
            VoiceMessage(type="agent_response", text=text, voice=speak_voice).to_dict()
        )
        await self._stream_tts(websocket, text, voice=speak_voice)


class VoiceConnectionManager:
    """Manages multiple voice WebSocket connections."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, client_id: str, websocket: WebSocket) -> None:
        """Add a new connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str) -> None:
        """Remove a connection."""
        self.active_connections.pop(client_id, None)

    async def broadcast(self, message: dict) -> None:
        """Send message to all connected clients."""
        for websocket in self.active_connections.values():
            try:
                await websocket.send_json(message)
            except Exception:
                pass

    def get_connection(self, client_id: str) -> Optional[WebSocket]:
        """Get a specific connection."""
        return self.active_connections.get(client_id)
