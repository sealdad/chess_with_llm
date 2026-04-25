"""
CosyVoice TTS provider — calls the remote CosyVoice TTS microservice over HTTP.

Drop-in replacement for TTSService (OpenAI). Same interface:
  synthesize(), stream(), synthesize_async(), stream_async()

Extra feature: pass emotion="happy" etc. to add CosyVoice instruct_text.
OpenAI TTSService ignores this kwarg via **kwargs — zero impact.
"""

import re
import asyncio
from typing import Optional, Generator, AsyncGenerator

import httpx


# Emotion presets → CosyVoice instruct_text (language-dependent)
EMOTION_PRESETS_ZH = {
    "happy":       "You are a helpful assistant. 请用开心愉悦的语气说话。<|endofprompt|>",
    "sad":         "You are a helpful assistant. 请用悲伤低落的语气说话。<|endofprompt|>",
    "angry":       "You are a helpful assistant. 请用生气愤怒的语气说话。<|endofprompt|>",
    "surprised":   "You are a helpful assistant. 请用惊讶的语气说话。<|endofprompt|>",
    "fearful":     "You are a helpful assistant. 请用恐惧害怕的语气说话。<|endofprompt|>",
    "disgusted":   "You are a helpful assistant. 请用厌恶不满的语气说话。<|endofprompt|>",
    "calm":        "You are a helpful assistant. 请用冷静平和的语气说话。<|endofprompt|>",
    "serious":     "You are a helpful assistant. 请用严肃认真的语气说话。<|endofprompt|>",
    "gentle":      "You are a helpful assistant. 请用温柔亲切的语气说话。<|endofprompt|>",
    "encouraging": "You are a helpful assistant. 请用热情鼓励的语气说话。<|endofprompt|>",
}
EMOTION_PRESETS_EN = {
    "happy":       "You are a helpful assistant. Please speak in a happy and cheerful tone.<|endofprompt|>",
    "sad":         "You are a helpful assistant. Please speak in a sad and melancholic tone.<|endofprompt|>",
    "angry":       "You are a helpful assistant. Please speak in an angry and frustrated tone.<|endofprompt|>",
    "surprised":   "You are a helpful assistant. Please speak in a surprised and astonished tone.<|endofprompt|>",
    "fearful":     "You are a helpful assistant. Please speak in a fearful and anxious tone.<|endofprompt|>",
    "disgusted":   "You are a helpful assistant. Please speak in a disgusted and disapproving tone.<|endofprompt|>",
    "calm":        "You are a helpful assistant. Please speak in a calm and composed tone.<|endofprompt|>",
    "serious":     "You are a helpful assistant. Please speak in a serious and stern tone.<|endofprompt|>",
    "gentle":      "You are a helpful assistant. Please speak in a gentle and warm tone.<|endofprompt|>",
    "encouraging": "You are a helpful assistant. Please speak in an encouraging and enthusiastic tone.<|endofprompt|>",
}
EMOTION_PRESETS = EMOTION_PRESETS_ZH  # legacy fallback


class CosyVoiceTTSProvider:
    """TTS provider that calls the CosyVoice microservice."""

    VOICES = ["default"]  # populated dynamically from the service
    FORMATS = ["mp3", "wav"]

    # Language → voice_id mapping for auto-detection
    LANG_VOICES = {
        "zh": "zh_default",
        "en": "en_default",
    }

    def __init__(
        self,
        service_url: str = "http://localhost:8003",
        default_voice: str = "default",
        timeout: float = 30.0,
        language: str = "",
    ):
        self.service_url = service_url.rstrip("/")
        self.default_voice = default_voice
        self.timeout = timeout
        self.language = language  # "zh-TW", "en", or "" (auto-detect from text)
        self._sync_client = None
        self._async_client = None

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(timeout=self.timeout)
        return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)
        return self._async_client

    @staticmethod
    def _detect_language(text: str) -> str:
        """Detect language from text. Returns 'zh' or 'en'."""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        return "zh" if chinese_chars > len(text) * 0.1 else "en"

    def _resolve_voice(self, text: str, voice: Optional[str]) -> str:
        """Pick voice_id: explicit voice > language setting > auto-detect from text."""
        if voice and voice != self.default_voice:
            return voice
        # Use agent language setting if available
        if self.language:
            lang = "zh" if "zh" in self.language else "en"
            return self.LANG_VOICES.get(lang, self.default_voice)
        # Fallback: auto-detect from text content
        lang = self._detect_language(text)
        return self.LANG_VOICES.get(lang, self.default_voice)

    def _resolve_emotion(self, emotion: Optional[str]) -> str:
        """Convert emotion name to CosyVoice instruct_text (language-aware)."""
        if not emotion:
            return ""
        is_zh = self.language and "zh" in self.language
        presets = EMOTION_PRESETS_ZH if is_zh else EMOTION_PRESETS_EN
        return presets.get(emotion, "")

    def _build_payload(
        self,
        text: str,
        voice: Optional[str],
        response_format: str,
        stream: bool,
        emotion: Optional[str] = None,
    ) -> dict:
        payload = {
            "text": text,
            "voice_id": self._resolve_voice(text, voice),
            "output_format": response_format,
            "stream": stream,
        }
        instruct = self._resolve_emotion(emotion)
        if instruct:
            payload["instruct_text"] = instruct
        return payload

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: str = "mp3",
        **kwargs,
    ) -> bytes:
        """Synchronous synthesis — returns complete audio bytes."""
        client = self._get_sync_client()
        payload = self._build_payload(
            text, voice, response_format, False, kwargs.get("emotion")
        )
        resp = client.post(f"{self.service_url}/synthesize", json=payload)
        resp.raise_for_status()
        return resp.content

    def stream(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: str = "mp3",
        chunk_size: int = 4096,
        **kwargs,
    ) -> Generator[bytes, None, None]:
        """Synchronous streaming — yields audio chunks."""
        client = self._get_sync_client()
        payload = self._build_payload(
            text, voice, response_format, True, kwargs.get("emotion")
        )
        with client.stream("POST", f"{self.service_url}/synthesize/stream", json=payload) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_bytes(chunk_size=chunk_size):
                yield chunk

    async def synthesize_async(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: str = "mp3",
        **kwargs,
    ) -> bytes:
        """Async synthesis — returns complete audio bytes."""
        client = self._get_async_client()
        payload = self._build_payload(
            text, voice, response_format, False, kwargs.get("emotion")
        )
        resp = await client.post(f"{self.service_url}/synthesize", json=payload)
        resp.raise_for_status()
        return resp.content

    async def stream_async(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: str = "mp3",
        chunk_size: int = 4096,
        **kwargs,
    ) -> AsyncGenerator[bytes, None]:
        """Async streaming — yields audio chunks."""
        client = self._get_async_client()
        payload = self._build_payload(
            text, voice, response_format, True, kwargs.get("emotion")
        )
        async with client.stream("POST", f"{self.service_url}/synthesize/stream", json=payload) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes(chunk_size=chunk_size):
                yield chunk
                await asyncio.sleep(0)

    def get_voice_info(self) -> dict:
        """Get available voices from the service."""
        try:
            client = self._get_sync_client()
            resp = client.get(f"{self.service_url}/voices")
            resp.raise_for_status()
            data = resp.json()
            return {v["voice_id"]: v["type"] for v in data.get("voices", [])}
        except Exception:
            return {"default": "Default CosyVoice speaker"}
