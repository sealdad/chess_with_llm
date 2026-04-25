"""
Speech-to-Text service using OpenAI Whisper API.
"""

import io
import os
import asyncio
from typing import Optional


class STTService:
    """Handles speech-to-text conversion via OpenAI Whisper."""

    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-1"):
        """
        Initialize STT service.

        Args:
            api_key: OpenAI API key (required — pulled from yaml settings by caller).
            model: Whisper model to use (default: whisper-1)
        """
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def transcribe(
        self,
        audio_bytes: bytes,
        audio_format: str = "wav",
        language: str = "en",
    ) -> dict:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio data
            audio_format: Format of audio (wav, mp3, webm, m4a, ogg, flac)
            language: Language hint for better accuracy

        Returns:
            dict with 'success', 'text', and optional 'error'
        """
        try:
            # Create a file-like object with proper name for format detection
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = f"audio.{audio_format}"

            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language=language,
                response_format="text",
            )

            return {"success": True, "text": response.strip()}

        except Exception as e:
            return {"success": False, "error": str(e), "text": ""}

    async def transcribe_async(
        self,
        audio_bytes: bytes,
        audio_format: str = "wav",
        language: str = "en",
    ) -> dict:
        """
        Async version of transcribe using asyncio.to_thread.

        Args:
            audio_bytes: Raw audio data
            audio_format: Format of audio
            language: Language hint

        Returns:
            dict with 'success', 'text', and optional 'error'
        """
        return await asyncio.to_thread(
            self.transcribe, audio_bytes, audio_format, language
        )

    def transcribe_with_timestamps(
        self,
        audio_bytes: bytes,
        audio_format: str = "wav",
        language: str = "en",
    ) -> dict:
        """
        Transcribe with word-level timestamps.

        Args:
            audio_bytes: Raw audio data
            audio_format: Format of audio
            language: Language hint

        Returns:
            dict with 'success', 'text', 'words' (with timestamps), and optional 'error'
        """
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = f"audio.{audio_format}"

            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )

            return {
                "success": True,
                "text": response.text,
                "words": response.words if hasattr(response, "words") else [],
                "duration": response.duration if hasattr(response, "duration") else 0,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "text": "", "words": []}
