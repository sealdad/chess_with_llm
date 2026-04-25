"""
Text-to-Speech service using OpenAI TTS API with streaming support.
"""

import os
import asyncio
from typing import Optional, Generator, AsyncGenerator


class TTSService:
    """Handles text-to-speech conversion via OpenAI TTS."""

    VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    FORMATS = ["mp3", "opus", "aac", "flac", "wav", "pcm"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_voice: str = "alloy",
        model: str = "tts-1",
    ):
        """
        Initialize TTS service.

        Args:
            api_key: OpenAI API key (required — pulled from yaml settings by caller).
            default_voice: Default voice to use (alloy, echo, fable, onyx, nova, shimmer)
            model: TTS model to use (tts-1 for speed, tts-1-hd for quality)
        """
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.default_voice = default_voice
        self.model = model

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: str = "mp3",
        **kwargs,  # absorbs extra params like emotion (CosyVoice only)
    ) -> bytes:
        """
        Convert text to audio (non-streaming).

        Args:
            text: Text to convert to speech
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            speed: Speed factor (0.25 to 4.0)
            response_format: Output format (mp3, opus, aac, flac, wav, pcm)

        Returns:
            Audio bytes in specified format
        """
        response = self.client.audio.speech.create(
            model=self.model,
            voice=voice or self.default_voice,
            input=text,
            speed=speed,
            response_format=response_format,
        )
        return response.content

    def stream(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: str = "mp3",
        chunk_size: int = 4096,
        **kwargs,
    ) -> Generator[bytes, None, None]:
        """
        Stream audio chunks for real-time playback.

        Args:
            text: Text to convert to speech
            voice: Voice to use
            speed: Speed factor (0.25 to 4.0)
            response_format: Output format
            chunk_size: Size of each audio chunk in bytes

        Yields:
            Audio data chunks as bytes
        """
        response = self.client.audio.speech.create(
            model=self.model,
            voice=voice or self.default_voice,
            input=text,
            speed=speed,
            response_format=response_format,
        )

        for chunk in response.iter_bytes(chunk_size=chunk_size):
            yield chunk

    async def synthesize_async(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: str = "mp3",
        **kwargs,
    ) -> bytes:
        """
        Async version of synthesize.

        Returns:
            Audio bytes in specified format
        """
        return await asyncio.to_thread(
            self.synthesize, text, voice, speed, response_format
        )

    async def stream_async(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: str = "mp3",
        chunk_size: int = 4096,
        **kwargs,
    ) -> AsyncGenerator[bytes, None]:
        """
        Async generator for streaming audio chunks.

        Note: OpenAI SDK is sync, so we run the API call in a thread
        and yield chunks with async sleep to allow other tasks.

        Args:
            text: Text to convert to speech
            voice: Voice to use
            speed: Speed factor
            response_format: Output format
            chunk_size: Size of each audio chunk

        Yields:
            Audio data chunks as bytes
        """
        # Get the full response in a thread
        response = await asyncio.to_thread(
            self.client.audio.speech.create,
            model=self.model,
            voice=voice or self.default_voice,
            input=text,
            speed=speed,
            response_format=response_format,
        )

        # Iterate over chunks, yielding control between each
        for chunk in response.iter_bytes(chunk_size=chunk_size):
            yield chunk
            await asyncio.sleep(0)  # Yield control to event loop

    def get_voice_info(self) -> dict:
        """
        Get information about available voices.

        Returns:
            dict with voice names and descriptions
        """
        return {
            "alloy": "Neutral and balanced",
            "echo": "Warm and engaging",
            "fable": "Authoritative and clear",
            "onyx": "Deep and resonant",
            "nova": "Friendly and upbeat",
            "shimmer": "Soft and expressive",
        }
