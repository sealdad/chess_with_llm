"""
Audio buffer for accumulating chunked audio input.
"""

import io
import wave
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class AudioBuffer:
    """
    Buffer for accumulating audio chunks from streaming input.

    Supports multiple audio formats and provides utilities for
    format conversion and validation.
    """

    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # 16-bit audio

    _chunks: list = field(default_factory=list)
    _format: str = "pcm16"

    def add_chunk(self, chunk: bytes) -> None:
        """
        Add an audio chunk to the buffer.

        Args:
            chunk: Raw audio bytes
        """
        self._chunks.append(chunk)

    def get_audio(self) -> bytes:
        """
        Get all buffered audio as a single bytes object.

        Returns:
            Combined audio data
        """
        if not self._chunks:
            return b""
        return b"".join(self._chunks)

    def get_audio_as_wav(self) -> bytes:
        """
        Get buffered audio as WAV format.

        Assumes input is raw PCM 16-bit mono audio.

        Returns:
            WAV formatted audio bytes
        """
        raw_audio = self.get_audio()
        if not raw_audio:
            return b""

        # Create WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.sample_width)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(raw_audio)

        wav_buffer.seek(0)
        return wav_buffer.read()

    def clear(self) -> None:
        """Clear the buffer."""
        self._chunks = []

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._chunks) == 0

    def duration_seconds(self) -> float:
        """
        Estimate duration of buffered audio in seconds.

        Returns:
            Estimated duration based on sample rate and format
        """
        total_bytes = sum(len(chunk) for chunk in self._chunks)
        bytes_per_second = self.sample_rate * self.channels * self.sample_width
        return total_bytes / bytes_per_second if bytes_per_second > 0 else 0

    @property
    def chunk_count(self) -> int:
        """Number of chunks in buffer."""
        return len(self._chunks)

    @property
    def total_bytes(self) -> int:
        """Total bytes in buffer."""
        return sum(len(chunk) for chunk in self._chunks)

    def set_format(self, format_type: str) -> None:
        """
        Set the expected input format.

        Args:
            format_type: Audio format (pcm16, webm, mp3, etc.)
        """
        self._format = format_type

    @property
    def format(self) -> str:
        """Get current format."""
        return self._format


class AudioChunkValidator:
    """Validates audio chunks for common issues."""

    @staticmethod
    def is_silent(chunk: bytes, threshold: int = 100) -> bool:
        """
        Check if audio chunk is mostly silent.

        Args:
            chunk: Audio bytes (assumed 16-bit PCM)
            threshold: Maximum average amplitude to consider silent

        Returns:
            True if chunk appears to be silent
        """
        if len(chunk) < 2:
            return True

        # Convert to 16-bit samples
        samples = []
        for i in range(0, len(chunk) - 1, 2):
            sample = int.from_bytes(chunk[i : i + 2], byteorder="little", signed=True)
            samples.append(abs(sample))

        if not samples:
            return True

        avg_amplitude = sum(samples) / len(samples)
        return avg_amplitude < threshold

    @staticmethod
    def estimate_speech_energy(chunk: bytes) -> float:
        """
        Estimate speech energy in a chunk.

        Args:
            chunk: Audio bytes (assumed 16-bit PCM)

        Returns:
            RMS energy value (0.0 to 1.0 normalized)
        """
        if len(chunk) < 2:
            return 0.0

        # Convert to 16-bit samples and calculate RMS
        sum_squares = 0
        count = 0

        for i in range(0, len(chunk) - 1, 2):
            sample = int.from_bytes(chunk[i : i + 2], byteorder="little", signed=True)
            sum_squares += sample * sample
            count += 1

        if count == 0:
            return 0.0

        rms = (sum_squares / count) ** 0.5
        # Normalize to 0-1 range (max 16-bit value is 32767)
        return min(1.0, rms / 32767.0)
