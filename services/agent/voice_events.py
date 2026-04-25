"""
Voice event queue for asynchronous voice-to-game communication.

Events flow: VoiceHandler -> STT -> IntentRouter -> VoiceEventQueue -> VoiceEventProcessor
"""

import asyncio
import time
from enum import IntEnum
from typing import Optional
from dataclasses import dataclass, field


class VoiceEventPriority(IntEnum):
    """Priority levels for voice events. Lower value = higher priority."""
    HIGH = 0      # Game commands: pause, resume, stop, resign
    NORMAL = 1    # Moves, mode switches
    LOW = 2       # Conversation, chitchat
    IGNORE = 3    # Filler words, noise, hallucinations


@dataclass
class VoiceEvent:
    """A classified voice input event."""
    text: str
    intent: str                          # Intent enum value string
    priority: VoiceEventPriority
    data: Optional[dict] = None
    timestamp: float = field(default_factory=time.time)


class VoiceEventQueue:
    """
    Async priority queue for voice events.

    High-priority events (game commands) are drained first.
    """

    def __init__(self, maxsize: int = 50):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._has_high_priority = False

    async def put(self, event: VoiceEvent) -> None:
        """Add an event to the queue. Drops if full."""
        if event.priority == VoiceEventPriority.IGNORE:
            return
        if event.priority == VoiceEventPriority.HIGH:
            self._has_high_priority = True
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            pass  # Drop oldest-style: caller can log if needed

    async def get_all(self) -> list:
        """
        Drain all pending events, sorted by priority (high first).

        Returns:
            List of VoiceEvent sorted by priority then timestamp.
        """
        events = []
        while not self._queue.empty():
            try:
                events.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        self._has_high_priority = False
        events.sort(key=lambda e: (e.priority, e.timestamp))
        return events

    def has_high_priority(self) -> bool:
        """Check if there are pending high-priority events."""
        return self._has_high_priority

    @property
    def pending(self) -> int:
        """Number of pending events."""
        return self._queue.qsize()
