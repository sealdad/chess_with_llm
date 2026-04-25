"""
Conversation context manager for maintaining chat history.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    """A single conversation message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dict for API calls."""
        return {
            "role": self.role,
            "content": self.content,
        }


class ConversationContext:
    """
    Manages conversation history and context for the agent.

    Maintains a sliding window of messages and provides
    context summarization for long conversations.
    """

    def __init__(self, max_messages: int = 20):
        """
        Initialize conversation context.

        Args:
            max_messages: Maximum messages to keep in history
        """
        self.max_messages = max_messages
        self.messages: List[Message] = []
        self.session_start = datetime.now()

    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        self.messages.append(Message(role="user", content=content))
        self._trim_history()

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to history."""
        self.messages.append(Message(role="assistant", content=content))
        self._trim_history()

    def add_exchange(self, user_content: str, assistant_content: str) -> None:
        """Add a complete user-assistant exchange."""
        self.add_user_message(user_content)
        self.add_assistant_message(assistant_content)

    def _trim_history(self) -> None:
        """Trim history to max_messages, keeping recent messages."""
        if len(self.messages) > self.max_messages:
            # Keep the most recent messages
            self.messages = self.messages[-self.max_messages:]

    def get_history(self) -> List[Dict]:
        """Get conversation history as list of dicts."""
        return [msg.to_dict() for msg in self.messages]

    def get_recent_history(self, n: int = 10) -> List[Dict]:
        """Get last N messages as list of dicts."""
        recent = self.messages[-n:] if len(self.messages) >= n else self.messages
        return [msg.to_dict() for msg in recent]

    def clear(self) -> None:
        """Clear all conversation history."""
        self.messages = []
        self.session_start = datetime.now()

    def get_last_user_message(self) -> Optional[str]:
        """Get the last user message content."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None

    def get_last_assistant_message(self) -> Optional[str]:
        """Get the last assistant message content."""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return None

    @property
    def message_count(self) -> int:
        """Total number of messages in history."""
        return len(self.messages)

    @property
    def session_duration(self) -> float:
        """Session duration in seconds."""
        return (datetime.now() - self.session_start).total_seconds()

    def to_dict(self) -> Dict:
        """Export context as dict."""
        return {
            "messages": self.get_history(),
            "message_count": self.message_count,
            "session_start": self.session_start.isoformat(),
            "session_duration_seconds": self.session_duration,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ConversationContext":
        """Create context from dict."""
        ctx = cls()
        for msg in data.get("messages", []):
            if msg["role"] == "user":
                ctx.add_user_message(msg["content"])
            else:
                ctx.add_assistant_message(msg["content"])
        return ctx


class ConversationManager:
    """
    Manages multiple conversation contexts (e.g., per game or per user).
    """

    def __init__(self):
        self.contexts: Dict[str, ConversationContext] = {}

    def get_context(self, session_id: str) -> ConversationContext:
        """Get or create a conversation context for a session."""
        if session_id not in self.contexts:
            self.contexts[session_id] = ConversationContext()
        return self.contexts[session_id]

    def clear_context(self, session_id: str) -> None:
        """Clear a specific session's context."""
        if session_id in self.contexts:
            self.contexts[session_id].clear()

    def remove_context(self, session_id: str) -> None:
        """Remove a session's context entirely."""
        self.contexts.pop(session_id, None)

    def clear_all(self) -> None:
        """Clear all contexts."""
        self.contexts.clear()
