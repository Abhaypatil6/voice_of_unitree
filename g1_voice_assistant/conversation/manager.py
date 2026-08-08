"""
conversation/manager.py
-----------------------
Conversation history manager.

Maintains a rolling window of user/assistant message pairs.
The manager is stateful for the duration of a single session.

Python 3.8 compatible.
"""

from __future__ import annotations

from typing import Dict, List

from logging_utils.logger import get_logger

logger = get_logger(__name__)


class ConversationManager:
    """
    Manages rolling conversation history for the LLM context window.

    Parameters
    ----------
    max_history_messages:
        Maximum number of user+assistant PAIRS to keep.
        When the limit is reached the oldest pair is dropped.
        Example: 10 → up to 20 individual messages retained.
    """

    def __init__(self, max_history_messages: int = 10) -> None:
        self._max_pairs = max_history_messages
        self._history: List[Dict[str, str]] = []
        logger.info("ConversationManager ready (max_pairs=%d)", max_history_messages)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def history(self) -> List[Dict[str, str]]:
        """Current history snapshot (copy – callers may not mutate)."""
        return list(self._history)

    def add_exchange(self, user_text: str, assistant_text: str) -> None:
        """
        Append a completed user/assistant exchange to the history.

        Prunes the oldest pair when the window is full.

        Parameters
        ----------
        user_text:
            The user's transcript for this turn.
        assistant_text:
            The assistant's generated response.
        """
        self._history.append({"role": "user",      "content": user_text})
        self._history.append({"role": "assistant", "content": assistant_text})

        # Prune: each pair = 2 messages
        max_messages = self._max_pairs * 2
        if len(self._history) > max_messages:
            dropped = len(self._history) - max_messages
            self._history = self._history[dropped:]
            logger.debug("History pruned: dropped %d old messages", dropped)

        logger.debug(
            "History updated: %d messages (%d pairs)",
            len(self._history),
            len(self._history) // 2,
        )

    def clear(self) -> None:
        """Reset the conversation history (e.g. on session start)."""
        self._history.clear()
        logger.info("Conversation history cleared")

    def __len__(self) -> int:
        return len(self._history)
