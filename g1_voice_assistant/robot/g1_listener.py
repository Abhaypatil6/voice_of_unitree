"""
robot/g1_listener.py
--------------------
Stage 1 – Unitree G1 /audio_msg subscriber.

Subscribes to the Unitree speech-recognition topic, parses the JSON
payload, applies basic quality filters, and saves persistent artifacts
for every accepted transcript.

DESIGN NOTES
~~~~~~~~~~~~
* The G1 publishes ``std_msgs/msg/String`` messages on ``/audio_msg``.
  The ``data`` field contains a JSON object, e.g.:

      {
        "index": 236,
        "timestamp": 1786166118025,
        "type": 0,
        "text": "Hello, who are you?",
        "angle": 0,
        "speaker_id": 0,
        "emotion": "<|NEUTRAL|>",
        "confidence": 0.92
      }

* Duplicate detection is done using the ``index`` field from Unitree.
  If no ``index`` is present we fall back to a hash of text+timestamp.

* Each unique accepted message triggers ``on_transcript_callback``.

* This module is ROS-agnostic in its public interface: it can be
  instantiated in a plain Python test by providing a mock node or by
  calling the internal ``_handle_raw_json`` method directly.

Python 3.8 compatible.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, Optional, Set, Tuple

from logging_utils.logger import get_logger
from logging_utils.artifact_logger import ArtifactLogger

logger = get_logger(__name__)


class G1Listener:
    """
    Subscribes to ``/audio_msg`` and fires a callback for each unique,
    non-empty transcript.

    Parameters
    ----------
    artifact_logger:
        :class:`ArtifactLogger` instance for the current session.
    on_transcript_callback:
        Callable invoked with the parsed audio message dict and the
        extracted transcript string whenever a valid new transcript
        is received.
    min_text_length:
        Minimum character count to accept a transcript.
    min_confidence:
        Minimum confidence value (0–1) to accept a transcript.
        0.0 accepts everything.
    topic:
        ROS topic name (default ``/audio_msg``).
    """

    def __init__(
        self,
        artifact_logger: ArtifactLogger,
        min_text_length: int = 2,
        min_confidence: float = 0.0,
        topic: str = "/audio_msg",
    ) -> None:
        self._artifact_logger = artifact_logger
        self._min_text_length = min_text_length
        self._min_confidence = min_confidence
        self._topic = topic

        import threading
        self._lock = threading.Lock()
        self._is_listening = True
        
        self._current_payload: Optional[Dict[str, Any]] = None
        self._current_text: str = ""
        self._current_index: Optional[int] = None
        self._current_received_ts: float = 0.0
        self._last_update_ts: float = 0.0

        self._seen_indices: Set[int] = set()
        self._total_received: int = 0
        self._total_accepted: int = 0

        logger.info(
            "G1Listener initialised (topic=%s, min_len=%d, min_conf=%.2f)",
            topic,
            min_text_length,
            min_confidence,
        )

    def set_listening(self, state: bool) -> None:
        """Enable or disable processing of incoming STT messages."""
        with self._lock:
            self._is_listening = state
            if state:
                # Clear buffer when resuming
                self._current_payload = None
                self._current_text = ""
                self._current_received_ts = 0.0
                self._last_update_ts = 0.0
                
    def get_final_transcript(
        self,
        silence_timeout: float = 1.2,
    ) -> Tuple[Optional[Dict[str, Any]], str, float]:
        """
        Check if the current utterance has finished (no new updates for `silence_timeout` seconds).
        Returns (payload, text, local_received_ts). If not finalized or empty, returns (None, "", 0.0).
        """
        with self._lock:
            if not self._current_text:
                return None, "", 0.0
                
            if (time.time() - self._last_update_ts) > silence_timeout:
                payload = self._current_payload
                text = self._current_text
                received_ts = self._current_received_ts
                current_index = self._current_index
                
                # Remember this turn's index so repeated stale payloads are ignored.
                if current_index is not None:
                    self._seen_indices.add(current_index)

                # Clear buffer
                self._current_payload = None
                self._current_text = ""
                self._current_index = None
                self._current_received_ts = 0.0
                self._last_update_ts = 0.0
                
                return payload, text, received_ts
                
        return None, "", 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # ROS callback
    # ──────────────────────────────────────────────────────────────────────────

    def ros_callback(self, msg: Any) -> None:
        """
        ROS subscriber callback for ``std_msgs/msg/String``.

        ``msg.data`` must be a JSON string.
        """
        received_ts = time.time()
        raw_data: str = getattr(msg, "data", "")
        self._handle_raw_json(raw_data, received_ts)

    # ──────────────────────────────────────────────────────────────────────────
    # Public testable method
    # ──────────────────────────────────────────────────────────────────────────

    def _handle_raw_json(self, raw_data: str, received_ts: Optional[float] = None) -> bool:
        """
        Parse, filter, and dispatch a raw JSON string.

        This method is intentionally public and testable without ROS.

        Parameters
        ----------
        raw_data:
            The JSON string from ``/audio_msg``.
        received_ts:
            Local epoch timestamp (defaults to ``time.time()``).

        Returns
        -------
        bool
            ``True`` if the transcript was accepted and dispatched.
        """
        if received_ts is None:
            received_ts = time.time()

        with self._lock:
            if not self._is_listening:
                return False

        self._total_received += 1

        # ── Step 1: Parse JSON ─────────────────────────────────────────────
        try:
            payload: Dict[str, Any] = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            logger.warning("Malformed JSON on %s: %s | raw: %.80r", self._topic, exc, raw_data)
            return False

        # ── Step 2: Extract fields ─────────────────────────────────────────
        unitree_index: Optional[int] = payload.get("index")
        unitree_ts: Optional[int] = payload.get("timestamp")
        text: str = (payload.get("text") or "").strip()
        confidence: float = float(payload.get("confidence") or 0.0)
        speaker_id: int = int(payload.get("speaker_id") or 0)
        angle: float = float(payload.get("angle") or 0.0)
        emotion: str = str(payload.get("emotion") or "")

        logger.debug(
            "Received (idx=%s ts=%s): %r  conf=%.3f",
            unitree_index,
            unitree_ts,
            text[:80],
            confidence,
        )

        # ── Step 3: Duplicate detection / current utterance updates ──────
        if unitree_index is not None:
            if unitree_index == self._current_index:
                if text == self._current_text:
                    logger.debug("Duplicate current STT update, skipping")
                    return False
                logger.debug("Updated STT for current in-flight index %d", unitree_index)
            elif unitree_index in self._seen_indices:
                logger.debug("Stale STT index %d after finalization, skipping", unitree_index)
                return False

        # ── Step 4: Quality filters ────────────────────────────────────────
        rejection_reason: Optional[str] = None

        if len(text) < self._min_text_length:
            rejection_reason = f"text too short ({len(text)} < {self._min_text_length})"
        elif self._min_confidence > 0.0 and confidence < self._min_confidence:
            rejection_reason = (
                f"confidence too low ({confidence:.3f} < {self._min_confidence:.3f})"
            )

        if rejection_reason:
            logger.info("Transcript rejected – %s | text: %r", rejection_reason, text)
            # Still save a raw artifact so the developer can inspect rejections
            self._artifact_logger.log_stt(
                raw_msg={**payload, "_rejection_reason": rejection_reason},
                transcript="[REJECTED] " + text,
                local_received_ts=received_ts,
            )
            return False

        # ── Step 5: Save artifact ──────────────────────────────────────────
        counter = self._artifact_logger.log_stt(
            raw_msg=payload,
            transcript=text,
            local_received_ts=received_ts,
        )

        self._total_accepted += 1
        logger.debug(
            "[STT Partial] '%s'  (idx=%s, conf=%.3f)",
            text[:80],
            unitree_index,
            confidence,
        )

        # ── Step 6: Buffer as current utterance ────────────────────────────
        with self._lock:
            self._current_payload = payload
            self._current_text = text
            self._current_index = unitree_index
            self._current_received_ts = received_ts
            self._last_update_ts = received_ts

        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Statistics
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, int]:
        """Return basic listener statistics."""
        return {
            "total_received": self._total_received,
            "total_accepted": self._total_accepted,
            "total_rejected": self._total_received - self._total_accepted,
        }
