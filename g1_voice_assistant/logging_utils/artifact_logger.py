"""
logging_utils/artifact_logger.py
---------------------------------
Persistent artifact writer for the G1 Voice Assistant.

Each pipeline run (STT → LLM → TTS → Speaker) produces a unique
session directory with structured sub-directories and JSON/text files.
This module handles ALL file I/O for the artifact system.

Directory layout:

    conversations/
    └── 2026-08-08_12-30-15/
        ├── session.json
        ├── 01_stt/
        │   ├── raw_000001.json          ← original /audio_msg payload
        │   └── transcript_000001.txt   ← extracted text
        ├── 02_llm/
        │   ├── request_000001.json
        │   └── response_000001.json
        ├── 03_tts/
        │   ├── audio_000001.wav
        │   └── metadata_000001.json
        └── 04_speaker/
            └── playback_000001.json

Python 3.8 compatible.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from logging_utils.logger import get_logger

logger = get_logger(__name__)


class ArtifactLogger:
    """
    Creates and manages one timestamped session directory and writes
    structured artifacts for every pipeline stage.

    Parameters
    ----------
    base_dir:
        Root conversations directory (e.g. ``./conversations``).
    session_ts:
        Optional override for the session timestamp string.
        Defaults to the current local time formatted as
        ``YYYY-MM-DD_HH-MM-SS``.
    """

    def __init__(
        self,
        base_dir: Path,
        session_ts: Optional[str] = None,
    ) -> None:
        ts = session_ts or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir: Path = Path(base_dir) / ts

        self._stt_dir: Path = self.session_dir / "01_stt"
        self._llm_dir: Path = self.session_dir / "02_llm"
        self._tts_dir: Path = self.session_dir / "03_tts"
        self._speaker_dir: Path = self.session_dir / "04_speaker"

        for d in (self._stt_dir, self._llm_dir, self._tts_dir, self._speaker_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._counter: int = 0
        self._session_start: float = time.time()

        # Write initial session.json
        self._write_session_json()
        logger.info("ArtifactLogger session: %s", self.session_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _next_counter(self) -> int:
        self._counter += 1
        return self._counter

    @staticmethod
    def _write_json(path: Path, data: Dict[str, Any]) -> None:
        """Atomically write *data* as JSON to *path*."""
        try:
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error("Failed to write artifact %s: %s", path, exc)

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        """Write *text* to *path*."""
        try:
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            logger.error("Failed to write artifact %s: %s", path, exc)

    def _write_session_json(self) -> None:
        self._write_json(
            self.session_dir / "session.json",
            {
                "session_dir": str(self.session_dir),
                "started_at": datetime.now().isoformat(),
                "started_epoch": self._session_start,
            },
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 1 – STT artifacts
    # ──────────────────────────────────────────────────────────────────────────

    def log_stt(
        self,
        raw_msg: Dict[str, Any],
        transcript: str,
        local_received_ts: float,
    ) -> int:
        """
        Save raw /audio_msg JSON and extracted transcript.

        Parameters
        ----------
        raw_msg:
            The complete original JSON dict from /audio_msg.
        transcript:
            Extracted ``text`` field (may be empty if absent).
        local_received_ts:
            ``time.time()`` value when the message was received locally.

        Returns
        -------
        int
            The counter value used for this entry (for cross-referencing).
        """
        n = self._next_counter()
        idx = f"{n:06d}"

        # raw_XXXXXX.json  – never modify the original payload
        self._write_json(
            self._stt_dir / f"raw_{idx}.json",
            {
                "local_received_epoch": local_received_ts,
                "local_received_iso": datetime.fromtimestamp(local_received_ts).isoformat(),
                "raw_audio_msg": raw_msg,
            },
        )

        # transcript_XXXXXX.txt  – plain text for easy grep
        self._write_text(
            self._stt_dir / f"transcript_{idx}.txt",
            transcript,
        )

        logger.debug("STT artifact %s saved (%d chars)", idx, len(transcript))
        return n

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 2 – LLM artifacts
    # ──────────────────────────────────────────────────────────────────────────

    def log_llm_request(
        self,
        counter: int,
        transcript: str,
        history: list,
        system_prompt: str,
        model: str,
        request_ts: float,
    ) -> None:
        """Save the LLM request payload (without API key)."""
        idx = f"{counter:06d}"
        self._write_json(
            self._llm_dir / f"request_{idx}.json",
            {
                "counter": counter,
                "request_iso": datetime.fromtimestamp(request_ts).isoformat(),
                "request_epoch": request_ts,
                "model": model,
                "system_prompt": system_prompt,
                "history_messages": history,
                "user_transcript": transcript,
                # API key intentionally omitted
            },
        )

    def log_llm_response(
        self,
        counter: int,
        response_text: str,
        latency_s: float,
        token_usage: Optional[Dict[str, Any]],
        error: Optional[str],
        http_status: Optional[int] = None,
    ) -> None:
        """Save the LLM response."""
        idx = f"{counter:06d}"
        self._write_json(
            self._llm_dir / f"response_{idx}.json",
            {
                "counter": counter,
                "response_iso": datetime.now().isoformat(),
                "response_text": response_text,
                "latency_s": round(latency_s, 3),
                "token_usage": token_usage,
                "http_status": http_status,
                "error": error,
            },
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 3 – TTS artifacts
    # ──────────────────────────────────────────────────────────────────────────

    def log_tts(
        self,
        counter: int,
        input_text: str,
        audio_path: Optional[Path],
        tts_backend: str,
        sample_rate: int,
        duration_s: Optional[float],
        latency_s: float,
        error: Optional[str],
    ) -> None:
        """Save TTS metadata."""
        idx = f"{counter:06d}"
        self._write_json(
            self._tts_dir / f"metadata_{idx}.json",
            {
                "counter": counter,
                "generated_iso": datetime.now().isoformat(),
                "input_text": input_text,
                "audio_file": str(audio_path) if audio_path else None,
                "tts_backend": tts_backend,
                "sample_rate": sample_rate,
                "duration_s": duration_s,
                "latency_s": round(latency_s, 3),
                "error": error,
            },
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 4 – Speaker/playback artifacts
    # ──────────────────────────────────────────────────────────────────────────

    def log_speaker(
        self,
        counter: int,
        audio_file: Optional[Path],
        playback_start: float,
        playback_end: float,
        success: bool,
        error: Optional[str],
    ) -> None:
        """Save playback metadata."""
        idx = f"{counter:06d}"
        self._write_json(
            self._speaker_dir / f"playback_{idx}.json",
            {
                "counter": counter,
                "audio_file": str(audio_file) if audio_file else None,
                "playback_start_iso": datetime.fromtimestamp(playback_start).isoformat(),
                "playback_end_iso": datetime.fromtimestamp(playback_end).isoformat(),
                "duration_s": round(playback_end - playback_start, 3),
                "success": success,
                "error": error,
            },
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Finalise session
    # ──────────────────────────────────────────────────────────────────────────

    def finalise(self) -> None:
        """Update session.json with end-of-session info."""
        end_ts = time.time()
        self._write_json(
            self.session_dir / "session.json",
            {
                "session_dir": str(self.session_dir),
                "started_at": datetime.fromtimestamp(self._session_start).isoformat(),
                "started_epoch": self._session_start,
                "ended_at": datetime.fromtimestamp(end_ts).isoformat(),
                "ended_epoch": end_ts,
                "total_exchanges": self._counter,
                "duration_s": round(end_ts - self._session_start, 1),
            },
        )
        logger.info(
            "Session closed: %d exchanges in %.1f s",
            self._counter,
            end_ts - self._session_start,
        )
