"""
llm/groq_stt.py
---------------
Groq Whisper speech-to-text client for standalone mode.

Sends a WAV audio buffer to the Groq Whisper transcription API and
returns the transcribed text.  Used ONLY in standalone mic mode
(main_standalone.py).  In ROS mode, STT is handled by the Unitree G1's
built-in system and this module is NOT used.

Model:
    Default: whisper-large-v3-turbo (fast, high-quality, English-focused)
    Configurable via STT_MODEL in .env

Design:
    - Language is forced to English (lang="en") to avoid hallucinations
      in other languages and to ensure the LLM always receives English.
    - Non-English speech will either be silently transcribed or produce
      English phonetics – the LLM system prompt then redirects the user.
    - API key is NEVER logged.
    - Empty/noise transcriptions are detected and returned as None.

Python 3.8 compatible.
"""

from __future__ import annotations

import io
import time
from typing import Optional

from logging_utils.logger import get_logger

logger = get_logger(__name__)

# Characters that suggest Whisper produced a noise/empty transcription
_NOISE_MARKERS = frozenset(["[music]", "[blank_audio]", "[silence]", "(music)", "(noise)"])


class STTError(RuntimeError):
    """Raised when the Groq Whisper API call fails."""


class GroqSTT:
    """
    Groq Whisper transcription client.

    Parameters
    ----------
    api_key:
        Groq API key (read from env; never hard-code).
    model:
        Groq Whisper model, e.g. 'whisper-large-v3-turbo'.
    language:
        BCP-47 language code to force.  'en' = English only.
        Forcing English prevents hallucinated non-English tokens.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "whisper-large-v3-turbo",
        language: str = "en",
    ) -> None:
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is empty. "
                "Set it in .env:\n    GROQ_API_KEY=gsk_<your-key>"
            )
        self._api_key = api_key
        self._model = model
        self._language = language
        self._client = None  # lazily initialised
        logger.info(
            "GroqSTT ready (model=%s, language=%s)",
            model,
            language,
        )

    def _get_client(self):  # type: ignore[return]
        if self._client is None:
            try:
                from groq import Groq  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "groq SDK not installed.  Run:\n    pip install groq"
                ) from exc
            self._client = Groq(api_key=self._api_key)
        return self._client

    def transcribe(self, wav_bytes: bytes) -> Optional[str]:
        """
        Transcribe a WAV audio buffer.

        Parameters
        ----------
        wav_bytes:
            Complete WAV file bytes (16-bit PCM, 16000 Hz, mono).

        Returns
        -------
        str or None
            Transcribed text (stripped), or None if empty/noise.

        Raises
        ------
        STTError
            If the API call fails.
        """
        if not wav_bytes:
            return None

        logger.info("[TRANSCRIBING] Sending %d bytes to Groq Whisper...", len(wav_bytes))
        t_start = time.monotonic()

        try:
            client = self._get_client()
            # Groq audio transcription expects a file-like tuple
            response = client.audio.transcriptions.create(
                model=self._model,
                file=("audio.wav", io.BytesIO(wav_bytes), "audio/wav"),
                language=self._language,
                response_format="text",
            )
            latency = time.monotonic() - t_start
            text = (response or "").strip() if isinstance(response, str) else ""
        except Exception as exc:
            latency = time.monotonic() - t_start
            logger.error("[ERROR] Groq Whisper error (%.2fs): %s", latency, exc)
            raise STTError(f"Groq Whisper API error: {exc}") from exc

        logger.info("[TRANSCRIBING] Done in %.2fs: %r", latency, text[:80])

        # Filter noise/empty transcriptions
        if not text:
            return None
        if text.lower() in _NOISE_MARKERS:
            logger.info("[TRANSCRIBING] Noise marker detected, discarding: %r", text)
            return None

        return text
