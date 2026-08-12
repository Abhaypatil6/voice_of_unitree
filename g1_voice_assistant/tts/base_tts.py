"""
tts/base_tts.py
---------------
Abstract base class for all TTS engines.

Python 3.8 compatible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class TTSError(RuntimeError):
    """Raised when TTS synthesis fails."""


class BaseTTS(ABC):
    """Output boundary for text-to-speech synthesis."""

    @abstractmethod
    def synthesize(self, text: str, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Convert *text* to speech and save as a WAV file, or speak directly.

        Parameters
        ----------
        text:
            Plain text to synthesise.
        output_path:
            Destination path for the generated WAV file, if applicable.
            Direct native TTS backends may ignore this parameter.

        Returns
        -------
        Optional[Path]
            Path to the generated WAV file, or ``None`` when speech is handled
            directly by the robot audio service.

        Raises
        ------
        TTSError
            If synthesis fails for any reason.
        """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Short identifier for this TTS engine, e.g. ``piper`` or ``espeak``."""

    @property
    def sample_rate(self) -> int:
        """Sample rate of the generated WAV (Hz).  Subclasses may override."""
        return 22050

    @property
    def speaks_directly(self) -> bool:
        """Whether the TTS engine speaks directly through the robot speaker."""
        return False
