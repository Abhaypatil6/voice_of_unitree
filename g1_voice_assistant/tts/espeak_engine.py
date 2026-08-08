"""
tts/espeak_engine.py
--------------------
espeak-ng TTS fallback engine.

espeak-ng is available via:
    sudo apt install espeak-ng

It produces robotic-sounding speech but is:
* Always available on Ubuntu (no downloads needed)
* Very fast (< 100 ms)
* Works on ARM64

Use this for testing the audio pipeline when Piper is not yet installed.

Python 3.8 compatible.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from logging_utils.logger import get_logger
from tts.base_tts import BaseTTS, TTSError

logger = get_logger(__name__)


class ESpeakEngine(BaseTTS):
    """
    TTS engine using the espeak-ng command-line tool.

    Parameters
    ----------
    voice:
        espeak voice name, e.g. ``en``, ``en-us``.
    speed:
        Words per minute (default 150).
    """

    def __init__(
        self,
        voice: str = "en",
        speed: int = 150,
    ) -> None:
        self._voice = voice
        self._speed = speed
        self._binary = self._find_binary()
        logger.info(
            "ESpeakEngine ready (binary=%s, voice=%s, speed=%d)",
            self._binary, voice, speed,
        )

    @staticmethod
    def _find_binary() -> str:
        for name in ("espeak-ng", "espeak"):
            try:
                subprocess.run([name, "--version"], capture_output=True, timeout=3)
                return name
            except FileNotFoundError:
                continue
        raise TTSError(
            "Neither espeak-ng nor espeak is installed.\n"
            "Install with:  sudo apt install espeak-ng\n"
            "Or install Piper TTS and set TTS_BACKEND=piper."
        )

    @property
    def engine_name(self) -> str:
        return "espeak"

    @property
    def sample_rate(self) -> int:
        return 22050

    def synthesize(self, text: str, output_path: Path) -> Path:
        """
        Synthesise *text* using espeak-ng and save as WAV.

        espeak-ng natively writes WAV format with the -w flag.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not text.strip():
            raise TTSError("Cannot synthesise empty text.")

        logger.info("ESpeakEngine: synthesising %d chars → %s", len(text), output_path.name)
        t_start = time.monotonic()

        cmd = [
            self._binary,
            "-v", self._voice,
            "-s", str(self._speed),
            "-w", str(output_path),
            text,
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            raise TTSError("espeak-ng timed out.")
        except Exception as exc:
            raise TTSError(f"espeak-ng subprocess error: {exc}") from exc

        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise TTSError(
                f"espeak-ng exited with code {proc.returncode}: {stderr}"
            )

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise TTSError(f"espeak-ng produced no output at: {output_path}")

        elapsed = time.monotonic() - t_start
        logger.info("ESpeakEngine: done in %.2f s", elapsed)
        return output_path
