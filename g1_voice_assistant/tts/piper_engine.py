"""
tts/piper_engine.py
-------------------
Piper TTS engine (offline, ARM64-compatible).

Piper is a fast, local neural TTS system by Rhasspy.
It runs entirely on CPU and produces high-quality speech.

Supported installation paths on the G1:
    1. Piper binary in PATH (preferred):
       - Download from https://github.com/rhasspy/piper/releases
       - Select the linux_aarch64 release for the Jetson Orin NX
       - E.g. piper_linux_aarch64.tar.gz
       - Extract and place the 'piper' binary in ~/piper/ or /usr/local/bin/
    2. pip install piper-tts (if available for Python 3.8 on ARM64)

Voice model setup:
    - Download .onnx + .onnx.json files from:
      https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US
    - Place both files in: tts/models/
    - Example: tts/models/en_US-ryan-high.onnx
               tts/models/en_US-ryan-high.onnx.json

This engine uses the piper BINARY via subprocess (most portable approach).
It does NOT depend on the piper-tts pip package.

Python 3.8 compatible.
"""

from __future__ import annotations

import subprocess
import time
import wave
from pathlib import Path
from typing import Optional

from logging_utils.logger import get_logger
from tts.base_tts import BaseTTS, TTSError

logger = get_logger(__name__)


class PiperEngine(BaseTTS):
    """
    Offline TTS using the Piper binary via subprocess.

    Parameters
    ----------
    model_name:
        Voice model stem, e.g. ``en_US-ryan-high``
        (must match the .onnx file in *models_dir*).
    models_dir:
        Directory containing the .onnx and .onnx.json files.
    piper_binary:
        Name or full path of the piper executable.
    length_scale:
        Speech rate (< 1 = faster, > 1 = slower, 1.0 = normal).
    """

    def __init__(
        self,
        model_name: str = "en_US-ryan-high",
        models_dir: Path = Path("tts/models"),
        piper_binary: str = "piper",
        length_scale: float = 1.0,
    ) -> None:
        self._model_name = model_name
        self._models_dir = Path(models_dir)
        self._piper_binary = piper_binary
        self._length_scale = length_scale

        # Resolve and validate paths
        self._onnx_path = self._models_dir / f"{model_name}.onnx"
        self._config_path = self._models_dir / f"{model_name}.onnx.json"

        self._validate()
        logger.info(
            "PiperEngine ready (model=%s, binary=%s, length_scale=%.2f)",
            model_name, piper_binary, length_scale,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        """Check that the binary and model files exist."""
        # Check binary is reachable
        try:
            result = subprocess.run(
                [self._piper_binary, "--version"],
                capture_output=True,
                timeout=5,
            )
            logger.debug("Piper version: %s", result.stdout.decode().strip())
        except FileNotFoundError:
            raise TTSError(
                f"Piper binary not found: '{self._piper_binary}'\n"
                "Download from: https://github.com/rhasspy/piper/releases\n"
                "For Jetson Orin NX use the linux_aarch64 release.\n"
                "Place the binary in PATH or set PIPER_BINARY=/full/path/piper"
            )
        except subprocess.TimeoutExpired:
            raise TTSError("Piper binary timed out during version check.")

        # Check model files
        if not self._onnx_path.exists():
            raise TTSError(
                f"Piper model not found: {self._onnx_path}\n"
                f"Download from: https://huggingface.co/rhasspy/piper-voices\n"
                f"Place both .onnx and .onnx.json in: {self._models_dir}\n"
                f"Expected: {self._onnx_path}"
            )
        if not self._config_path.exists():
            raise TTSError(
                f"Piper config not found: {self._config_path}\n"
                "The .onnx.json must accompany the .onnx file."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # BaseTTS interface
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def engine_name(self) -> str:
        return "piper"

    @property
    def sample_rate(self) -> int:
        # Piper's sample rate depends on the voice model.
        # Most en_US voices produce 22050 Hz.
        # We detect the actual rate from the generated WAV after synthesis.
        return 22050

    def synthesize(self, text: str, output_path: Path) -> Path:
        """
        Synthesise *text* using the Piper binary and write to *output_path*.

        Piper is invoked as:
            echo "<text>" | piper --model <onnx> --output_file <wav>
                            --length_scale <scale>

        Parameters
        ----------
        text:
            Plain text to synthesise (no SSML).
        output_path:
            WAV file destination.

        Returns
        -------
        Path
            Path to the generated WAV.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not text.strip():
            raise TTSError("Cannot synthesise empty text.")

        logger.info("PiperEngine: synthesising %d chars → %s", len(text), output_path.name)
        t_start = time.monotonic()

        cmd = [
            self._piper_binary,
            "--model",       str(self._onnx_path),
            "--output_file", str(output_path),
            "--length_scale", str(self._length_scale),
        ]

        try:
            proc = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            raise TTSError("Piper synthesis timed out (>60 s).")
        except Exception as exc:
            raise TTSError(f"Piper subprocess error: {exc}") from exc

        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise TTSError(
                f"Piper exited with code {proc.returncode}.\n"
                f"stderr: {stderr}"
            )

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise TTSError(f"Piper produced no output at: {output_path}")

        elapsed = time.monotonic() - t_start
        logger.info(
            "PiperEngine: done in %.2f s → %s (%.1f KB)",
            elapsed, output_path.name, output_path.stat().st_size / 1024,
        )
        return output_path

    def get_wav_sample_rate(self, wav_path: Path) -> int:
        """Read the actual sample rate from a generated WAV file."""
        try:
            with wave.open(str(wav_path), "rb") as wf:
                return wf.getframerate()
        except Exception:
            return self.sample_rate

    def get_wav_duration(self, wav_path: Path) -> Optional[float]:
        """Return the duration of a WAV file in seconds."""
        try:
            with wave.open(str(wav_path), "rb") as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            return None
