"""
Piper TTS engine.

Generates WAV audio using the Piper binary.

This engine is language-model agnostic:
the caller decides whether to use the English or Hindi Piper model.

Python 3.8 compatible.
"""

from __future__ import annotations

import os
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
    Offline Piper TTS engine.

    The selected model determines the language.

    Examples:
        en_US-ryan-high
        hi_IN-rohan-medium
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

        self._onnx_path = (
            self._models_dir
            / f"{model_name}.onnx"
        )

        self._config_path = (
            self._models_dir
            / f"{model_name}.onnx.json"
        )

        self._validate()

        logger.info(
            "PiperEngine ready "
            "(model=%s, binary=%s, length_scale=%.2f)",
            model_name,
            piper_binary,
            length_scale,
        )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def _validate(self) -> None:
        """Validate Piper binary and model files."""

        env = os.environ.copy()

        env["LD_LIBRARY_PATH"] = (
            "/home/unitree/piper/piper:"
            + env.get("LD_LIBRARY_PATH", "")
        )

        try:

            result = subprocess.run(
                [
                    self._piper_binary,
                    "--version",
                ],
                capture_output=True,
                timeout=5,
                env=env,
            )

            logger.debug(
                "Piper version: %s",
                result.stdout.decode(
                    "utf-8",
                    errors="replace",
                ).strip(),
            )

        except FileNotFoundError as exc:

            raise TTSError(
                "Piper binary not found: "
                f"'{self._piper_binary}'"
            ) from exc

        except subprocess.TimeoutExpired as exc:

            raise TTSError(
                "Piper binary timed out "
                "during version check."
            ) from exc

        if not self._onnx_path.exists():

            raise TTSError(
                "Piper model not found: "
                f"{self._onnx_path}"
            )

        if not self._config_path.exists():

            raise TTSError(
                "Piper model configuration "
                "not found: "
                f"{self._config_path}"
            )

    # ---------------------------------------------------------
    # BaseTTS
    # ---------------------------------------------------------

    @property
    def engine_name(self) -> str:
        return "piper"

    @property
    def sample_rate(self) -> int:
        return 22050

    # ---------------------------------------------------------
    # Synthesis
    # ---------------------------------------------------------

    def synthesize(
        self,
        text: str,
        output_path: Path,
    ) -> Path:
        """
        Convert text to speech using Piper.
        """

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        text = text.strip()

        if not text:

            raise TTSError(
                "Cannot synthesize empty text."
            )

        logger.info(
            "PiperEngine: synthesising "
            "%d chars -> %s",
            len(text),
            output_path.name,
        )

        start_time = time.monotonic()

        espeak_data = os.environ.get(
            "PIPER_ESPEAK_DATA",
            "/usr/lib/aarch64-linux-gnu/"
            "espeak-ng-data",
        )

        command = [
            self._piper_binary,
            "--model",
            str(self._onnx_path),
            "--output_file",
            str(output_path),
            "--length_scale",
            str(self._length_scale),
            "--espeak_data",
            espeak_data,
        ]

        env = os.environ.copy()

        env["LD_LIBRARY_PATH"] = (
            "/home/unitree/piper/piper:"
            + env.get("LD_LIBRARY_PATH", "")
        )

        try:

            process = subprocess.run(
                command,
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=60,
                env=env,
            )

        except subprocess.TimeoutExpired as exc:

            raise TTSError(
                "Piper synthesis timed out "
                "(>60 seconds)."
            ) from exc

        except Exception as exc:

            raise TTSError(
                f"Piper subprocess error: {exc}"
            ) from exc

        if process.returncode != 0:

            stderr = process.stderr.decode(
                "utf-8",
                errors="replace",
            ).strip()

            raise TTSError(
                "Piper exited with code "
                f"{process.returncode}.\n"
                f"stderr: {stderr}"
            )

        if (
            not output_path.exists()
            or output_path.stat().st_size == 0
        ):

            raise TTSError(
                "Piper produced no output at: "
                f"{output_path}"
            )

        elapsed = (
            time.monotonic() - start_time
        )

        logger.info(
            "PiperEngine: done in %.2f s -> %s "
            "(%.1f KB)",
            elapsed,
            output_path.name,
            output_path.stat().st_size / 1024,
        )

        return output_path

    # ---------------------------------------------------------
    # WAV utilities
    # ---------------------------------------------------------

    def get_wav_sample_rate(
        self,
        wav_path: Path,
    ) -> int:

        try:

            with wave.open(
                str(wav_path),
                "rb",
            ) as wav:

                return wav.getframerate()

        except Exception:

            return self.sample_rate

    def get_wav_duration(
        self,
        wav_path: Path,
    ) -> Optional[float]:

        try:

            with wave.open(
                str(wav_path),
                "rb",
            ) as wav:

                return (
                    wav.getnframes()
                    / wav.getframerate()
                )

        except Exception:

            return None