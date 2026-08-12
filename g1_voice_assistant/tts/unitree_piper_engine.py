"""
Piper TTS -> Unitree G1 speaker.

Piper generates the WAV.
A separate Unitree audio worker sends the audio
through AudioClient.PlayStream() to the G1 speaker.

No ALSA is used.
No TtsMaker() is used.

Python 3.8 compatible.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from logging_utils.logger import get_logger
from tts.base_tts import BaseTTS, TTSError
from tts.piper_engine import PiperEngine

logger = get_logger(__name__)


class UnitreePiperTTSEngine(BaseTTS):

    def __init__(
        self,
        model_name: str = "en_US-ryan-high",
        models_dir: Path = Path("tts/models"),
        piper_binary: str = "piper",
        length_scale: float = 1.0,
        sdk_path: Optional[Path] = None,
        network_interface: str = "eth0",
    ) -> None:

        self._piper = PiperEngine(
            model_name=model_name,
            models_dir=models_dir,
            piper_binary=piper_binary,
            length_scale=length_scale,
        )

        self._sdk_path = Path(
            sdk_path
            or Path.home()
            / "Projects"
            / "unitree_sdk2_python"
        ).expanduser()

        self._network_interface = (
            network_interface or "eth0"
        )

        self._worker = None

        self._start_worker()

    # ---------------------------------------------------------
    # BaseTTS properties
    # ---------------------------------------------------------

    @property
    def engine_name(self) -> str:
        return "unitree_piper"

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def speaks_directly(self) -> bool:
        return True

    # ---------------------------------------------------------
    # Worker
    # ---------------------------------------------------------

    def _start_worker(self) -> None:

        worker_path = (
            Path(__file__).resolve().parent
            / "unitree_audio_worker.py"
        )

        if not worker_path.exists():
            raise TTSError(
                "Unitree audio worker not found: "
                f"{worker_path}"
            )

        command = [
            sys.executable,
            str(worker_path),
            str(self._sdk_path),
            self._network_interface,
        ]

        env = os.environ.copy()

        python_paths = [
            str(self._sdk_path),
            os.path.expanduser(
                "~/.local/lib/python3.8/site-packages"
            ),
        ]

        existing = env.get("PYTHONPATH", "")

        if existing:
            python_paths.append(existing)

        env["PYTHONPATH"] = ":".join(
            python_paths
        )

        logger.info(
            "Starting Unitree audio worker..."
        )

        try:

            self._worker = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )

        except Exception as exc:

            raise TTSError(
                "Failed to start Unitree audio worker: "
                f"{exc}"
            ) from exc

        self._wait_for_worker()

    def _wait_for_worker(self) -> None:

        if self._worker is None:
            raise TTSError(
                "Unitree audio worker is not running."
            )

        if self._worker.stdout is None:
            raise TTSError(
                "Unitree audio worker stdout unavailable."
            )

        while True:

            line = self._worker.stdout.readline()

            if not line:

                exit_code = (
                    self._worker.poll()
                )

                raise TTSError(
                    "Unitree audio worker stopped "
                    "during startup. "
                    f"Exit code: {exit_code}"
                )

            line = line.strip()

            if line:

                logger.info(
                    "Unitree audio worker: %s",
                    line,
                )

            if (
                "AudioClient initialized"
                in line
            ):

                logger.info(
                    "Unitree audio playback: READY"
                )

                return

            if line.startswith("ERROR:"):

                raise TTSError(line)

    # ---------------------------------------------------------
    # Synthesis + playback
    # ---------------------------------------------------------

    def synthesize(
        self,
        text: str,
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:

        text = text.strip()

        if not text:
            raise TTSError(
                "Cannot synthesize empty text."
            )

        if self._worker is None:
            raise TTSError(
                "Unitree audio worker is not running."
            )

        if self._worker.poll() is not None:
            raise TTSError(
                "Unitree audio worker has stopped."
            )

        if output_path is None:

            output_path = (
                Path("conversations")
                / "tts"
                / "response.wav"
            )

        output_path = Path(
            output_path
        )

        # -----------------------------------------------------
        # 1. Piper generates WAV
        # -----------------------------------------------------

        wav_path = self._piper.synthesize(
            text,
            output_path,
        )

        logger.info(
            "Piper generated audio: %s",
            wav_path,
        )

        # -----------------------------------------------------
        # 2. Send WAV path to worker
        # -----------------------------------------------------

        if self._worker.stdin is None:
            raise TTSError(
                "Unitree audio worker stdin unavailable."
            )

        command = (
            "PLAY "
            + str(wav_path.resolve())
        )

        logger.info(
            "Sending audio to G1 speaker..."
        )

        try:

            self._worker.stdin.write(
                command + "\n"
            )

            self._worker.stdin.flush()

        except Exception as exc:

            raise TTSError(
                "Failed to send audio to Unitree "
                f"worker: {exc}"
            ) from exc

        # -----------------------------------------------------
        # 3. Wait for playback completion
        # -----------------------------------------------------

        if self._worker.stdout is None:
            raise TTSError(
                "Unitree audio worker stdout unavailable."
            )

        while True:

            line = (
                self._worker.stdout.readline()
            )

            if not line:

                exit_code = (
                    self._worker.poll()
                )

                raise TTSError(
                    "Unitree audio worker stopped "
                    "during playback. "
                    f"Exit code: {exit_code}"
                )

            line = line.strip()

            if line:

                logger.info(
                    "Unitree audio worker: %s",
                    line,
                )

            if line == "PLAY_COMPLETE":

                logger.info(
                    "G1 playback complete."
                )

                return wav_path

            if line.startswith(
                "PLAY_ERROR"
            ):

                raise TTSError(line)

    # ---------------------------------------------------------
    # Shutdown
    # ---------------------------------------------------------

    def close(self) -> None:

        if self._worker is None:
            return

        if self._worker.poll() is None:

            try:

                if self._worker.stdin is not None:

                    self._worker.stdin.write(
                        "__STOP__\n"
                    )

                    self._worker.stdin.flush()

                    self._worker.stdin.write(
                        "__QUIT__\n"
                    )

                    self._worker.stdin.flush()

            except Exception:
                pass

            try:

                self._worker.wait(
                    timeout=3
                )

            except subprocess.TimeoutExpired:

                self._worker.terminate()

                try:

                    self._worker.wait(
                        timeout=2
                    )

                except subprocess.TimeoutExpired:

                    self._worker.kill()

        self._worker = None

        logger.info(
            "Unitree audio worker stopped."
        )

    def __del__(self) -> None:

        try:
            self.close()

        except Exception:
            pass