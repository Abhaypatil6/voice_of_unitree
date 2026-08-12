"""
Unitree G1 native TTS engine.

The main ROS 2 process does NOT initialize Unitree DDS directly.

Instead, this class starts unitree_tts_worker.py as a separate process.
The worker owns the Unitree DDS connection and communicates through stdin/stdout.

Architecture:

    ROS 2 main process
          |
          | text
          v
    UnitreeNativeTTSEngine
          |
          v
    unitree_tts_worker.py
          |
          v
    Unitree DDS
          |
          v
    G1 speaker
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from logging_utils.logger import get_logger
from tts.base_tts import BaseTTS, TTSError


logger = get_logger(__name__)


def _resolve_sdk_path(
    explicit_path: Optional[Path] = None,
) -> Path:
    """Find the Unitree SDK directory."""

    if explicit_path is not None:
        path = Path(explicit_path).expanduser()

        if path.exists():
            return path

    env_path = os.environ.get("UNITREE_SDK_PATH", "").strip()

    if env_path:
        path = Path(env_path).expanduser()

        if path.exists():
            return path

    candidate = (
        Path.home()
        / "Projects"
        / "unitree_sdk2_python"
    )

    if candidate.exists():
        return candidate

    raise TTSError(
        "Unitree SDK not found. Expected: {}".format(
            candidate
        )
    )


class UnitreeNativeTTSEngine(BaseTTS):
    """
    Unitree native TTS using a separate worker process.

    The worker owns the Unitree DDS connection so it does not
    conflict with ROS 2/rclpy in the main process.
    """

    def __init__(
        self,
        sdk_path: Optional[Path] = None,
        speaker_id: int = 0,
        timeout: float = 10.0,
        network_interface: Optional[str] = None,
    ) -> None:

        self._sdk_path = _resolve_sdk_path(sdk_path)
        self._speaker_id = speaker_id
        self._timeout = timeout
        self._network_interface = (
            network_interface or "eth0"
        )

        self._worker: Optional[subprocess.Popen] = None

        self._start_worker()

    @property
    def engine_name(self) -> str:
        return "unitree_native"

    @property
    def sample_rate(self) -> int:
        return 0

    @property
    def speaks_directly(self) -> bool:
        return True

    def _start_worker(self) -> None:
        """Start the standalone Unitree TTS worker."""

        worker_path = (
            Path(__file__).resolve().parent
            / "unitree_tts_worker.py"
        )

        if not worker_path.exists():
            raise TTSError(
                "Unitree TTS worker not found: {}".format(
                    worker_path
                )
            )

        command = [
            sys.executable,
            str(worker_path),
            str(self._sdk_path),
            self._network_interface,
            str(self._speaker_id),
        ]

        logger.info(
            "Starting Unitree TTS worker: %s",
            " ".join(command),
        )

        env = os.environ.copy()

        # Make Unitree SDK available to the worker.
        python_paths = [
            str(self._sdk_path),
            os.path.expanduser(
                "~/.local/lib/python3.8/site-packages"
            ),
        ]

        existing_pythonpath = env.get(
            "PYTHONPATH",
            "",
        )

        if existing_pythonpath:
            python_paths.append(existing_pythonpath)

        env["PYTHONPATH"] = ":".join(
            python_paths
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
                "Failed to start Unitree TTS worker: {}".format(
                    exc
                )
            ) from exc

        logger.info(
            "Unitree TTS worker started, PID=%s",
            self._worker.pid,
        )

        # Read worker startup messages.
        self._wait_for_worker_ready()

    def _wait_for_worker_ready(self) -> None:
        """Wait for the worker to initialize Unitree DDS."""

        if self._worker is None:
            raise TTSError(
                "Unitree TTS worker is not running."
            )

        if self._worker.stdout is None:
            raise TTSError(
                "Unitree TTS worker stdout unavailable."
            )

        startup_lines = []

        while True:
            line = self._worker.stdout.readline()

            if not line:
                code = self._worker.poll()

                raise TTSError(
                    "Unitree TTS worker stopped during startup "
                    "(exit code {}). Output: {}".format(
                        code,
                        " ".join(startup_lines),
                    )
                )

            line = line.rstrip()

            if line:
                logger.info(
                    "Unitree TTS worker: %s",
                    line,
                )

                startup_lines.append(line)

            if "AudioClient initialized" in line:
                logger.info(
                    "Unitree native speaker: READY"
                )
                return

            if line.startswith("ERROR:"):
                raise TTSError(line)

    def synthesize(
        self,
        text: str,
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Send text to the Unitree TTS worker.

        No ALSA.
        No Piper.
        No local WAV.
        """

        text = text.strip()

        if not text:
            raise TTSError(
                "Cannot synthesize empty text."
            )

        if self._worker is None:
            raise TTSError(
                "Unitree TTS worker is not running."
            )

        if self._worker.poll() is not None:
            raise TTSError(
                "Unitree TTS worker has stopped."
            )

        if self._worker.stdin is None:
            raise TTSError(
                "Unitree TTS worker stdin unavailable."
            )

        safe_text = (
            text
            if len(text) <= 120
            else text[:117] + "..."
        )

        logger.info(
            "G1 Native TTS: sending response: %s",
            safe_text,
        )

        try:
            self._worker.stdin.write(
                text.replace("\n", " ") + "\n"
            )

            self._worker.stdin.flush()

        except Exception as exc:
            raise TTSError(
                "Failed to send text to Unitree TTS worker: {}".format(
                    exc
                )
            ) from exc

        # Wait for worker acknowledgement.
        if self._worker.stdout is not None:

            while True:
                line = self._worker.stdout.readline()

                if not line:
                    raise TTSError(
                        "Unitree TTS worker stopped while speaking."
                    )

                line = line.rstrip()

                if line:
                    logger.info(
                        "Unitree TTS worker: %s",
                        line,
                    )

                if line.startswith("TTS_RESULT"):
                    try:
                        result = int(
                            line.split(maxsplit=1)[1]
                        )
                    except Exception:
                        result = -1

                    if result != 0:
                        raise TTSError(
                            "Unitree TtsMaker returned {}".format(
                                result
                            )
                        )

                    logger.info(
                        "G1 Native TTS: API return code = %s",
                        result,
                    )

                    return None

                if line.startswith("TTS_ERROR"):
                    raise TTSError(line)

    def close(self) -> None:
        """Stop the Unitree TTS worker."""

        if self._worker is None:
            return

        if self._worker.poll() is None:

            try:
                if self._worker.stdin is not None:
                    self._worker.stdin.write(
                        "__QUIT__\n"
                    )
                    self._worker.stdin.flush()

            except Exception:
                pass

            try:
                self._worker.wait(timeout=3)

            except subprocess.TimeoutExpired:
                self._worker.terminate()

                try:
                    self._worker.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._worker.kill()

        self._worker = None

        logger.info(
            "Unitree TTS worker stopped"
        )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# Backward-compatible alias.
UnitreeNativeSpeaker = UnitreeNativeTTSEngine