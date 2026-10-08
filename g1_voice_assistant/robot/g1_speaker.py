"""
robot/g1_speaker.py
-------------------
Stage 4 – G1 audio speaker output.

Plays a WAV file through the G1's speaker.

IMPLEMENTATION STRATEGY
~~~~~~~~~~~~~~~~~~~~~~~
The G1 runs on a Jetson Orin NX with an NVIDIA APE audio card
exposed as ALSA device ``hw:1,x`` (card 1, APE).

We use two backends, in order of preference:

1. **ALSA / aplay** (primary, no Python deps):
   - Uses ``aplay`` subprocess
   - Configure ALSA_OUTPUT_DEVICE in .env to target the correct device
   - Example: ``ALSA_OUTPUT_DEVICE=hw:1,0``  (APE card 1, device 0)
   - Leave blank to use the ALSA ``default`` device
   - Supports interruptible playback (barge-in) via Popen + kill

2. **sounddevice** (fallback for development machine):
   - Python package that wraps PortAudio
   - Works on the dev laptop for testing
   - Supports interruptible playback via sd.stop()

BARGE-IN SUPPORT
~~~~~~~~~~~~~~~~
Both backends support interruption via the ``stop()`` method on the
speaker object.  G1Speaker calls ``speaker.stop()`` when asked to interrupt.
The ``_is_playing`` attribute is True during playback, False otherwise.

DO NOT USE Unitree SDK AudioClient blindly:
   Before enabling the ``unitree`` backend, inspect the API on the G1:
       ros2 interface show unitree_go/msg/AudioData
       ros2 topic echo /api/audiohub/request

The ``unitree`` backend is left as a skeleton to be filled after
inspecting the real interface on the G1.

Python 3.8 compatible.
"""

from __future__ import annotations

import subprocess
import threading
import time
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from logging_utils.logger import get_logger
from logging_utils.artifact_logger import ArtifactLogger

logger = get_logger(__name__)


class SpeakerError(RuntimeError):
    """Raised when audio playback fails."""


class BaseSpeaker(ABC):
    """Abstract speaker output boundary with optional barge-in support."""

    @abstractmethod
    def play(self, audio_path: Path) -> None:
        """Play the WAV file at *audio_path* and wait for completion."""

    def stop(self) -> bool:
        """
        Interrupt current playback.

        Returns
        -------
        bool
            True if playback was successfully interrupted, False if not
            supported or nothing was playing.
        """
        return False  # Default: no barge-in support

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short identifier for this backend."""

    @property
    def is_playing(self) -> bool:
        """True if audio is currently playing."""
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Backend 1 – ALSA / aplay (primary for G1)
# ─────────────────────────────────────────────────────────────────────────────

class ALSASpeaker(BaseSpeaker):
    """
    Play a WAV file through an ALSA device using ``aplay``.

    This is the recommended backend for the G1's APE audio card.
    Supports barge-in via process kill.

    Parameters
    ----------
    device:
        ALSA device string, e.g. ``hw:1,0`` for APE card 1 device 0.
        ``None`` uses the ALSA ``default`` device.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self._device = device
        self._proc: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
        self._lock = threading.Lock()
        self._playing = False
        logger.info(
            "ALSASpeaker ready (device=%s, barge-in=supported)",
            device or "default",
        )

    @property
    def backend_name(self) -> str:
        return "alsa"

    @property
    def is_playing(self) -> bool:
        return self._playing

    def play(self, audio_path: Path) -> None:
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        cmd = ["aplay"]
        if self._device:
            cmd += ["-D", self._device]
        cmd.append(str(audio_path))

        logger.info(
            "ALSASpeaker: playing %s (device=%s)",
            audio_path.name,
            self._device or "default",
        )

        try:
            with self._lock:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                self._proc = proc
                self._playing = True
        except FileNotFoundError:
            self._playing = False
            raise SpeakerError(
                "'aplay' not found.  Install with:\n"
                "    sudo apt install alsa-utils"
            )

        try:
            _, stderr_data = proc.communicate(timeout=120)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            self._playing = False
            raise SpeakerError(f"aplay timed out playing {audio_path}")
        finally:
            with self._lock:
                self._proc = None
                self._playing = False

        # returncode -2 / -15 = killed by stop() – treat as clean
        if returncode not in (0, -2, -15):
            stderr = (stderr_data or b"").decode("utf-8", errors="replace").strip()
            raise SpeakerError(
                f"aplay failed (exit {returncode}) for {audio_path.name}.\n"
                f"stderr: {stderr}\n"
                f"Hint: check ALSA_OUTPUT_DEVICE in .env\n"
                f"      Run 'aplay -l' to list available devices."
            )

        logger.info("ALSASpeaker: playback complete")

    def stop(self) -> bool:
        """Kill the running aplay process (barge-in)."""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                logger.info("ALSASpeaker: interrupting playback (barge-in)")
                self._proc.terminate()
                return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Backend 2 – sounddevice (dev machine / testing)
# ─────────────────────────────────────────────────────────────────────────────

class LocalSpeaker(BaseSpeaker):
    """
    Play a WAV file using the sounddevice Python package.

    Useful for testing on the dev laptop.  May not work on the G1 if
    sounddevice cannot access the APE audio device.
    Supports barge-in via sd.stop().

    Parameters
    ----------
    output_device:
        sounddevice device index.  ``None`` = system default.
    """

    def __init__(self, output_device: Optional[int] = None) -> None:
        self._output_device = output_device
        self._playing = False
        self._lock = threading.Lock()
        logger.info("LocalSpeaker ready (device=%s, barge-in=supported)", output_device)

    @property
    def backend_name(self) -> str:
        return "local"

    @property
    def is_playing(self) -> bool:
        return self._playing

    def play(self, audio_path: Path) -> None:
        try:
            import sounddevice as sd  # type: ignore
            import scipy.io.wavfile as wavfile  # type: ignore
        except ImportError:
            # Fallback to aplay if sounddevice not installed
            logger.warning("sounddevice not installed, falling back to aplay")
            ALSASpeaker(device=None).play(audio_path)
            return

        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            sample_rate, data = wavfile.read(str(audio_path))
            logger.info("LocalSpeaker: playing %s (%d Hz)", audio_path.name, sample_rate)
            with self._lock:
                self._playing = True
            sd.play(data, samplerate=sample_rate, device=self._output_device)
            sd.wait()
            logger.info("LocalSpeaker: playback complete")
        except Exception as exc:
            raise SpeakerError(f"Local playback failed: {exc}") from exc
        finally:
            with self._lock:
                self._playing = False

    def stop(self) -> bool:
        """Stop sounddevice playback (barge-in)."""
        try:
            import sounddevice as sd  # type: ignore
            if self._playing:
                logger.info("LocalSpeaker: interrupting playback (barge-in)")
                sd.stop()
                return True
        except Exception:
            pass
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Backend 3 – Unitree SDK (skeleton – do NOT enable without inspecting the API)
# ─────────────────────────────────────────────────────────────────────────────

class UnitreeSpeaker(BaseSpeaker):
    """
    SKELETON – play audio via the Unitree G1 AudioClient SDK.

    DO NOT enable this without first inspecting the actual Unitree API on the G1:

        ros2 interface show unitree_go/msg/AudioData
        ros2 topic echo /api/audiohub/request

    This class will raise NotImplementedError until the real API
    is verified and implemented.
    """

    @property
    def backend_name(self) -> str:
        return "unitree"

    def play(self, audio_path: Path) -> None:
        raise NotImplementedError(
            "UnitreeSpeaker is not yet implemented.\n"
            "Before implementing:\n"
            "  1. On the G1, run: ros2 topic echo /api/audiohub/request\n"
            "  2. Inspect the message format\n"
            "  3. Check: ros2 interface show unitree_go/msg/AudioData\n"
            "  4. Look for existing Unitree audio examples in the SDK\n"
            "Set SPEAKER_BACKEND=alsa to use aplay instead."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Playback wrapper with artifact logging and barge-in support
# ─────────────────────────────────────────────────────────────────────────────

class G1Speaker:
    """
    High-level speaker with ArtifactLogger integration and barge-in support.

    Wraps a BaseSpeaker backend and saves playback metadata for every
    play() call.

    Parameters
    ----------
    speaker:
        The underlying BaseSpeaker backend.
    artifact_logger:
        ArtifactLogger for the current session.
    interruptible:
        If True, calling stop() will interrupt ongoing playback.
    """

    def __init__(
        self,
        speaker: BaseSpeaker,
        artifact_logger: ArtifactLogger,
        interruptible: bool = True,
    ) -> None:
        self._speaker = speaker
        self._artifact_logger = artifact_logger
        self._interruptible = interruptible
        logger.info(
            "G1Speaker ready (backend=%s, interruptible=%s)",
            speaker.backend_name,
            interruptible,
        )

    def play(self, counter: int, audio_path: Path) -> bool:
        """
        Play *audio_path* and save playback artifact.

        Parameters
        ----------
        counter:
            Pipeline turn counter (for artifact filenames).
        audio_path:
            WAV file to play.

        Returns
        -------
        bool
            True on success, False on error (never raises).
        """
        audio_path = Path(audio_path)
        start_ts = time.time()
        error_msg: Optional[str] = None
        success = False

        try:
            self._speaker.play(audio_path)
            success = True
        except (SpeakerError, FileNotFoundError) as exc:
            error_msg = str(exc)
            logger.error("[ERROR] [Speaker #%06d] Playback failed: %s", counter, exc)
        except Exception as exc:
            error_msg = f"Unexpected error: {exc}"
            logger.exception("[ERROR] [Speaker #%06d] Unexpected playback error", counter)

        end_ts = time.time()

        self._artifact_logger.log_speaker(
            counter=counter,
            audio_file=audio_path,
            playback_start=start_ts,
            playback_end=end_ts,
            success=success,
            error=error_msg,
        )

        return success

    def stop(self) -> bool:
        """
        Interrupt ongoing playback (barge-in).

        Returns
        -------
        bool
            True if playback was interrupted, False otherwise.
        """
        if not self._interruptible:
            return False
        return self._speaker.stop()

    @property
    def is_playing(self) -> bool:
        """True if audio is currently playing."""
        return self._speaker.is_playing


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_speaker(
    backend: str,
    alsa_device: Optional[str] = None,
    artifact_logger: Optional[ArtifactLogger] = None,
) -> BaseSpeaker:
    """
    Create and return the speaker backend.

    Parameters
    ----------
    backend:
        One of: ``alsa``, ``local``, ``unitree``.
    alsa_device:
        ALSA device string (used when backend=alsa).
    artifact_logger:
        If provided, wraps the speaker in G1Speaker for artifact logging.
    """
    backend = backend.strip().lower()

    if backend == "alsa":
        speaker: BaseSpeaker = ALSASpeaker(device=alsa_device)
    elif backend == "local":
        speaker = LocalSpeaker()
    elif backend == "unitree":
        speaker = UnitreeSpeaker()
    else:
        raise ValueError(
            f"Unknown SPEAKER_BACKEND={backend!r}. "
            "Expected one of: alsa, local, unitree."
        )

    return speaker
