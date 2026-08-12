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

2. **sounddevice** (fallback for development machine):
   - Python package that wraps PortAudio
   - Works on the dev laptop for testing
   - May not have access to the G1's APE device

DO NOT USE Unitree SDK AudioClient blindly:
   Before enabling the ``unitree`` backend, inspect the API on the G1:
       ros2 interface show unitree_go/msg/AudioData
       ros2 topic echo /api/audiohub/request

The ``unitree`` backend is left as a skeleton to be filled after
inspecting the real interface on the G1.

Python 3.8 compatible.
"""

from __future__ import annotations

import importlib.util
import os
import site
import subprocess
import sys
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
    """Abstract speaker output boundary."""

    @abstractmethod
    def play(self, audio_path: Path) -> None:
        """Play the WAV file at *audio_path* and wait for completion."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short identifier for this backend."""


# ─────────────────────────────────────────────────────────────────────────────
# Backend 1 – ALSA / aplay (primary for G1)
# ─────────────────────────────────────────────────────────────────────────────

class ALSASpeaker(BaseSpeaker):
    """
    Play a WAV file through an ALSA device using ``aplay``.

    This is the recommended backend for the G1's APE audio card.

    Parameters
    ----------
    device:
        ALSA device string, e.g. ``hw:1,0`` for APE card 1 device 0.
        ``None`` uses the ALSA ``default`` device.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self._device = device
        logger.info(
            "ALSASpeaker ready (device=%s)",
            device or "default",
        )

    @property
    def backend_name(self) -> str:
        return "alsa"

    def play(self, audio_path: Path) -> None:
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        cmd = ["aplay"]
        if self._device:
            cmd += ["-D", self._device]
        cmd.append(str(audio_path))

        logger.info("ALSASpeaker: playing %s (device=%s)", audio_path.name, self._device or "default")

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=120,
            )
        except FileNotFoundError:
            raise SpeakerError(
                "'aplay' not found.  Install with:\n"
                "    sudo apt install alsa-utils"
            )
        except subprocess.TimeoutExpired:
            raise SpeakerError(f"aplay timed out playing {audio_path}")

        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise SpeakerError(
                f"aplay failed (exit {proc.returncode}) for {audio_path.name}.\n"
                f"stderr: {stderr}\n"
                f"Hint: check ALSA_OUTPUT_DEVICE in .env\n"
                f"      Run 'aplay -l' to list available devices."
            )

        logger.info("ALSASpeaker: playback complete")


# ─────────────────────────────────────────────────────────────────────────────
# Backend 2 – sounddevice (dev machine / testing)
# ─────────────────────────────────────────────────────────────────────────────

class LocalSpeaker(BaseSpeaker):
    """
    Play a WAV file using the sounddevice Python package.

    Useful for testing on the dev laptop.  May not work on the G1 if
    sounddevice cannot access the APE audio device.

    Parameters
    ----------
    output_device:
        sounddevice device index.  ``None`` = system default.
    """

    def __init__(self, output_device: Optional[int] = None) -> None:
        self._output_device = output_device
        logger.info("LocalSpeaker ready (device=%s)", output_device)

    @property
    def backend_name(self) -> str:
        return "local"

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
            sd.play(data, samplerate=sample_rate, device=self._output_device)
            sd.wait()
            logger.info("LocalSpeaker: playback complete")
        except Exception as exc:
            raise SpeakerError(f"Local playback failed: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Backend 3 – Unitree SDK (skeleton – do NOT enable without inspecting the API)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_unitree_sdk_path(explicit_path: Optional[str] = None) -> Optional[str]:
    if explicit_path:
        explicit_path = os.path.expanduser(explicit_path)
        if os.path.exists(explicit_path):
            return explicit_path

    env_path = os.environ.get("UNITREE_SDK_PATH", "").strip()
    if env_path and os.path.exists(env_path):
        return env_path

    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "unitree_sdk2_python"),
        os.path.expanduser("~/Projects/unitree_sdk2_python"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    return None


class UnitreeSpeaker(BaseSpeaker):
    """
    Play audio via the Unitree G1 AudioClient SDK.
    Runs the SDK in an isolated spawned process to avoid CycloneDDS domain
    collisions with the main ROS 2 process.
    """
    def __init__(self, network_interface: str = "eth0") -> None:
        self._network_interface = network_interface

        # We must use 'spawn' to ensure the child process does not inherit
        # the main process's ROS 2 DDS file descriptors or memory space.
        import multiprocessing as mp
        import queue
        ctx = mp.get_context("spawn")

        self._queue = ctx.Queue()
        init_status_queue = ctx.Queue()

        self._process = ctx.Process(
            target=self._worker,
            args=(self._queue, init_status_queue, self._network_interface),
        )
        self._process.daemon = True
        self._process.start()

        # Wait for initialization status
        try:
            status = init_status_queue.get(timeout=15.0)
            if status != "OK":
                raise SpeakerError(f"Unitree SDK worker failed to initialize: {status}")
        except queue.Empty:
            if self._process.is_alive():
                self._process.terminate()
            raise SpeakerError("Unitree SDK worker initialization timed out.")

        logger.info("UnitreeSpeaker process spawned and ready on %s", self._network_interface)

    @staticmethod
    def _worker(queue, init_status_queue, network_interface: str):
        try:
            sdk_path = _resolve_unitree_sdk_path(os.environ.get("UNITREE_SDK_PATH"))
            if sdk_path:
                sys.path.insert(0, sdk_path)
                try:
                    site.addsitedir(sdk_path)
                except Exception:
                    pass

            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
            
            # Isolated initialization of Unitree's CycloneDDS participant
            try:
                ChannelFactoryInitialize(0, network_interface)
            except Exception as exc:
                init_status_queue.put(f"ChannelFactoryInitialize(0) error: {exc}")
                return
            
            client = AudioClient()
            client.SetTimeout(10.0)
            client.Init()
            
            # Signal success to parent process
            init_status_queue.put("OK")
            
        except ImportError as exc:
            init_status_queue.put(f"unitree_sdk2py not found. Please install bindings. {exc}")
            return
        except Exception as exc:
            init_status_queue.put(f"ChannelFactory init failed: {exc}")
            return

        while True:
            try:
                audio_path_str = queue.get()
                if audio_path_str is None:
                    break
                    
                with wave.open(audio_path_str, "rb") as wf:
                    if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                        print("SpeakerError: requires 16-bit mono WAV.")
                        continue
                    
                    frames = wf.readframes(wf.getnframes())
                    sample_rate = wf.getframerate()
                    duration = wf.getnframes() / float(sample_rate)
                    
                stream_id = f"g1_voice_{int(time.time())}"
                client.PlayStream("voice_assistant", stream_id, frames)
                
                # Sleep to wait for playback completion
                time.sleep(duration + 0.1)
                
            except Exception as exc:
                print(f"Unitree speaker worker error: {exc}")

    @property
    def backend_name(self) -> str:
        return "unitree"

    def play(self, audio_path: Path) -> None:
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        # Ensure process is alive before queuing
        if not self._process.is_alive():
            raise SpeakerError("UnitreeSpeaker background process died unexpectedly.")
            
        logger.info("UnitreeSpeaker: sending %s to background process", audio_path.name)
        self._queue.put(str(audio_path))
        
        # We need to block until it finishes. A simple way is to wait for the expected duration.
        # This isn't perfect synchronization, but matches the original behavior.
        try:
            with wave.open(str(audio_path), "rb") as wf:
                duration = wf.getnframes() / float(wf.getframerate())
            time.sleep(duration + 0.2)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Playback wrapper with artifact logging
# ─────────────────────────────────────────────────────────────────────────────

class G1Speaker:
    """
    High-level speaker with ArtifactLogger integration.

    Wraps a BaseSpeaker backend and saves playback metadata for every
    play() call.

    Parameters
    ----------
    speaker:
        The underlying BaseSpeaker backend.
    artifact_logger:
        ArtifactLogger for the current session.
    """

    def __init__(self, speaker: BaseSpeaker, artifact_logger: ArtifactLogger) -> None:
        self._speaker = speaker
        self._artifact_logger = artifact_logger
        logger.info("G1Speaker ready (backend=%s)", speaker.backend_name)

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
            logger.error("[Speaker #%06d] Playback failed: %s", counter, exc)
        except Exception as exc:
            error_msg = f"Unexpected error: {exc}"
            logger.exception("[Speaker #%06d] Unexpected playback error", counter)

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


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_speaker(
    backend: str,
    alsa_device: Optional[str] = None,
    network_interface: Optional[str] = None,
    artifact_logger: Optional[ArtifactLogger] = None,
) -> Optional[BaseSpeaker]:
    """
    Create and return the speaker backend.

    Parameters
    ----------
    backend:
        One of: ``alsa``, ``local``, ``unitree``.
    alsa_device:
        ALSA device string (used when backend=alsa).
    network_interface:
        Unitree network interface (used when backend=unitree).
    artifact_logger:
        If provided, wraps the speaker in G1Speaker for artifact logging.
    """
    backend = backend.strip().lower()

    if backend == "alsa":
        speaker: BaseSpeaker = ALSASpeaker(device=alsa_device)
    elif backend == "local":
        speaker = LocalSpeaker()
    elif backend == "unitree":
        try:
            speaker = UnitreeSpeaker(network_interface=network_interface or "eth0")
        except (SpeakerError, Exception) as exc:
            logger.warning(
                "Unitree speaker backend unavailable (%s); speaker output disabled.",
                exc,
            )
            return None
    else:
        raise ValueError(
            f"Unknown SPEAKER_BACKEND={backend!r}. "
            "Expected one of: alsa, local, unitree."
        )

    return speaker
