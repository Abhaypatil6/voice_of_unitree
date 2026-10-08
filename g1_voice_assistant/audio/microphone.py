"""
audio/microphone.py
-------------------
Microphone capture with voice activity detection (VAD) for standalone mode.

This module provides a simple, robust microphone recorder that:
  1. Listens continuously for speech using RMS-based VAD.
  2. Records audio once speech is detected.
  3. Stops recording after a configurable silence timeout.
  4. Filters out very short (noise) bursts via MIN_SPEECH_DURATION.
  5. Yields raw PCM bytes for each speech segment.

This is used in standalone mode (main_standalone.py) when the Unitree G1
ROS 2 system is not available.  In ROS mode, VAD is handled by the
Unitree hardware and this module is NOT used.

Dependencies:
    pip install pyaudio

Python 3.8 compatible.
"""

from __future__ import annotations

import io
import math
import struct
import wave
from typing import Generator, Optional

from logging_utils.logger import get_logger

logger = get_logger(__name__)


def _calculate_rms(chunk: bytes, sample_width: int = 2) -> float:
    """
    Compute normalised RMS amplitude of a PCM audio chunk.

    Parameters
    ----------
    chunk:
        Raw PCM bytes (16-bit signed little-endian).
    sample_width:
        Bytes per sample (2 = 16-bit).

    Returns
    -------
    float
        RMS in range [0.0, 1.0].
    """
    if not chunk:
        return 0.0
    count = len(chunk) // sample_width
    if count == 0:
        return 0.0
    fmt = f"<{count}h"
    try:
        samples = struct.unpack(fmt, chunk[:count * sample_width])
    except struct.error:
        return 0.0
    rms = math.sqrt(sum(s * s for s in samples) / count)
    return rms / 32768.0  # normalise to [0, 1]


class MicrophoneRecorder:
    """
    Captures microphone audio with RMS-based VAD.

    Parameters
    ----------
    sample_rate:
        Recording sample rate (must be 16000 for Groq Whisper).
    channels:
        Number of audio channels (1 = mono).
    chunk_size:
        Frames per PyAudio read buffer.
    vad_threshold:
        Normalised RMS threshold for speech onset (0.0–1.0).
    silence_timeout:
        Seconds of silence after which recording stops.
    min_speech_duration:
        Minimum seconds of speech before yielding a segment.
    device_index:
        PyAudio device index.  None = system default.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        vad_threshold: float = 0.01,
        silence_timeout: float = 1.5,
        min_speech_duration: float = 0.3,
        device_index: Optional[int] = None,
    ) -> None:
        self._rate = sample_rate
        self._channels = channels
        self._chunk = chunk_size
        self._threshold = vad_threshold
        self._silence_timeout = silence_timeout
        self._min_speech_duration = min_speech_duration
        self._device_index = device_index
        self._audio: Optional[object] = None  # PyAudio instance
        self._stream: Optional[object] = None

    def _import_pyaudio(self):  # type: ignore[return]
        try:
            import pyaudio  # type: ignore
            return pyaudio
        except ImportError:
            raise ImportError(
                "pyaudio is required for microphone input in standalone mode.\n"
                "Install with:\n"
                "    pip install pyaudio\n"
                "On Ubuntu, you may also need:\n"
                "    sudo apt install portaudio19-dev python3-pyaudio"
            )

    def open(self) -> None:
        """Open the microphone stream."""
        pyaudio = self._import_pyaudio()
        self._audio = pyaudio.PyAudio()
        assert self._audio is not None

        # Print available devices to help with configuration
        pa = self._audio
        device_count = pa.get_device_count()  # type: ignore[union-attr]
        logger.info("Available audio input devices:")
        for i in range(device_count):
            info = pa.get_device_info_by_index(i)  # type: ignore[union-attr]
            if info.get("maxInputChannels", 0) > 0:
                logger.info(
                    "  [%d] %s (channels=%d, rate=%d)",
                    i,
                    info.get("name", "unknown"),
                    info.get("maxInputChannels", 0),
                    int(info.get("defaultSampleRate", 0)),
                )

        self._stream = pa.open(  # type: ignore[union-attr]
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            input_device_index=self._device_index,
        )
        logger.info(
            "MicrophoneRecorder open (rate=%d, threshold=%.4f, silence=%.1fs)",
            self._rate,
            self._threshold,
            self._silence_timeout,
        )

    def close(self) -> None:
        """Close the microphone stream."""
        if self._stream is not None:
            try:
                self._stream.stop_stream()  # type: ignore[union-attr]
                self._stream.close()  # type: ignore[union-attr]
            except Exception:
                pass
        if self._audio is not None:
            try:
                self._audio.terminate()  # type: ignore[union-attr]
            except Exception:
                pass
        logger.info("MicrophoneRecorder closed")

    def __enter__(self) -> "MicrophoneRecorder":
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def record_segment(self) -> Optional[bytes]:
        """
        Block until a complete speech segment is captured.

        Waits for speech onset, records until silence timeout, then
        returns the raw PCM bytes.  Returns None if no valid speech
        was detected (e.g. too short).

        Returns
        -------
        bytes or None
            16-bit signed PCM audio bytes at self._rate Hz, or None.
        """
        if self._stream is None:
            raise RuntimeError("MicrophoneRecorder is not open. Call open() first.")

        chunk_duration = self._chunk / self._rate  # seconds per chunk
        silence_chunks_needed = int(self._silence_timeout / chunk_duration)
        min_speech_chunks = int(self._min_speech_duration / chunk_duration)

        frames = []
        speaking = False
        silence_chunks = 0

        logger.info("[LISTENING] Waiting for speech...")

        while True:
            try:
                data = self._stream.read(self._chunk, exception_on_overflow=False)  # type: ignore[union-attr]
            except Exception as exc:
                logger.warning("[LISTENING] Microphone read error: %s", exc)
                return None

            rms = _calculate_rms(data)

            if not speaking:
                if rms >= self._threshold:
                    # Speech onset detected
                    speaking = True
                    silence_chunks = 0
                    frames = [data]
                    logger.info("[SPEECH DETECTED] RMS=%.4f", rms)
            else:
                frames.append(data)

                if rms < self._threshold:
                    silence_chunks += 1
                    if silence_chunks >= silence_chunks_needed:
                        # Silence timeout reached – end of utterance
                        break
                else:
                    silence_chunks = 0

        # Filter too-short utterances
        if len(frames) < min_speech_chunks:
            logger.info(
                "[LISTENING] Utterance too short (%d chunks < %d minimum), discarding",
                len(frames),
                min_speech_chunks,
            )
            return None

        logger.info("[RECORDING] Captured %d frames (%.2f s)", len(frames), len(frames) * chunk_duration)
        return b"".join(frames)

    def pcm_to_wav(self, pcm_bytes: bytes) -> bytes:
        """
        Wrap raw PCM bytes in a WAV container.

        Parameters
        ----------
        pcm_bytes:
            Raw 16-bit signed PCM audio.

        Returns
        -------
        bytes
            Complete WAV file bytes (suitable for writing to .wav or sending to API).
        """
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self._channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self._rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()
