"""
tests/test_stage4_speaker.py
-----------------------------
Unit tests for Stage 4: G1Speaker / audio playback.

Tests run WITHOUT needing aplay or sounddevice.
Subprocess calls are mocked.

    cd g1_voice_assistant
    python3 -m pytest tests/test_stage4_speaker.py -v
"""

from __future__ import annotations

import json
import sys
import wave
import struct
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from logging_utils.artifact_logger import ArtifactLogger
from robot.g1_speaker import (
    ALSASpeaker, LocalSpeaker, G1Speaker, SpeakerError, create_speaker,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tiny_wav(path: Path, sample_rate: int = 22050) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = sample_rate // 10
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * num_samples, *([0] * num_samples)))
    return path


def _make_art(tmp_path: Path) -> ArtifactLogger:
    return ArtifactLogger(base_dir=tmp_path, session_ts="speaker_test")


# ── Tests: ALSASpeaker ────────────────────────────────────────────────────────

def _make_popen_mock(returncode: int = 0, stderr: bytes = b""):
    """Build a mock subprocess.Popen object for ALSASpeaker tests."""
    mock_proc = MagicMock()
    mock_proc.returncode = returncode
    mock_proc.communicate.return_value = (b"", stderr)
    mock_proc.poll.return_value = returncode
    return mock_proc


def test_alsa_speaker_success(tmp_path: Path) -> None:
    """ALSASpeaker.play() with aplay returning 0 should succeed."""
    wav = _make_tiny_wav(tmp_path / "test.wav")
    speaker = ALSASpeaker(device="hw:1,0")

    mock_proc = _make_popen_mock(returncode=0)

    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        speaker.play(wav)  # should not raise

    # Verify aplay was called with correct device flag
    cmd = mock_popen.call_args[0][0]
    assert "aplay" in cmd
    assert "-D" in cmd
    assert "hw:1,0" in cmd


def test_alsa_speaker_default_device(tmp_path: Path) -> None:
    """ALSASpeaker with no device should NOT include -D flag."""
    wav = _make_tiny_wav(tmp_path / "test.wav")
    speaker = ALSASpeaker(device=None)

    mock_proc = _make_popen_mock(returncode=0)

    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        speaker.play(wav)

    cmd = mock_popen.call_args[0][0]
    assert "-D" not in cmd


def test_alsa_speaker_raises_on_aplay_error(tmp_path: Path) -> None:
    """Non-zero aplay exit code should raise SpeakerError."""
    wav = _make_tiny_wav(tmp_path / "test.wav")
    speaker = ALSASpeaker()

    mock_proc = _make_popen_mock(returncode=1, stderr=b"ALSA lib: no such device")

    with patch("subprocess.Popen", return_value=mock_proc):
        try:
            speaker.play(wav)
            assert False, "Expected SpeakerError"
        except SpeakerError as exc:
            assert "aplay failed" in str(exc).lower()


def test_alsa_speaker_raises_if_aplay_missing(tmp_path: Path) -> None:
    """Missing aplay binary should raise SpeakerError."""
    wav = _make_tiny_wav(tmp_path / "test.wav")
    speaker = ALSASpeaker()

    with patch("subprocess.Popen", side_effect=FileNotFoundError("aplay")):
        try:
            speaker.play(wav)
            assert False, "Expected SpeakerError"
        except SpeakerError as exc:
            assert "aplay" in str(exc).lower()


def test_speaker_raises_on_missing_file(tmp_path: Path) -> None:
    """Playing a non-existent file should raise FileNotFoundError."""
    speaker = ALSASpeaker()
    try:
        speaker.play(tmp_path / "does_not_exist.wav")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


# ── Tests: G1Speaker (with artifact logging and barge-in) ────────────────────

def test_g1speaker_saves_success_artifact(tmp_path: Path) -> None:
    """Successful playback must create a playback artifact."""
    wav = _make_tiny_wav(tmp_path / "resp.wav")
    art = _make_art(tmp_path)

    inner = MagicMock(spec=ALSASpeaker)
    inner.backend_name = "alsa"
    inner.is_playing = False
    inner.play.return_value = None  # success

    g1 = G1Speaker(speaker=inner, artifact_logger=art, interruptible=True)
    ok = g1.play(counter=1, audio_path=wav)

    assert ok is True
    artifacts = list((art.session_dir / "04_speaker").glob("playback_*.json"))
    assert len(artifacts) == 1
    data = json.loads(artifacts[0].read_text())
    assert data["success"] is True
    assert data["error"] is None


def test_g1speaker_saves_failure_artifact(tmp_path: Path) -> None:
    """Failed playback must still create an artifact and return False."""
    wav = _make_tiny_wav(tmp_path / "resp.wav")
    art = _make_art(tmp_path)

    inner = MagicMock(spec=ALSASpeaker)
    inner.backend_name = "alsa"
    inner.is_playing = False
    inner.play.side_effect = SpeakerError("ALSA device busy")

    g1 = G1Speaker(speaker=inner, artifact_logger=art, interruptible=True)
    ok = g1.play(counter=1, audio_path=wav)

    assert ok is False
    artifacts = list((art.session_dir / "04_speaker").glob("playback_*.json"))
    assert len(artifacts) == 1
    data = json.loads(artifacts[0].read_text())
    assert data["success"] is False
    assert "ALSA device busy" in data["error"]


def test_g1speaker_does_not_raise_on_error(tmp_path: Path) -> None:
    """G1Speaker.play() must never raise – errors are caught and logged."""
    wav = _make_tiny_wav(tmp_path / "resp.wav")
    art = _make_art(tmp_path)

    inner = MagicMock(spec=ALSASpeaker)
    inner.backend_name = "alsa"
    inner.is_playing = False
    inner.play.side_effect = RuntimeError("Unexpected crash!")

    g1 = G1Speaker(speaker=inner, artifact_logger=art, interruptible=True)
    # Must not raise
    result = g1.play(counter=1, audio_path=wav)
    assert result is False


def test_g1speaker_stop_when_interruptible(tmp_path: Path) -> None:
    """G1Speaker.stop() should call the inner speaker's stop() when interruptible."""
    art = _make_art(tmp_path)
    inner = MagicMock(spec=ALSASpeaker)
    inner.backend_name = "alsa"
    inner.stop.return_value = True

    g1 = G1Speaker(speaker=inner, artifact_logger=art, interruptible=True)
    result = g1.stop()

    assert result is True
    inner.stop.assert_called_once()


def test_g1speaker_stop_when_not_interruptible(tmp_path: Path) -> None:
    """G1Speaker.stop() should return False when interruptible=False."""
    art = _make_art(tmp_path)
    inner = MagicMock(spec=ALSASpeaker)
    inner.backend_name = "alsa"

    g1 = G1Speaker(speaker=inner, artifact_logger=art, interruptible=False)
    result = g1.stop()

    assert result is False
    inner.stop.assert_not_called()


# ── Tests: Factory ────────────────────────────────────────────────────────────

def test_create_speaker_alsa(tmp_path: Path) -> None:
    speaker = create_speaker("alsa", alsa_device="hw:1,0")
    assert speaker.backend_name == "alsa"
    assert isinstance(speaker, ALSASpeaker)


def test_create_speaker_local(tmp_path: Path) -> None:
    speaker = create_speaker("local")
    assert speaker.backend_name == "local"
    assert isinstance(speaker, LocalSpeaker)


def test_create_speaker_invalid() -> None:
    try:
        create_speaker("invalid")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "SPEAKER_BACKEND" in str(exc)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
