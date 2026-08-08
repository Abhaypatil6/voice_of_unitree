"""
tests/test_stage3_tts.py
------------------------
Unit tests for Stage 3: TTS engines.

Tests run WITHOUT needing piper or espeak installed.
The subprocess calls are mocked.

    cd g1_voice_assistant
    python3 -m pytest tests/test_stage3_tts.py -v

Python 3.8 compatible.
"""

from __future__ import annotations

import sys
import wave
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tts.base_tts import TTSError
from tts.tts_engine import create_tts_engine


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tiny_wav(path: Path, sample_rate: int = 22050) -> Path:
    """Write a tiny valid WAV file to *path* for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = sample_rate // 10  # 100ms of silence
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * num_samples, *([0] * num_samples)))
    return path


def _mock_piper_success(output_path: Path) -> MagicMock:
    """
    Return a mock CompletedProcess that also creates the output WAV file.
    Used to simulate piper subprocess success.
    """
    def _side_effect(*args, **kwargs):
        # Find output_file in the command args
        cmd = args[0]
        out_idx = cmd.index("--output_file") + 1
        out_path = Path(cmd[out_idx])
        _make_tiny_wav(out_path)
        result = MagicMock()
        result.returncode = 0
        result.stdout = b""
        result.stderr = b""
        return result
    return _side_effect


# ── Tests: PiperEngine ────────────────────────────────────────────────────────

def test_piper_synthesize_success(tmp_path: Path) -> None:
    """Successful piper call should return the output path."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "en_US-test.onnx").touch()
    (models_dir / "en_US-test.onnx.json").touch()

    with patch("subprocess.run") as mock_run:
        # version check
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = b"piper 1.0.0"
        mock_run.return_value.stderr = b""

        from tts.piper_engine import PiperEngine
        engine = PiperEngine(
            model_name="en_US-test",
            models_dir=models_dir,
            piper_binary="piper",
        )

        # synthesis call – need to create the WAV
        mock_run.side_effect = _mock_piper_success(tmp_path / "out.wav")
        out = engine.synthesize("Hello world", tmp_path / "out.wav")

    assert out.exists()
    assert out.suffix == ".wav"


def test_piper_raises_if_binary_missing(tmp_path: Path) -> None:
    """Missing piper binary should raise TTSError at construction."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "en_US-test.onnx").touch()
    (models_dir / "en_US-test.onnx.json").touch()

    with patch("subprocess.run", side_effect=FileNotFoundError("piper not found")):
        from tts.piper_engine import PiperEngine
        try:
            PiperEngine(model_name="en_US-test", models_dir=models_dir)
            assert False, "Expected TTSError"
        except TTSError as exc:
            assert "Piper binary not found" in str(exc)


def test_piper_raises_if_model_missing(tmp_path: Path) -> None:
    """Missing .onnx file should raise TTSError."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    # No .onnx file

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = b"piper 1.0.0"

        from tts.piper_engine import PiperEngine
        try:
            PiperEngine(model_name="missing_model", models_dir=models_dir)
            assert False, "Expected TTSError"
        except TTSError as exc:
            assert "model not found" in str(exc).lower()


def test_piper_raises_on_empty_text(tmp_path: Path) -> None:
    """Empty text input should raise TTSError."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "en_US-test.onnx").touch()
    (models_dir / "en_US-test.onnx.json").touch()

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = b"piper 1.0.0"

        from tts.piper_engine import PiperEngine
        engine = PiperEngine(model_name="en_US-test", models_dir=models_dir)

        try:
            engine.synthesize("", tmp_path / "out.wav")
            assert False, "Expected TTSError"
        except TTSError as exc:
            assert "empty" in str(exc).lower()


def test_piper_raises_on_subprocess_error(tmp_path: Path) -> None:
    """Non-zero return code from piper should raise TTSError."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "en_US-test.onnx").touch()
    (models_dir / "en_US-test.onnx.json").touch()

    call_count = 0

    def _side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        if call_count == 1:
            # version check OK
            result.returncode = 0
            result.stdout = b"piper 1.0.0"
            result.stderr = b""
        else:
            # synthesis fails
            result.returncode = 1
            result.stdout = b""
            result.stderr = b"ONNX error"
        return result

    with patch("subprocess.run", side_effect=_side_effect):
        from tts.piper_engine import PiperEngine
        engine = PiperEngine(model_name="en_US-test", models_dir=models_dir)
        try:
            engine.synthesize("Hello", tmp_path / "out.wav")
            assert False, "Expected TTSError"
        except TTSError as exc:
            assert "piper exited" in str(exc).lower()


# ── Tests: ESpeakEngine ───────────────────────────────────────────────────────

def test_espeak_synthesize_success(tmp_path: Path) -> None:
    """Successful espeak call should return the output path."""
    out = tmp_path / "output.wav"
    call_count = 0

    def _side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        cmd = args[0]
        result = MagicMock()
        result.returncode = 0
        result.stderr = b""
        if "-w" in cmd:
            # Synthesis call – create the WAV file
            w_idx = cmd.index("-w") + 1
            _make_tiny_wav(Path(cmd[w_idx]))
        return result

    with patch("subprocess.run", side_effect=_side_effect):
        from tts.espeak_engine import ESpeakEngine
        engine = ESpeakEngine()
        result = engine.synthesize("Hello, I am G1.", out)

    assert result == out
    assert out.exists()


def test_espeak_raises_if_not_installed(tmp_path: Path) -> None:
    """Missing espeak binary must raise TTSError."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        from tts.espeak_engine import ESpeakEngine
        try:
            ESpeakEngine()
            assert False, "Expected TTSError"
        except TTSError as exc:
            assert "espeak" in str(exc).lower()


# ── Tests: Factory ────────────────────────────────────────────────────────────

def test_create_tts_engine_none(tmp_path: Path) -> None:
    """TTS_BACKEND=none should return None."""
    result = create_tts_engine(backend="none")
    assert result is None


def test_create_tts_engine_invalid(tmp_path: Path) -> None:
    """Unknown backend should raise ValueError."""
    try:
        create_tts_engine(backend="invalid_engine")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Unknown TTS_BACKEND" in str(exc)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
