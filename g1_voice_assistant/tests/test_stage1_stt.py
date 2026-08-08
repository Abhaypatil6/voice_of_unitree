"""
tests/test_stage1_stt.py
------------------------
Unit tests for Stage 1: STT listener (no ROS required).

Tests run entirely without ROS 2.  The G1Listener._handle_raw_json()
method is called directly with sample JSON strings.

Run on the dev machine:
    cd g1_voice_assistant
    pip install pytest
    python -m pytest tests/test_stage1_stt.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from logging_utils.artifact_logger import ArtifactLogger
from robot.g1_listener import G1Listener


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_listener(tmp_dir: Path) -> Tuple[G1Listener, List[Tuple[Dict, str]], ArtifactLogger]:
    """Create a G1Listener with a temp artifact dir and a capture callback."""
    received: List[Tuple[Dict, str]] = []

    art = ArtifactLogger(base_dir=tmp_dir, session_ts="test_session")

    def _cb(payload: Dict[str, Any], text: str) -> None:
        received.append((payload, text))

    listener = G1Listener(
        artifact_logger=art,
        on_transcript_callback=_cb,
        min_text_length=2,
        min_confidence=0.0,
    )
    return listener, received, art


SAMPLE_MSG = {
    "index": 236,
    "timestamp": 1786166118025,
    "type": 0,
    "text": "Hello, who are you?",
    "angle": 0,
    "speaker_id": 0,
    "emotion": "<|NEUTRAL|>",
    "confidence": 0.92,
}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_valid_transcript_accepted(tmp_path: Path) -> None:
    """A normal transcript should be accepted and trigger the callback."""
    listener, received, art = _make_listener(tmp_path)

    result = listener._handle_raw_json(json.dumps(SAMPLE_MSG))

    assert result is True
    assert len(received) == 1
    payload, text = received[0]
    assert text == "Hello, who are you?"
    assert payload["index"] == 236


def test_artifacts_created(tmp_path: Path) -> None:
    """Accepting a transcript must create raw JSON and transcript text files."""
    listener, received, art = _make_listener(tmp_path)
    listener._handle_raw_json(json.dumps(SAMPLE_MSG))

    stt_dir = art.session_dir / "01_stt"
    raw_files = list(stt_dir.glob("raw_*.json"))
    transcript_files = list(stt_dir.glob("transcript_*.txt"))

    assert len(raw_files) == 1, f"Expected 1 raw file, got: {raw_files}"
    assert len(transcript_files) == 1

    # Verify raw file contents
    raw_data = json.loads(raw_files[0].read_text(encoding="utf-8"))
    assert raw_data["raw_audio_msg"]["index"] == 236
    assert "local_received_epoch" in raw_data
    assert "local_received_iso" in raw_data

    # Verify transcript content
    transcript = transcript_files[0].read_text(encoding="utf-8")
    assert transcript == "Hello, who are you?"


def test_duplicate_index_rejected(tmp_path: Path) -> None:
    """The same Unitree index should only be processed once."""
    listener, received, art = _make_listener(tmp_path)

    msg_json = json.dumps(SAMPLE_MSG)
    r1 = listener._handle_raw_json(msg_json)
    r2 = listener._handle_raw_json(msg_json)  # same index=236

    assert r1 is True
    assert r2 is False
    assert len(received) == 1


def test_short_text_rejected(tmp_path: Path) -> None:
    """A transcript that is too short should be rejected."""
    listener, received, art = _make_listener(tmp_path)

    msg = {**SAMPLE_MSG, "index": 237, "text": "."}  # 1 char < min_len=2
    result = listener._handle_raw_json(json.dumps(msg))

    assert result is False
    assert len(received) == 0


def test_empty_text_rejected(tmp_path: Path) -> None:
    """Empty transcript text should be rejected."""
    listener, received, art = _make_listener(tmp_path)

    msg = {**SAMPLE_MSG, "index": 238, "text": ""}
    result = listener._handle_raw_json(json.dumps(msg))

    assert result is False


def test_malformed_json_rejected(tmp_path: Path) -> None:
    """Invalid JSON must be silently dropped without crashing."""
    listener, received, art = _make_listener(tmp_path)

    result = listener._handle_raw_json("{this is not valid json")

    assert result is False
    assert len(received) == 0


def test_confidence_filter(tmp_path: Path) -> None:
    """Transcripts below the confidence threshold should be rejected."""
    art = ArtifactLogger(base_dir=tmp_path, session_ts="conf_test")
    received = []

    listener = G1Listener(
        artifact_logger=art,
        on_transcript_callback=lambda p, t: received.append(t),
        min_text_length=2,
        min_confidence=0.8,  # require 0.8
    )

    # Low confidence
    low_conf_msg = {**SAMPLE_MSG, "index": 300, "confidence": 0.5}
    r1 = listener._handle_raw_json(json.dumps(low_conf_msg))
    assert r1 is False

    # High confidence
    high_conf_msg = {**SAMPLE_MSG, "index": 301, "confidence": 0.9}
    r2 = listener._handle_raw_json(json.dumps(high_conf_msg))
    assert r2 is True
    assert len(received) == 1


def test_different_indices_both_accepted(tmp_path: Path) -> None:
    """Two messages with different indices should both be accepted."""
    listener, received, art = _make_listener(tmp_path)

    msg1 = {**SAMPLE_MSG, "index": 100, "text": "Hello world"}
    msg2 = {**SAMPLE_MSG, "index": 101, "text": "How are you"}

    r1 = listener._handle_raw_json(json.dumps(msg1))
    r2 = listener._handle_raw_json(json.dumps(msg2))

    assert r1 is True
    assert r2 is True
    assert len(received) == 2


def test_noisy_unitree_samples(tmp_path: Path) -> None:
    """Reproduce the noisy STT examples from the spec.  Most should be rejected."""
    art = ArtifactLogger(base_dir=tmp_path, session_ts="noisy_test")
    accepted_texts = []

    noisy_listener = G1Listener(
        artifact_logger=art,
        on_transcript_callback=lambda p, t: accepted_texts.append(t),
        min_text_length=2,
        min_confidence=0.0,
    )

    noisy = [
        {"index": 1, "text": "还有 youて。"},   # len > 2 → accepted
        {"index": 2, "text": "you。"},          # len > 2 → accepted
        {"index": 3, "text": "."},              # 1 char < min_len=2 → rejected
        {"index": 4, "text": "Hello."},         # good → accepted
    ]

    results = []
    for item in noisy:
        msg = {**SAMPLE_MSG, **item, "confidence": 0.5}
        results.append(noisy_listener._handle_raw_json(json.dumps(msg)))

    # "." should be the only rejection (len < 2)
    assert results[2] is False, "Single dot should be rejected"
    # Stats use the correct listener
    assert noisy_listener.stats["total_received"] == len(noisy)
    assert noisy_listener.stats["total_accepted"] == 3  # indices 1,2,4


def test_session_json_created(tmp_path: Path) -> None:
    """session.json must exist right after ArtifactLogger is created."""
    art = ArtifactLogger(base_dir=tmp_path, session_ts="sess_test")
    session_file = art.session_dir / "session.json"
    assert session_file.exists()
    data = json.loads(session_file.read_text())
    assert "started_at" in data


def test_finalise_updates_session_json(tmp_path: Path) -> None:
    """Calling finalise() should add ended_at and total_exchanges."""
    art = ArtifactLogger(base_dir=tmp_path, session_ts="finalise_test")
    listener, received, _ = _make_listener(tmp_path)

    # Use the art from _make_listener
    art2 = ArtifactLogger(base_dir=tmp_path, session_ts="finalise2_test")
    listener2 = G1Listener(
        artifact_logger=art2,
        on_transcript_callback=lambda p, t: None,
        min_text_length=2,
    )
    listener2._handle_raw_json(json.dumps(SAMPLE_MSG))
    art2.finalise()

    data = json.loads((art2.session_dir / "session.json").read_text())
    assert "ended_at" in data
    assert data["total_exchanges"] == 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
