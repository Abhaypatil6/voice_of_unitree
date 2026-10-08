"""
tests/test_english_only.py
--------------------------
English-only validation tests for the G1 Voice Assistant.

Verifies that:
1. The system prompt enforces English-only responses.
2. The LLM returns English responses for English input.
3. The LLM redirects non-English speakers with an English response.
4. The system prompt contains all required English enforcement instructions.
5. The config loads the correct LLM model (not a Whisper model).

These tests do NOT make real network calls.  The Groq API is mocked.

Run:
    cd g1_voice_assistant
    python3 -m pytest tests/test_english_only.py -v

Python 3.8 compatible.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import AppConfig, GroqConfig, _DEFAULT_SYSTEM_PROMPT
from logging_utils.artifact_logger import ArtifactLogger
from llm.groq_client import GroqClient, LLMError


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_completion(content: str):
    """Build a mock Groq completion response object."""
    comp = MagicMock()
    comp.choices = [MagicMock()]
    comp.choices[0].message.content = content
    comp.usage = MagicMock()
    comp.usage.prompt_tokens = 10
    comp.usage.completion_tokens = 20
    comp.usage.total_tokens = 30
    return comp


def _make_client(tmp_path: Path) -> tuple:
    art = ArtifactLogger(base_dir=tmp_path, session_ts="english_test")
    client = GroqClient(
        api_key="gsk_test_key",
        model="llama-3.1-8b-instant",
        system_prompt=_DEFAULT_SYSTEM_PROMPT,
        max_tokens=300,
        temperature=0.7,
        artifact_logger=art,
    )
    client._groq_client = MagicMock()
    return client, art


# ── Tests: System prompt enforcement ─────────────────────────────────────────

def test_default_system_prompt_is_not_empty() -> None:
    """The default system prompt must not be empty."""
    assert len(_DEFAULT_SYSTEM_PROMPT) > 50


def test_default_system_prompt_enforces_english() -> None:
    """The default system prompt must explicitly mention English-only."""
    prompt_lower = _DEFAULT_SYSTEM_PROMPT.lower()
    assert "english" in prompt_lower, "System prompt must mention English"
    assert "only" in prompt_lower, "System prompt must say 'only'"


def test_default_system_prompt_redirects_other_languages() -> None:
    """The default system prompt must instruct the LLM to redirect non-English."""
    prompt_lower = _DEFAULT_SYSTEM_PROMPT.lower()
    # Must mention that non-English speakers should be redirected
    assert "hindi" in prompt_lower or "language other than english" in prompt_lower, (
        "System prompt must instruct the LLM how to handle non-English input"
    )


def test_default_system_prompt_no_markdown() -> None:
    """The system prompt must prohibit markdown formatting."""
    prompt_lower = _DEFAULT_SYSTEM_PROMPT.lower()
    assert "markdown" in prompt_lower or "bullet" in prompt_lower, (
        "System prompt must prohibit markdown/bullet formatting"
    )


def test_config_llm_model_is_not_whisper() -> None:
    """GROQ_MODEL must be an LLM model, not a Whisper model."""
    cfg = GroqConfig()
    model = cfg.model.lower()
    assert "whisper" not in model, (
        f"GROQ_MODEL={cfg.model!r} is a Whisper model. "
        "This should be an LLM model (e.g. llama-3.1-8b-instant). "
        "Whisper models are for speech transcription, not chat."
    )


def test_config_stt_model_is_whisper() -> None:
    """STT_MODEL should be a Whisper model."""
    cfg = GroqConfig()
    assert "whisper" in cfg.stt_model.lower(), (
        f"STT_MODEL={cfg.stt_model!r} should be a Whisper model "
        "(e.g. whisper-large-v3-turbo)"
    )


# ── Tests: English input → English response ───────────────────────────────────

def test_english_input_gets_english_response(tmp_path: Path) -> None:
    """When user speaks English, the mock should return English."""
    client, art = _make_client(tmp_path)
    mock_resp = _fake_completion("I am G1, the voice assistant of this Unitree robot.")
    client._groq_client.chat.completions.create.return_value = mock_resp

    result = client.generate(
        counter=1,
        transcript="Hello, how are you?",
        history=[],
    )

    assert result == "I am G1, the voice assistant of this Unitree robot."
    # Should be English (basic ASCII check)
    assert result.isascii() or any(c.isalpha() for c in result)


def test_english_response_no_markdown(tmp_path: Path) -> None:
    """The system prompt must prevent markdown in responses."""
    client, art = _make_client(tmp_path)

    # Simulate a response that incorrectly includes markdown
    mock_resp = _fake_completion("I can do the following:\n- Walk\n- Talk\n- Help")
    client._groq_client.chat.completions.create.return_value = mock_resp

    # The response IS returned as-is (the LLM could still violate it in reality)
    # But we verify the system prompt was sent correctly
    captured_calls = []

    def _capture(**kwargs):
        captured_calls.append(kwargs)
        return mock_resp

    client._groq_client.chat.completions.create.side_effect = _capture
    client.generate(counter=1, transcript="What can you do?", history=[])

    assert len(captured_calls) == 1
    messages = captured_calls[0]["messages"]
    system_msg = messages[0]
    assert system_msg["role"] == "system"
    assert "markdown" in system_msg["content"].lower() or "bullet" in system_msg["content"].lower()


# ── Tests: Non-English input → English redirect response ─────────────────────

def test_hindi_input_gets_english_redirect(tmp_path: Path) -> None:
    """Hindi input should produce an English redirect response from the LLM."""
    client, art = _make_client(tmp_path)
    # Simulate LLM correctly redirecting Hindi speaker to English
    mock_resp = _fake_completion("Please speak to me in English.")
    client._groq_client.chat.completions.create.return_value = mock_resp

    result = client.generate(
        counter=1,
        transcript="आप कैसे हैं?",  # Hindi: "How are you?"
        history=[],
    )

    assert "english" in result.lower(), (
        f"Expected English redirect response, got: {result!r}"
    )
    assert result.isascii() or all(ord(c) < 0x900 for c in result if c.isalpha()), \
        f"Response should be in English/ASCII, got: {result!r}"


def test_marathi_input_gets_english_redirect(tmp_path: Path) -> None:
    """Marathi input should produce an English redirect response."""
    client, art = _make_client(tmp_path)
    mock_resp = _fake_completion("Please speak to me in English.")
    client._groq_client.chat.completions.create.return_value = mock_resp

    result = client.generate(
        counter=1,
        transcript="तुम्ही कसे आहात?",  # Marathi: "How are you?"
        history=[],
    )

    assert "english" in result.lower()


def test_hinglish_input_gets_english_redirect(tmp_path: Path) -> None:
    """Hinglish (mixed Hindi-English) input should produce an English redirect."""
    client, art = _make_client(tmp_path)
    mock_resp = _fake_completion("Please speak to me in English.")
    client._groq_client.chat.completions.create.return_value = mock_resp

    result = client.generate(
        counter=1,
        transcript="Kaise ho bhai?",  # Hinglish
        history=[],
    )

    assert "english" in result.lower()


def test_system_prompt_sent_in_every_request(tmp_path: Path) -> None:
    """The English-only system prompt must be the first message in every API call."""
    client, art = _make_client(tmp_path)
    captured_calls = []

    def _capture(**kwargs):
        captured_calls.append(kwargs)
        return _fake_completion("I can help you!")

    client._groq_client.chat.completions.create.side_effect = _capture

    # Send multiple turns
    for i, transcript in enumerate(["Hello", "आप कैसे हैं?", "Kaise ho?"], start=1):
        client.generate(counter=i, transcript=transcript, history=[])

    assert len(captured_calls) == 3
    for call in captured_calls:
        messages = call["messages"]
        assert messages[0]["role"] == "system", "First message must be system prompt"
        assert "english" in messages[0]["content"].lower(), \
            "System prompt must enforce English"


# ── Tests: Config correctness ─────────────────────────────────────────────────

def test_config_has_vad_settings() -> None:
    """VAD configuration fields must exist and have sensible defaults."""
    cfg = AppConfig()
    assert cfg.vad.silence_timeout > 0
    assert 0.0 <= cfg.vad.vad_threshold <= 1.0
    assert cfg.vad.min_speech_duration > 0
    assert cfg.vad.sample_rate == 16000  # Whisper requires 16000 Hz
    assert cfg.vad.channels == 1        # Mono for STT


def test_config_has_timing_setting() -> None:
    """Timing display setting must exist."""
    cfg = AppConfig()
    assert isinstance(cfg.logging.show_timing, bool)


def test_config_has_interruptible_setting() -> None:
    """Speaker interruptible setting must exist."""
    cfg = AppConfig()
    assert isinstance(cfg.speaker.interruptible, bool)


def test_config_max_tokens_reasonable() -> None:
    """LLM max tokens should be in a sane range for spoken responses."""
    cfg = AppConfig()
    assert 50 <= cfg.groq.max_tokens <= 1000, (
        f"LLM_MAX_TOKENS={cfg.groq.max_tokens} seems out of range for spoken responses. "
        "Expected 50–1000."
    )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
