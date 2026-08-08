"""
tests/test_stage2_llm.py
------------------------
Unit tests for Stage 2: Groq LLM client + ConversationManager.

These tests do NOT make real network calls.  The Groq API is mocked.
Run without any API key:

    cd g1_voice_assistant
    python3 -m pytest tests/test_stage2_llm.py -v

Python 3.8 compatible.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from logging_utils.artifact_logger import ArtifactLogger
from llm.groq_client import GroqClient, LLMError
from conversation.manager import ConversationManager


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_completion(content: str, prompt_tokens: int = 10, completion_tokens: int = 20):
    """Build a mock Groq completion response object."""
    comp = MagicMock()
    comp.choices = [MagicMock()]
    comp.choices[0].message.content = content
    comp.usage = MagicMock()
    comp.usage.prompt_tokens = prompt_tokens
    comp.usage.completion_tokens = completion_tokens
    comp.usage.total_tokens = prompt_tokens + completion_tokens
    return comp


def _make_client(tmp_path: Path, api_key: str = "gsk_test_key") -> tuple:
    art = ArtifactLogger(base_dir=tmp_path, session_ts="llm_test")
    client = GroqClient(
        api_key=api_key,
        model="llama-3.1-8b-instant",
        system_prompt="You are G1.",
        max_tokens=100,
        temperature=0.7,
        artifact_logger=art,
    )
    # Inject a mock SDK client so no real network or groq package needed
    client._groq_client = MagicMock()
    return client, art


# ── Tests: GroqClient ─────────────────────────────────────────────────────────

def test_groq_client_raises_without_api_key(tmp_path: Path) -> None:
    """Empty API key must raise ValueError immediately."""
    art = ArtifactLogger(base_dir=tmp_path, session_ts="no_key")
    try:
        GroqClient(
            api_key="",
            model="llama-3.1-8b-instant",
            system_prompt="You are G1.",
            max_tokens=100,
            temperature=0.7,
            artifact_logger=art,
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "GROQ_API_KEY" in str(exc)


def test_groq_generate_success(tmp_path: Path) -> None:
    """Successful API call should return response text."""
    client, art = _make_client(tmp_path)
    mock_resp = _fake_completion("Hello! I am G1, your robot assistant.")
    client._groq_client.chat.completions.create.return_value = mock_resp

    result = client.generate(
        counter=1,
        transcript="Hello, who are you?",
        history=[],
    )

    assert result == "Hello! I am G1, your robot assistant."


def test_groq_generate_saves_request_artifact(tmp_path: Path) -> None:
    """generate() must save a request JSON before making the API call."""
    client, art = _make_client(tmp_path)
    client._groq_client.chat.completions.create.return_value = _fake_completion("I can help you!")

    client.generate(counter=1, transcript="What can you do?", history=[])

    llm_dir = art.session_dir / "02_llm"
    req_files = list(llm_dir.glob("request_*.json"))
    assert len(req_files) == 1

    data = json.loads(req_files[0].read_text())
    assert data["user_transcript"] == "What can you do?"
    assert data["model"] == "llama-3.1-8b-instant"
    assert "api_key" not in str(data)  # key must never appear in artifacts


def test_groq_generate_saves_response_artifact(tmp_path: Path) -> None:
    """generate() must save a response JSON after the API call."""
    client, art = _make_client(tmp_path)
    client._groq_client.chat.completions.create.return_value = _fake_completion("One plus one is two.")

    client.generate(counter=1, transcript="What is 1+1?", history=[])

    llm_dir = art.session_dir / "02_llm"
    resp_files = list(llm_dir.glob("response_*.json"))
    assert len(resp_files) == 1

    data = json.loads(resp_files[0].read_text())
    assert data["response_text"] == "One plus one is two."
    assert "latency_s" in data
    assert data["http_status"] == 200
    assert data["token_usage"]["completion_tokens"] == 20


def test_groq_generate_raises_on_api_error(tmp_path: Path) -> None:
    """API errors must raise LLMError and still save an artifact."""
    client, art = _make_client(tmp_path)
    client._groq_client.chat.completions.create.side_effect = RuntimeError("Connection refused")

    try:
        client.generate(counter=1, transcript="Hello?", history=[])
        assert False, "Expected LLMError"
    except LLMError as exc:
        assert "Connection refused" in str(exc)

    # Response artifact must still exist even on error
    llm_dir = art.session_dir / "02_llm"
    resp_files = list(llm_dir.glob("response_*.json"))
    assert len(resp_files) == 1
    data = json.loads(resp_files[0].read_text())
    assert data["error"] is not None
    assert data["response_text"] == ""


def test_groq_history_injected_in_messages(tmp_path: Path) -> None:
    """Conversation history must appear in the messages sent to the API."""
    client, art = _make_client(tmp_path)
    mock_resp = _fake_completion("Nice to meet you!")
    captured_calls: List[Any] = []

    history = [
        {"role": "user",      "content": "Hi there!"},
        {"role": "assistant", "content": "Hello! How can I help?"},
    ]

    def _capture(**kwargs: Any) -> Any:
        captured_calls.append(kwargs)
        return mock_resp

    client._groq_client.chat.completions.create.side_effect = _capture
    client.generate(counter=1, transcript="What is your name?", history=history)

    assert len(captured_calls) == 1
    messages = captured_calls[0]["messages"]

    roles = [m["role"] for m in messages]
    assert roles[0] == "system"
    assert roles[1] == "user"      # history[0]
    assert roles[2] == "assistant" # history[1]
    assert roles[3] == "user"      # current transcript


def test_groq_api_key_never_in_artifacts(tmp_path: Path) -> None:
    """API key must never appear in any saved artifact."""
    secret_key = "gsk_THIS_IS_THE_SECRET_KEY_DO_NOT_LEAK"
    art = ArtifactLogger(base_dir=tmp_path, session_ts="security_test")
    client = GroqClient(
        api_key=secret_key,
        model="llama-3.1-8b-instant",
        system_prompt="You are G1.",
        max_tokens=100,
        temperature=0.7,
        artifact_logger=art,
    )
    # Inject mock so groq package not needed
    client._groq_client = MagicMock()
    client._groq_client.chat.completions.create.return_value = _fake_completion("I won't reveal secrets.")

    client.generate(counter=1, transcript="What is the key?", history=[])

    # Check every file in the session dir
    for f in art.session_dir.rglob("*.json"):
        content = f.read_text(encoding="utf-8")
        assert secret_key not in content, f"API key leaked into {f}"


# ── Tests: ConversationManager ────────────────────────────────────────────────

def test_conversation_history_empty_on_start() -> None:
    mgr = ConversationManager(max_history_messages=5)
    assert mgr.history == []
    assert len(mgr) == 0


def test_conversation_add_exchange() -> None:
    mgr = ConversationManager(max_history_messages=5)
    mgr.add_exchange("Hello", "Hi there!")
    h = mgr.history
    assert len(h) == 2
    assert h[0] == {"role": "user",      "content": "Hello"}
    assert h[1] == {"role": "assistant", "content": "Hi there!"}


def test_conversation_history_is_copy() -> None:
    """history property must return a copy so callers cannot mutate internal state."""
    mgr = ConversationManager(max_history_messages=5)
    mgr.add_exchange("A", "B")
    copy = mgr.history
    copy.clear()
    assert len(mgr) == 2  # internal state unchanged


def test_conversation_pruning() -> None:
    """When max_pairs is exceeded, oldest pairs are dropped."""
    mgr = ConversationManager(max_history_messages=2)  # max 2 pairs = 4 messages
    mgr.add_exchange("Turn1 user", "Turn1 asst")
    mgr.add_exchange("Turn2 user", "Turn2 asst")
    mgr.add_exchange("Turn3 user", "Turn3 asst")  # should push out Turn1

    h = mgr.history
    # Should have Turn2 and Turn3 only (4 messages)
    assert len(h) == 4
    contents = [m["content"] for m in h]
    assert "Turn1 user" not in contents
    assert "Turn2 user" in contents
    assert "Turn3 user" in contents


def test_conversation_clear() -> None:
    mgr = ConversationManager()
    mgr.add_exchange("A", "B")
    mgr.clear()
    assert mgr.history == []


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
