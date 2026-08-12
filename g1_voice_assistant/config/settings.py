"""
config/settings.py
------------------
Centralised, environment-driven configuration for the G1 Voice Assistant.

All tuneable parameters are read from a ``.env`` file (or environment
variables) via python-dotenv.  Sensible defaults are provided so the
system can at least start without a ``.env`` present.

Python 3.8 compatible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    # Load .env from the project root (g1_voice_assistant/.env)
    _ENV_PATH = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=_ENV_PATH, override=False)
except ImportError:
    pass  # dotenv optional at import time; env vars may already be set


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "").strip().lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return default


# ─────────────────────────────────────────────────────────────────────────────
# Configuration sections
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GroqConfig:
    """Groq LLM API settings."""

    api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY", ""))
    """Groq API key – never log or save this."""

    model: str = field(
        default_factory=lambda: _env("GROQ_MODEL", "llama-3.1-8b-instant")
    )
    """Groq model identifier.  Configurable without code changes."""

    max_tokens: int = field(
        default_factory=lambda: _env_int("LLM_MAX_TOKENS", 300)
    )

    temperature: float = field(
        default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.7)
    )

    system_prompt: str = field(
        default_factory=lambda: _env(
            "SYSTEM_PROMPT",
            (
                "You are G1, a helpful voice assistant built into a "
                "Unitree G1 humanoid robot. Respond only in English or Hindi. "
                "Prefer the user's language when it is English, Hindi, or Hinglish. "
                "Do not use other languages. Keep answers concise, conversational, "
                "and speech-friendly. Avoid markdown, lists, emojis, URLs, code, "
                "and unnecessary characters. Do not output pronunciation instructions."
            ),
        )
    )

    base_url: str = "https://api.groq.com/openai/v1"


@dataclass
class ROSConfig:
    """ROS 2 interface settings."""

    audio_topic: str = field(
        default_factory=lambda: _env("ROS_AUDIO_TOPIC", "/audio_msg")
    )
    """The ROS topic publishing Unitree STT JSON strings."""

    min_confidence: float = field(
        default_factory=lambda: _env_float("MIN_TRANSCRIPT_CONFIDENCE", 0.0)
    )
    """Drop transcripts below this confidence threshold (0 = accept all)."""

    min_text_length: int = field(
        default_factory=lambda: _env_int("MIN_TRANSCRIPT_LENGTH", 2)
    )
    """Drop transcripts shorter than this many characters (filters noise)."""


@dataclass
class TTSConfig:
    """Text-to-Speech settings."""

    backend: str = field(
        default_factory=lambda: _env("TTS_BACKEND", "unitree")
    )
    """TTS backend: unitree | piper | espeak | none."""

    piper_model: str = field(
        default_factory=lambda: _env("PIPER_MODEL", "en_US-ryan-high")
    )
    """Default English Piper model."""

    piper_fallback_model: str = field(
        default_factory=lambda: _env("PIPER_FALLBACK_MODEL", "en_US-ryan-high")
    )
    """Fallback English Piper model used when the preferred voice is unavailable."""

    piper_hindi_model: str = field(
        default_factory=lambda: _env("PIPER_HINDI_MODEL", "hi_IN-rohan-medium")
    )
    """Fallback Hindi Piper model name."""

    piper_models_dir: Path = field(
        default_factory=lambda: Path(_env("PIPER_MODELS_DIR", "tts/models"))
    )

    piper_binary: str = field(
        default_factory=lambda: _env("PIPER_BINARY", "piper")
    )

    piper_length_scale: float = field(
        default_factory=lambda: _env_float("PIPER_LENGTH_SCALE", 0.95)
    )
    """Speech rate: <1 faster, >1 slower."""

    unitree_sdk_path: Optional[Path] = field(
        default_factory=lambda: Path(_env("UNITREE_SDK_PATH", "")) if _env("UNITREE_SDK_PATH", "") else None
    )
    """Path to the Unitree SDK Python package root."""

    unitree_speaker_id: int = field(
        default_factory=lambda: _env_int(
            "UNITREE_TTS_SPEAKER_ID",
            _env_int("UNITREE_SPEAKER_ID", 0),
        )
    )
    """Speaker ID used for Unitree native TTS."""

    unitree_fallback: str = field(
        default_factory=lambda: _env("UNITREE_FALLBACK", "").strip().lower()
    )
    """Optional native Unitree TTS fallback when the SDK cannot initialise. Use 'piper'."""

    espeak_voice: str = field(
        default_factory=lambda: _env("ESPEAK_VOICE", "en-us")
    )
    """espeak voice name for clearer speech."""

    espeak_speed: int = field(
        default_factory=lambda: _env_int("ESPEAK_SPEED", 150)
    )
    """espeak speaking speed in words per minute."""

    save_audio_artifacts: bool = field(
        default_factory=lambda: _env_bool("SAVE_AUDIO_ARTIFACTS", False)
    )
    """Whether generated WAV audio should be stored in the conversation artifacts."""


@dataclass
class SpeakerConfig:
    """Speaker/audio playback settings."""

    backend: str = field(
        default_factory=lambda: _env("SPEAKER_BACKEND", "local")
    )
    """Speaker backend: local | unitree."""

    alsa_output_device: Optional[str] = field(
        default_factory=lambda: _env("ALSA_OUTPUT_DEVICE", "") or None
    )
    """ALSA device string, e.g. 'hw:1,0'.  None = ALSA default."""

    unitree_network_interface: Optional[str] = field(
        default_factory=lambda: _env("UNITREE_NETWORK_INTERFACE", "") or None
    )


@dataclass
class ConversationConfig:
    """Conversation memory settings."""

    max_history_messages: int = field(
        default_factory=lambda: _env_int("MAX_HISTORY_MESSAGES", 10)
    )
    """Max user+assistant message PAIRS retained in context."""


@dataclass
class PathConfig:
    """File system path settings."""

    conversation_dir: Path = field(
        default_factory=lambda: Path(_env("CONVERSATION_DIR", "./conversations"))
    )
    """Root directory for per-session conversation artifacts."""

    log_dir: Path = field(
        default_factory=lambda: Path(_env("LOG_DIR", "./logs"))
    )


@dataclass
class LogConfig:
    """Logging settings."""

    level: str = field(
        default_factory=lambda: _env("LOG_LEVEL", "INFO")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Top-level aggregator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AppConfig:
    """Single entry-point for all configuration sections."""

    groq: GroqConfig = field(default_factory=GroqConfig)
    ros: ROSConfig = field(default_factory=ROSConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    logging: LogConfig = field(default_factory=LogConfig)


# Module-level singleton – import this from other modules
config: AppConfig = AppConfig()
