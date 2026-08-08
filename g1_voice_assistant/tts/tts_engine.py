"""
tts/tts_engine.py
-----------------
TTS engine factory.

Selects and instantiates the correct TTS backend based on configuration.

Priority (based on TTS_BACKEND env variable):
    piper   → PiperEngine  (preferred – high quality, offline)
    espeak  → ESpeakEngine (fallback – robotic but always available)
    none    → None (disables TTS; useful for testing Groq without speaker)

Python 3.8 compatible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from logging_utils.logger import get_logger
from tts.base_tts import BaseTTS, TTSError

logger = get_logger(__name__)


def create_tts_engine(
    backend: str,
    piper_model: str = "en_US-ryan-high",
    piper_models_dir: Path = Path("tts/models"),
    piper_binary: str = "piper",
    piper_length_scale: float = 1.0,
) -> Optional[BaseTTS]:
    """
    Instantiate and return the configured TTS engine.

    Parameters
    ----------
    backend:
        One of ``piper``, ``espeak``, or ``none``.
    piper_model:
        Piper voice model stem (used only when backend=piper).
    piper_models_dir:
        Directory containing .onnx and .onnx.json files.
    piper_binary:
        Path or name of the piper binary.
    piper_length_scale:
        Piper speech rate.

    Returns
    -------
    BaseTTS or None
        The TTS engine instance, or ``None`` if backend is ``none``.

    Raises
    ------
    TTSError
        If the selected backend cannot be initialised.
    ValueError
        If an unknown backend name is provided.
    """
    backend = backend.strip().lower()

    if backend == "piper":
        from tts.piper_engine import PiperEngine
        logger.info("TTS backend: Piper (model=%s)", piper_model)
        return PiperEngine(
            model_name=piper_model,
            models_dir=piper_models_dir,
            piper_binary=piper_binary,
            length_scale=piper_length_scale,
        )

    if backend == "espeak":
        from tts.espeak_engine import ESpeakEngine
        logger.info("TTS backend: espeak-ng")
        return ESpeakEngine()

    if backend == "none":
        logger.warning(
            "TTS_BACKEND=none – TTS is disabled.  "
            "LLM responses will NOT be spoken."
        )
        return None

    raise ValueError(
        f"Unknown TTS_BACKEND={backend!r}. "
        "Expected one of: piper, espeak, none"
    )
