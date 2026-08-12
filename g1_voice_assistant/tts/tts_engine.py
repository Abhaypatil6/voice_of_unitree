"""
TTS engine factory.

Supported TTS backends:
    unitree_piper -> Piper -> Unitree G1 speaker
    piper         -> Piper -> local WAV
    espeak        -> eSpeak fallback
    unitree       -> Unitree native TTS
    none          -> TTS disabled

Language support for Piper:
    English -> en_US-ryan-high
    Hindi   -> hi_IN-rohan-medium

Chinese TTS is not supported.

Python 3.8 compatible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from logging_utils.logger import get_logger
from tts.base_tts import BaseTTS, TTSError

logger = get_logger(__name__)


# ==============================================================
# LANGUAGE DETECTION
# ==============================================================

def _detect_hindi_text(text: str) -> bool:
    """
    Detect whether text should be spoken using the Hindi Piper voice.

    Hindi is detected primarily by Devanagari characters.
    Romanized Hindi is detected using common Hindi words.
    """

    if not text or not text.strip():
        return False

    # ----------------------------------------------------------
    # Devanagari detection
    # ----------------------------------------------------------

    if any(
        "\u0900" <= ch <= "\u097F"
        for ch in text
    ):
        return True

    # ----------------------------------------------------------
    # Romanized Hindi detection
    # ----------------------------------------------------------

    low = text.lower()

    hindi_hints = [
        "main ",
        "mera ",
        "meri ",
        "mere ",
        "mujhe ",
        "tum ",
        "tumhara ",
        "aap ",
        "aapka ",
        "aapki ",
        "hai ",
        "hain ",
        "ho ",
        "kya ",
        "kaise ",
        "kaisa ",
        "kyun ",
        "kyon ",
        "nahin",
        "nahi",
        "haan",
        "shukriya",
        "dhanyavaad",
        "acha",
        "achha",
        "bahut",
        "kuch ",
        "chahiye",
        "karna",
        "kar ",
        "karo ",
        "sakta",
        "sakti",
        "sakhta",
    ]

    matches = sum(
        1
        for hint in hindi_hints
        if hint in low
    )

    return matches >= 2


# ==============================================================
# LANGUAGE-AWARE PIPER
# ==============================================================

class LanguageAwarePiperEngine(BaseTTS):
    """
    Selects English or Hindi Piper voice automatically.

    English:
        en_US-ryan-high

    Hindi:
        hi_IN-rohan-medium
    """

    def __init__(
        self,
        english_model: str,
        hindi_model: str,
        models_dir: Path,
        piper_binary: str,
        piper_length_scale: float,
        fallback_english_model: Optional[str] = None,
    ) -> None:

        from tts.piper_engine import PiperEngine

        # ------------------------------------------------------
        # English engine
        # ------------------------------------------------------

        try:

            self._english_engine = PiperEngine(
                model_name=english_model,
                models_dir=models_dir,
                piper_binary=piper_binary,
                length_scale=piper_length_scale,
            )

        except TTSError as exc:

            if (
                fallback_english_model is None
                or fallback_english_model == english_model
            ):
                raise

            logger.warning(
                "English Piper voice %s unavailable: %s. "
                "Falling back to %s.",
                english_model,
                exc,
                fallback_english_model,
            )

            self._english_engine = PiperEngine(
                model_name=fallback_english_model,
                models_dir=models_dir,
                piper_binary=piper_binary,
                length_scale=piper_length_scale,
            )

        # ------------------------------------------------------
        # Hindi engine
        # ------------------------------------------------------

        self._hindi_engine = None

        try:

            self._hindi_engine = PiperEngine(
                model_name=hindi_model,
                models_dir=models_dir,
                piper_binary=piper_binary,
                length_scale=piper_length_scale,
            )

        except TTSError as exc:

            logger.warning(
                "Hindi Piper model unavailable: %s. "
                "Hindi TTS will be unavailable.",
                exc,
            )

        self._english_model = english_model
        self._hindi_model = hindi_model

    # ----------------------------------------------------------
    # BaseTTS
    # ----------------------------------------------------------

    @property
    def engine_name(self) -> str:
        return "piper_language_aware"

    @property
    def sample_rate(self) -> int:
        return self._english_engine.sample_rate

    @property
    def speaks_directly(self) -> bool:
        return False

    # ----------------------------------------------------------
    # Synthesis
    # ----------------------------------------------------------

    def synthesize(
        self,
        text: str,
        output_path: Path,
    ) -> Path:

        text = text.strip()

        if not text:
            raise TTSError(
                "Cannot synthesize empty text."
            )

        # ------------------------------------------------------
        # Hindi
        # ------------------------------------------------------

        if _detect_hindi_text(text):

            if self._hindi_engine is None:

                raise TTSError(
                    "Hindi Piper model is not installed. "
                    "Expected model: "
                    f"{self._hindi_model}"
                )

            logger.info(
                "Language detected: HINDI -> Piper model=%s",
                self._hindi_model,
            )

            return self._hindi_engine.synthesize(
                text,
                output_path,
            )

        # ------------------------------------------------------
        # English
        # ------------------------------------------------------

        logger.info(
            "Language detected: ENGLISH -> Piper model=%s",
            self._english_model,
        )

        return self._english_engine.synthesize(
            text,
            output_path,
        )


# ==============================================================
# TTS FACTORY
# ==============================================================

def create_tts_engine(
    backend: str,
    piper_model: str = "en_US-ryan-high",
    piper_hindi_model: str = "hi_IN-rohan-medium",
    piper_models_dir: Path = Path("tts/models"),
    piper_binary: str = "piper",
    piper_length_scale: float = 1.0,
    fallback_english_model: Optional[str] = None,
    espeak_voice: str = "en-us",
    espeak_speed: int = 150,
    unitree_sdk_path: Optional[Path] = None,
    unitree_speaker_id: int = 0,
    unitree_network_interface: Optional[str] = None,
    unitree_fallback: str = "",
) -> Optional[BaseTTS]:

    backend = backend.strip().lower()

    # ==========================================================
    # Piper -> Unitree G1 speaker
    # ==========================================================

    if backend == "unitree_piper":

        from tts.unitree_piper_engine import (
            UnitreePiperTTSEngine,
        )

        logger.info(
            "TTS backend: Piper -> Unitree G1 speaker "
            "(English model=%s, Hindi model=%s, "
            "network_interface=%s)",
            piper_model,
            piper_hindi_model,
            unitree_network_interface or "eth0",
        )

        return UnitreePiperTTSEngine(
            model_name=piper_model,
            models_dir=piper_models_dir,
            piper_binary=piper_binary,
            length_scale=piper_length_scale,
            sdk_path=unitree_sdk_path,
            network_interface=(
                unitree_network_interface
                or "eth0"
            ),
        )

    # ==========================================================
    # Native Unitree TTS
    # ==========================================================

    if backend == "unitree":

        from tts.unitree_engine import (
            UnitreeNativeTTSEngine,
        )

        logger.info(
            "TTS backend: Unitree native TTS "
            "(sdk_path=%s, speaker_id=%s, "
            "network_interface=%s)",
            (
                str(unitree_sdk_path)
                if unitree_sdk_path is not None
                else "<auto>"
            ),
            unitree_speaker_id,
            unitree_network_interface or "eth0",
        )

        try:

            return UnitreeNativeTTSEngine(
                sdk_path=unitree_sdk_path,
                speaker_id=unitree_speaker_id,
                network_interface=(
                    unitree_network_interface
                    or "eth0"
                ),
            )

        except TTSError as exc:

            if unitree_fallback == "piper":

                logger.warning(
                    "Unitree native TTS unavailable: %s. "
                    "Falling back to Piper.",
                    exc,
                )

                backend = "piper"

            else:

                raise

    # ==========================================================
    # Local Piper
    # ==========================================================

    if backend == "piper":

        logger.info(
            "TTS backend: Piper "
            "(English=%s, Hindi=%s)",
            piper_model,
            piper_hindi_model,
        )

        return LanguageAwarePiperEngine(
            english_model=piper_model,
            hindi_model=piper_hindi_model,
            models_dir=piper_models_dir,
            piper_binary=piper_binary,
            piper_length_scale=piper_length_scale,
            fallback_english_model=fallback_english_model,
        )

    # ==========================================================
    # eSpeak fallback
    # ==========================================================

    if backend == "espeak":

        from tts.espeak_engine import ESpeakEngine

        logger.info(
            "TTS backend: espeak-ng "
            "(voice=%s, speed=%d)",
            espeak_voice,
            espeak_speed,
        )

        return ESpeakEngine(
            voice=espeak_voice,
            speed=espeak_speed,
        )

    # ==========================================================
    # Disabled
    # ==========================================================

    if backend == "none":

        logger.warning(
            "TTS_BACKEND=none - TTS is disabled."
        )

        return None

    # ==========================================================
    # Unknown backend
    # ==========================================================

    raise ValueError(
        f"Unknown TTS_BACKEND={backend!r}. "
        "Expected one of: "
        "unitree_piper, piper, unitree, "
        "espeak, none"
    )