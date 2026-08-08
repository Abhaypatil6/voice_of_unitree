"""
main.py
-------
G1 Voice Assistant – Full Pipeline (Stages 5 + 6)

Pipeline:
    /audio_msg → G1Listener → ConversationManager
        → GroqClient → TTSEngine → G1Speaker
        → LOOP

Usage on the G1:
    source /opt/ros/foxy/setup.bash
    cd ~/g1_voice_assistant
    python3 main.py

Environment variables are loaded from .env (see .env.example).

Python 3.8 compatible.
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# ── Add project root to sys.path ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import config
from logging_utils.logger import get_logger
from logging_utils.artifact_logger import ArtifactLogger
from robot.g1_listener import G1Listener
from robot.ros_interface import ROSInterface
from robot.g1_speaker import G1Speaker, create_speaker
from llm.groq_client import GroqClient, LLMError
from tts.tts_engine import create_tts_engine
from tts.base_tts import TTSError, BaseTTS
from conversation.manager import ConversationManager

logger = get_logger(__name__, level=config.logging.level)

# ── Session-level output directory for TTS WAV files ─────────────────────────
_TTS_OUTPUT_DIR = Path("output")


class VoiceAssistantPipeline:
    """
    Orchestrates the full STT → LLM → TTS → Speaker pipeline.

    This class is stateful for one session: it holds the conversation
    history, artifact logger, and counter.  Instantiate once per session.

    Parameters
    ----------
    artifact_logger:
        Per-session artifact logger.
    groq_client:
        Initialised GroqClient.
    tts_engine:
        Initialised TTS engine (or None to skip TTS/speaker).
    g1_speaker:
        Initialised G1Speaker (or None if tts_engine is None).
    conv_manager:
        Conversation history manager.
    """

    def __init__(
        self,
        artifact_logger: ArtifactLogger,
        groq_client: GroqClient,
        tts_engine: Optional[BaseTTS],
        g1_speaker: Optional[G1Speaker],
        conv_manager: ConversationManager,
    ) -> None:
        self._art = artifact_logger
        self._groq = groq_client
        self._tts = tts_engine
        self._speaker = g1_speaker
        self._conv = conv_manager
        self._counter: int = 0
        self._is_speaking: bool = False  # echo suppression flag

    # ──────────────────────────────────────────────────────────────────────────
    # Main callback (called by G1Listener on each valid transcript)
    # ──────────────────────────────────────────────────────────────────────────

    def on_transcript(self, payload: Dict[str, Any], transcript: str) -> None:
        """
        Full pipeline handler.  Called in the ROS spin thread.

        Design: each stage saves its artifact before proceeding.
        If any stage fails, subsequent stages are skipped but previous
        artifacts are preserved.
        """
        # ── Echo suppression: ignore transcripts while speaking ───────────
        if self._is_speaking:
            logger.info("Pipeline: skipping transcript (currently speaking)")
            return

        self._counter += 1
        turn = self._counter
        logger.info("=" * 55)
        logger.info("[Turn %d] USER: %s", turn, transcript)

        # ── Stage 2: LLM ──────────────────────────────────────────────────
        response_text: Optional[str] = None
        try:
            response_text = self._groq.generate(
                counter=turn,
                transcript=transcript,
                history=self._conv.history,
            )
            logger.info("[Turn %d] G1  : %s", turn, response_text)
        except LLMError as exc:
            logger.error("[Turn %d] LLM failed: %s", turn, exc)
            return  # STT artifact already saved; abort this turn

        # Update conversation history
        self._conv.add_exchange(transcript, response_text)

        # ── Stage 3: TTS ──────────────────────────────────────────────────
        if self._tts is None:
            logger.info("[Turn %d] TTS disabled – response not spoken", turn)
            return

        wav_path = _TTS_OUTPUT_DIR / f"response_{turn:06d}.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)

        tts_start = time.monotonic()
        wav_generated: Optional[Path] = None
        tts_error: Optional[str] = None

        try:
            wav_generated = self._tts.synthesize(response_text, wav_path)
        except TTSError as exc:
            tts_error = str(exc)
            logger.error("[Turn %d] TTS failed: %s", turn, exc)

        tts_latency = time.monotonic() - tts_start
        duration: Optional[float] = None

        if wav_generated and hasattr(self._tts, "get_wav_duration"):
            try:
                duration = self._tts.get_wav_duration(wav_generated)  # type: ignore[attr-defined]
            except Exception:
                pass

        self._art.log_tts(
            counter=turn,
            input_text=response_text,
            audio_path=wav_generated,
            tts_backend=self._tts.engine_name,
            sample_rate=self._tts.sample_rate,
            duration_s=duration,
            latency_s=tts_latency,
            error=tts_error,
        )

        if tts_error or wav_generated is None:
            return  # LLM artifact saved; TTS artifact saved; abort speaker

        # ── Stage 4: Speaker ──────────────────────────────────────────────
        if self._speaker is None:
            return

        self._is_speaking = True
        try:
            self._speaker.play(counter=turn, audio_path=wav_generated)
        finally:
            self._is_speaking = False

        logger.info("[Turn %d] complete", turn)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Bootstrap and run the full voice assistant pipeline."""
    logger.info("=" * 60)
    logger.info("G1 Voice Assistant – Full Pipeline")
    logger.info("=" * 60)
    logger.info("ROS topic    : %s", config.ros.audio_topic)
    logger.info("Groq model   : %s", config.groq.model)
    logger.info("TTS backend  : %s", config.tts.backend)
    logger.info("Speaker      : %s", config.speaker.backend)
    logger.info("Max history  : %d pairs", config.conversation.max_history_messages)
    logger.info("Artifacts    : %s", config.paths.conversation_dir)
    logger.info("=" * 60)

    # ── Validate API key ──────────────────────────────────────────────────
    if not config.groq.api_key:
        logger.critical(
            "GROQ_API_KEY is not set in .env!\n"
            "    cp .env.example .env\n"
            "    # then set GROQ_API_KEY=gsk_..."
        )
        sys.exit(1)

    # ── Create session artifact logger ────────────────────────────────────
    config.paths.conversation_dir.mkdir(parents=True, exist_ok=True)
    artifact_logger = ArtifactLogger(base_dir=config.paths.conversation_dir)

    # ── Initialise Groq client ────────────────────────────────────────────
    try:
        groq_client = GroqClient(
            api_key=config.groq.api_key,
            model=config.groq.model,
            system_prompt=config.groq.system_prompt,
            max_tokens=config.groq.max_tokens,
            temperature=config.groq.temperature,
            artifact_logger=artifact_logger,
        )
    except (ImportError, ValueError) as exc:
        logger.critical("Cannot initialise Groq client: %s", exc)
        sys.exit(1)

    # ── Initialise TTS engine ─────────────────────────────────────────────
    tts_engine: Optional[BaseTTS] = None
    try:
        tts_engine = create_tts_engine(
            backend=config.tts.backend,
            piper_model=config.tts.piper_model,
            piper_models_dir=config.tts.piper_models_dir,
            piper_binary=config.tts.piper_binary,
            piper_length_scale=config.tts.piper_length_scale,
        )
    except (TTSError, ValueError, ImportError) as exc:
        logger.warning("TTS engine failed to initialise: %s", exc)
        logger.warning("Voice responses will be SKIPPED (text-only mode).")
        tts_engine = None

    # ── Initialise speaker ────────────────────────────────────────────────
    g1_speaker: Optional[G1Speaker] = None
    if tts_engine is not None:
        try:
            raw_speaker = create_speaker(
                backend=config.speaker.backend,
                alsa_device=config.speaker.alsa_output_device,
            )
            g1_speaker = G1Speaker(
                speaker=raw_speaker,
                artifact_logger=artifact_logger,
            )
        except (ValueError, Exception) as exc:
            logger.warning("Speaker failed to initialise: %s", exc)
            logger.warning("Audio playback will be SKIPPED.")
            g1_speaker = None

    # ── Initialise conversation manager ──────────────────────────────────
    conv_manager = ConversationManager(
        max_history_messages=config.conversation.max_history_messages
    )

    # ── Build pipeline ────────────────────────────────────────────────────
    pipeline = VoiceAssistantPipeline(
        artifact_logger=artifact_logger,
        groq_client=groq_client,
        tts_engine=tts_engine,
        g1_speaker=g1_speaker,
        conv_manager=conv_manager,
    )

    # ── Initialise ROS 2 ──────────────────────────────────────────────────
    ros = ROSInterface(node_name="g1_voice_assistant")
    try:
        ros.init()
    except ImportError as exc:
        logger.critical("ROS 2 not available: %s", exc)
        logger.critical("Source your ROS 2 setup script:\n    source /opt/ros/foxy/setup.bash")
        artifact_logger.finalise()
        sys.exit(1)

    listener = G1Listener(
        artifact_logger=artifact_logger,
        on_transcript_callback=pipeline.on_transcript,
        min_text_length=config.ros.min_text_length,
        min_confidence=config.ros.min_confidence,
        topic=config.ros.audio_topic,
    )

    ros.create_subscription(
        topic=config.ros.audio_topic,
        callback=listener.ros_callback,
    )

    # ── Print startup summary ─────────────────────────────────────────────
    tts_status = tts_engine.engine_name if tts_engine else "DISABLED"
    spk_status = g1_speaker is not None
    logger.info("")
    logger.info("✓ ROS 2 node started")
    logger.info("✓ Listening on: %s", config.ros.audio_topic)
    logger.info("  TTS engine : %s", tts_status)
    logger.info("  Speaker    : %s", "enabled" if spk_status else "DISABLED")
    logger.info("")
    logger.info("Waiting for speech …  (Ctrl+C to stop)")
    logger.info("=" * 60)

    # ── Spin ──────────────────────────────────────────────────────────────
    shutdown_event = [False]

    def _sig_handler(signum: int, frame: Any) -> None:
        logger.info("\nShutdown signal received")
        shutdown_event[0] = True

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        while not shutdown_event[0]:
            ros.spin_once(timeout_sec=0.1)
    finally:
        stats = listener.stats
        logger.info(
            "Session stats: received=%d  accepted=%d  rejected=%d",
            stats["total_received"],
            stats["total_accepted"],
            stats["total_rejected"],
        )
        artifact_logger.finalise()
        ros.shutdown()
        logger.info("Artifacts saved: %s", artifact_logger.session_dir)
        logger.info("Pipeline stopped.")


if __name__ == "__main__":
    main()
