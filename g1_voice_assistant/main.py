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

import json
import signal
import sys
import tempfile
import time
from datetime import datetime
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
        self._state: str = "LISTENING"

    def _set_state(self, state: str) -> None:
        self._state = state
        logger.info("Pipeline state: %s", state)

    def run_turn(
        self,
        payload: Dict[str, Any],
        transcript: str,
        received_ts: Optional[float] = None,
    ) -> None:
        """
        Full pipeline handler.  Executed in the main thread synchronously.
        """
        self._counter += 1
        turn = self._counter
        self._set_state("PROCESSING")
        logger.info("=" * 55)
        logger.info("[Turn %d] USER: %s", turn, transcript)

        turn_dir = self._art.session_dir / f"turn_{turn:06d}"
        turn_dir.mkdir(parents=True, exist_ok=True)

        if received_ts is None:
            received_ts = time.time()

        # Log the final STT transcript and raw payload for this exact turn.
        (turn_dir / "user_stt.txt").write_text(transcript, encoding="utf-8")
        if payload is not None:
            (turn_dir / "stt_payload.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        metadata: Dict[str, Any] = {
            "turn_id": turn,
            "user_text": transcript,
            "assistant_text": None,
            "timestamps": {
                "stt_received": received_ts,
                "llm_start": None,
                "llm_end": None,
                "tts_start": None,
                "tts_end": None,
                "playback_start": None,
                "playback_end": None,
            },
            "status": "processing",
            "errors": {},
        }

        def _save_metadata(status: str) -> None:
            metadata["status"] = status
            metadata["updated_at"] = datetime.now().isoformat()
            (turn_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        _save_metadata("processing")

        # ── Stage 2: LLM ──────────────────────────────────────────────────
        response_text: Optional[str] = None
        try:
            metadata["timestamps"]["llm_start"] = time.time()
            response_text = self._groq.generate(
                counter=turn,
                transcript=transcript,
                history=self._conv.history,
            )
            metadata["timestamps"]["llm_end"] = time.time()
            metadata["assistant_text"] = response_text
            logger.info("[Turn %d] G1  : %s", turn, response_text)
        except LLMError as exc:
            metadata["errors"]["llm"] = str(exc)
            _save_metadata("llm_failed")
            logger.error("[Turn %d] LLM failed: %s", turn, exc)
            return  # STT artifact already saved; abort this turn

        # Update conversation history
        self._conv.add_exchange(transcript, response_text)

        # Persist the LLM response inside this turn directory.
        (turn_dir / "llm_response.txt").write_text(response_text, encoding="utf-8")

        # ── Stage 3: TTS ──────────────────────────────────────────────────
        if self._tts is None:
            logger.info("[Turn %d] TTS disabled – response not spoken", turn)
            _save_metadata("tts_disabled")
            return

        # If audio artifacts are disabled, generate into a temp WAV first and
        # preserve only the final playback file if requested.
        if config.tts.save_audio_artifacts:
            wav_path = turn_dir / f"response_{turn:06d}.wav"
            wav_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav_path = Path(temp_file.name)
            temp_file.close()

        tts_start = time.monotonic()
        wav_generated: Optional[Path] = None
        tts_error: Optional[str] = None

        try:
            metadata["timestamps"]["tts_start"] = time.time()
            wav_generated = self._tts.synthesize(response_text, wav_path)
            metadata["timestamps"]["tts_end"] = time.time()
        except TTSError as exc:
            tts_error = str(exc)
            metadata["errors"]["tts"] = tts_error
            metadata["timestamps"]["tts_end"] = time.time()
            _save_metadata("tts_failed")
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
            if not config.tts.save_audio_artifacts:
                try:
                    if wav_generated is not None:
                        wav_generated.unlink()
                    elif wav_path is not None and wav_path.exists():
                        wav_path.unlink()
                except Exception:
                    pass
            return  # LLM artifact saved; TTS artifact saved; abort speaker

        if self._tts.speaks_directly:
            logger.info("[Turn %d] TTS engine speaks directly through the robot speaker; skipping local playback.", turn)
            metadata["status"] = "complete"
            _save_metadata(metadata["status"])
            return

        # ── Stage 4: Speaker ──────────────────────────────────────────────
        if self._speaker is None:
            _save_metadata("speaker_disabled")
            if not config.tts.save_audio_artifacts:
                try:
                    wav_generated.unlink()
                except Exception:
                    pass
            return

        self._set_state("SPEAKING")
        playback_success = False
        try:
            metadata["timestamps"]["playback_start"] = time.time()
            playback_success = self._speaker.play(counter=turn, audio_path=wav_generated)
            metadata["timestamps"]["playback_end"] = time.time()
        except Exception as exc:
            metadata["errors"]["speaker"] = str(exc)
            metadata["timestamps"]["playback_end"] = time.time()
            logger.error("[Turn %d] Playback failed: %s", turn, exc)

        if not config.tts.save_audio_artifacts:
            try:
                wav_generated.unlink()
            except Exception:
                pass

        metadata["status"] = "complete" if playback_success else "speaker_failed"
        _save_metadata(metadata["status"])
        logger.info("[Turn %d] complete", turn)
        self._set_state("LISTENING")


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

    # ── Initialise ROS 2 ──────────────────────────────────────────────────
    ros = ROSInterface(node_name="g1_voice_assistant")
    try:
        ros.init()
    except ImportError as exc:
        logger.critical("ROS 2 not available: %s", exc)
        logger.critical("Source your ROS 2 setup script:\n    source /opt/ros/foxy/setup.bash")
        artifact_logger.finalise()
        sys.exit(1)

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
            fallback_english_model=config.tts.piper_fallback_model,
            espeak_voice=config.tts.espeak_voice,
            espeak_speed=config.tts.espeak_speed,
            unitree_sdk_path=config.tts.unitree_sdk_path,
            unitree_speaker_id=config.tts.unitree_speaker_id,
            unitree_network_interface=config.speaker.unitree_network_interface,
            unitree_fallback=config.tts.unitree_fallback,
        )
    except (TTSError, ValueError, ImportError) as exc:
        logger.warning("TTS engine failed to initialise: %s", exc)
        logger.warning("Voice responses will be SKIPPED (text-only mode).")
        tts_engine = None
    else:
        if tts_engine is not None and tts_engine.speaks_directly:
            logger.info("Unitree native speaker: READY")
            logger.info("Unitree network interface: %s", config.speaker.unitree_network_interface or "eth0")
            logger.info("Unitree speaker ID: %s", config.tts.unitree_speaker_id)

    # ── Initialise speaker ────────────────────────────────────────────────
    g1_speaker: Optional[G1Speaker] = None
    if tts_engine is not None and not tts_engine.speaks_directly:
        try:
            raw_speaker = create_speaker(
                backend=config.speaker.backend,
                alsa_device=config.speaker.alsa_output_device,
                network_interface=config.speaker.unitree_network_interface,
            )
            if raw_speaker is not None:
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

    listener = G1Listener(
        artifact_logger=artifact_logger,
        min_text_length=config.ros.min_text_length,
        min_confidence=config.ros.min_confidence,
        topic=config.ros.audio_topic,
    )

    ros.create_subscription(
        topic=config.ros.audio_topic,
        callback=listener.ros_callback,
    )

    # ── Print startup summary ─────────────────────────────────────────────
    if tts_engine is None:
        tts_status = "DISABLED"
    else:
        tts_status = tts_engine.engine_name

    if tts_engine is not None and tts_engine.speaks_directly:
        speaker_status = "Unitree native TTS (direct via robot audio service)"
    else:
        speaker_status = config.speaker.backend if g1_speaker is not None else "DISABLED"

    logger.info("")
    logger.info("✓ ROS 2 node started")
    logger.info("✓ Listening on: %s", config.ros.audio_topic)
    logger.info("  TTS engine : %s", tts_status)
    logger.info("  Speaker    : %s", speaker_status)
    logger.info("")
    logger.info("Waiting for speech …  (Ctrl+C to stop)")
    logger.info("=" * 60)

    # ── Spin ──────────────────────────────────────────────────────────────
    ros.spin_in_background()
    
    shutdown_event = [False]

    def _sig_handler(signum: int, frame: Any) -> None:
        logger.info("\nShutdown signal received")
        shutdown_event[0] = True

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        while not shutdown_event[0]:
            payload, final_text, received_ts = listener.get_final_transcript(silence_timeout=1.0)
            
            if final_text:
                # We have a finalized turn!
                # 1. Stop listening (to prevent echo and duplicate buffering)
                listener.set_listening(False)
                
                try:
                    pipeline.run_turn(payload, final_text, received_ts)
                finally:
                    # 2. Resume listening after speaking is done
                    listener.set_listening(True)
            else:
                time.sleep(0.05)
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
