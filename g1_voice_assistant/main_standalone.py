"""
main_standalone.py
------------------
G1 Voice Assistant – Standalone Mode (No ROS 2 required)

This entry point runs the voice assistant using:
  - Local microphone (PyAudio) instead of Unitree ROS /audio_msg
  - Groq Whisper STT instead of Unitree built-in STT
  - Groq LLM (same as ROS mode)
  - Piper/espeak TTS (same as ROS mode)
  - Local speaker or ALSA (same as ROS mode)

Use this for:
  - Development and testing on a laptop without the G1 robot
  - Running on the G1 without sourcing ROS 2
  - Testing the voice pipeline end-to-end before deploying

For the full G1 ROS 2 integration (Unitree STT), use main.py instead.

Pipeline:
    Microphone (PyAudio)
        ↓
    VAD (RMS-based)
        ↓
    Groq Whisper STT
        ↓
    Groq LLM
        ↓
    Piper/espeak TTS
        ↓
    Speaker (ALSA/local)
        ↓ LOOP

Language: English only.
The STT is forced to English.
The LLM system prompt enforces English-only responses.
Non-English speech results in an English redirect response.

Usage:
    cd g1_voice_assistant
    python3 main_standalone.py

Requirements:
    pip install pyaudio groq python-dotenv
    GROQ_API_KEY must be set in .env

Python 3.8 compatible.
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path
from typing import Optional

# ── Add project root to sys.path ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import config
from logging_utils.logger import get_logger
from logging_utils.artifact_logger import ArtifactLogger
from audio.microphone import MicrophoneRecorder
from llm.groq_stt import GroqSTT, STTError
from llm.groq_client import GroqClient, LLMError
from tts.tts_engine import create_tts_engine
from tts.base_tts import TTSError, BaseTTS
from robot.g1_speaker import G1Speaker, create_speaker
from conversation.manager import ConversationManager

logger = get_logger(__name__, level=config.logging.level)

# ── Session-level output directory for TTS WAV files ─────────────────────────
_TTS_OUTPUT_DIR = Path("output")


def run_pipeline(
    mic: MicrophoneRecorder,
    stt: GroqSTT,
    groq: GroqClient,
    tts_engine: Optional[BaseTTS],
    g1_speaker: Optional[G1Speaker],
    conv_manager: ConversationManager,
    artifact_logger: ArtifactLogger,
    show_timing: bool,
    shutdown_event: list,
) -> None:
    """
    Main conversation loop for standalone mode.

    Runs until shutdown_event[0] is True (set by SIGINT/SIGTERM handler).
    """
    counter = 0

    while not shutdown_event[0]:
        # ── Stage 1: Record microphone audio ──────────────────────────────
        try:
            pcm_bytes = mic.record_segment()
        except KeyboardInterrupt:
            break
        except Exception as exc:
            logger.error("[ERROR] Microphone error: %s", exc)
            time.sleep(0.5)
            continue

        if pcm_bytes is None:
            # Too short or noise – listen again
            continue

        if shutdown_event[0]:
            break

        # ── Stage 2: Groq Whisper STT ─────────────────────────────────────
        stt_start = time.monotonic()
        wav_bytes = mic.pcm_to_wav(pcm_bytes)
        transcript: Optional[str] = None

        try:
            transcript = stt.transcribe(wav_bytes)
        except STTError as exc:
            logger.error("[ERROR] STT failed: %s", exc)
            continue

        stt_latency = time.monotonic() - stt_start

        if not transcript:
            logger.info("[TRANSCRIBING] Empty or noise transcription, ignoring")
            continue

        counter += 1
        turn = counter
        turn_start = time.monotonic()

        logger.info("=" * 55)
        logger.info("[USER] %s", transcript)

        # Save STT artifact (simulated ROS-style payload for consistency)
        artifact_logger.log_stt(
            raw_msg={
                "index": turn,
                "timestamp": int(time.time() * 1000),
                "type": 0,
                "text": transcript,
                "angle": 0,
                "speaker_id": 0,
                "emotion": "",
                "confidence": 1.0,
                "source": "groq_whisper",
            },
            transcript=transcript,
            local_received_ts=time.time(),
        )

        # ── Stage 3: LLM ──────────────────────────────────────────────────
        response_text: Optional[str] = None
        llm_latency: float = 0.0

        try:
            logger.info("[THINKING] Sending to Groq LLM...")
            llm_start = time.monotonic()
            response_text = groq.generate(
                counter=turn,
                transcript=transcript,
                history=conv_manager.history,
            )
            llm_latency = time.monotonic() - llm_start
            logger.info("[ASSISTANT] %s", response_text)
        except LLMError as exc:
            logger.error("[ERROR] LLM failed: %s", exc)
            continue

        # Update conversation history
        conv_manager.add_exchange(transcript, response_text)

        # ── Stage 4: TTS ──────────────────────────────────────────────────
        if tts_engine is None:
            logger.info("[TTS] Disabled – response not spoken")
            continue

        wav_path = _TTS_OUTPUT_DIR / f"response_{turn:06d}.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)

        tts_start = time.monotonic()
        wav_generated: Optional[Path] = None
        tts_error: Optional[str] = None
        duration: Optional[float] = None

        logger.info("[TTS] Synthesising response...")
        try:
            wav_generated = tts_engine.synthesize(response_text, wav_path)
        except TTSError as exc:
            tts_error = str(exc)
            logger.error("[ERROR] TTS failed: %s", exc)

        tts_latency = time.monotonic() - tts_start

        if wav_generated and hasattr(tts_engine, "get_wav_duration"):
            try:
                duration = tts_engine.get_wav_duration(wav_generated)  # type: ignore[attr-defined]
            except Exception:
                pass

        artifact_logger.log_tts(
            counter=turn,
            input_text=response_text,
            audio_path=wav_generated,
            tts_backend=tts_engine.engine_name,
            sample_rate=tts_engine.sample_rate,
            duration_s=duration,
            latency_s=tts_latency,
            error=tts_error,
        )

        if tts_error or wav_generated is None:
            continue

        # ── Stage 5: Speaker ──────────────────────────────────────────────
        if g1_speaker is None:
            continue

        logger.info("[PLAYING] Speaking response...")
        g1_speaker.play(counter=turn, audio_path=wav_generated)

        total_latency = time.monotonic() - turn_start

        # ── Timing summary ────────────────────────────────────────────────
        if show_timing:
            logger.info(
                "[TIMING] Turn %d  |  STT: %.2fs  LLM: %.2fs  TTS: %.2fs  Total: %.2fs",
                turn, stt_latency, llm_latency, tts_latency, total_latency,
            )

        logger.info("[LISTENING] Ready for next input")


def main() -> None:
    """Bootstrap and run the standalone voice assistant pipeline."""
    logger.info("=" * 60)
    logger.info("G1 Voice Assistant – Standalone Mode (No ROS 2)")
    logger.info("=" * 60)
    logger.info("STT model    : %s (Groq Whisper, English forced)", config.groq.stt_model)
    logger.info("LLM model    : %s", config.groq.model)
    logger.info("TTS backend  : %s", config.tts.backend)
    logger.info("Speaker      : %s", config.speaker.backend)
    logger.info("Silence tmout: %.1fs", config.vad.silence_timeout)
    logger.info("VAD threshold: %.4f", config.vad.vad_threshold)
    logger.info("Min speech   : %.1fs", config.vad.min_speech_duration)
    logger.info("Mic device   : %s", config.vad.mic_device_index or "default")
    logger.info("Max history  : %d pairs", config.conversation.max_history_messages)
    logger.info("Language     : English only")
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

    # ── Initialise Groq STT ───────────────────────────────────────────────
    try:
        stt = GroqSTT(
            api_key=config.groq.api_key,
            model=config.groq.stt_model,
            language="en",  # Force English to prevent hallucinated non-English tokens
        )
    except (ImportError, ValueError) as exc:
        logger.critical("Cannot initialise Groq STT: %s", exc)
        sys.exit(1)

    # ── Initialise Groq LLM client ────────────────────────────────────────
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
        logger.critical("Cannot initialise Groq LLM client: %s", exc)
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
        logger.warning("[TTS] Engine failed to initialise: %s", exc)
        logger.warning("[TTS] Voice responses will be SKIPPED (text-only mode).")
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
                interruptible=config.speaker.interruptible,
            )
        except (ValueError, Exception) as exc:
            logger.warning("[SPEAKER] Failed to initialise: %s", exc)
            logger.warning("[SPEAKER] Audio playback will be SKIPPED.")
            g1_speaker = None

    # ── Initialise conversation manager ──────────────────────────────────
    conv_manager = ConversationManager(
        max_history_messages=config.conversation.max_history_messages
    )

    # ── Print startup summary ─────────────────────────────────────────────
    tts_status = tts_engine.engine_name if tts_engine else "DISABLED"
    logger.info("")
    logger.info("✓ Standalone mode active")
    logger.info("✓ Microphone  : ready")
    logger.info("  STT         : Groq Whisper (%s)", config.groq.stt_model)
    logger.info("  LLM         : %s", config.groq.model)
    logger.info("  TTS         : %s", tts_status)
    logger.info("  Speaker     : %s", "enabled" if g1_speaker else "DISABLED")
    logger.info("  Language    : English only")
    logger.info("")
    logger.info("[LISTENING] Speak naturally.  (Ctrl+C to stop)")
    logger.info("=" * 60)

    # ── Shutdown event ────────────────────────────────────────────────────
    shutdown_event = [False]

    def _sig_handler(signum: int, frame: object) -> None:
        logger.info("\nShutdown signal received")
        shutdown_event[0] = True

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # ── Open microphone and run pipeline ──────────────────────────────────
    mic = MicrophoneRecorder(
        sample_rate=config.vad.sample_rate,
        channels=config.vad.channels,
        chunk_size=config.vad.chunk_size,
        vad_threshold=config.vad.vad_threshold,
        silence_timeout=config.vad.silence_timeout,
        min_speech_duration=config.vad.min_speech_duration,
        device_index=config.vad.mic_device_index,
    )

    try:
        with mic:
            run_pipeline(
                mic=mic,
                stt=stt,
                groq=groq_client,
                tts_engine=tts_engine,
                g1_speaker=g1_speaker,
                conv_manager=conv_manager,
                artifact_logger=artifact_logger,
                show_timing=config.logging.show_timing,
                shutdown_event=shutdown_event,
            )
    except ImportError as exc:
        logger.critical("[ERROR] %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("[ERROR] Unexpected pipeline error: %s", exc)
    finally:
        artifact_logger.finalise()
        logger.info("Artifacts saved: %s", artifact_logger.session_dir)
        logger.info("Standalone pipeline stopped.")


if __name__ == "__main__":
    main()
