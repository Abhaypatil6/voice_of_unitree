"""
stage3_tts_test.py
------------------
Stage 3 standalone test – TTS synthesis without ROS or Groq.

Tests that the TTS engine can:
    1. Synthesise a fixed text string
    2. Write a valid WAV file
    3. Measure latency

Usage:
    cd g1_voice_assistant

    # With espeak-ng (always available):
    TTS_BACKEND=espeak python3 stage3_tts_test.py

    # With Piper (after installing the binary and model):
    TTS_BACKEND=piper python3 stage3_tts_test.py

The output WAV is saved to output/stage3_test.wav and played via ALSA.
"""

from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import config
from logging_utils.logger import get_logger
from tts.tts_engine import create_tts_engine
from tts.base_tts import TTSError

logger = get_logger(__name__, level=config.logging.level)

TEST_TEXTS = [
    "Hello, I am G1, your Unitree robot assistant.",
    "I can hear you through my microphone and speak to you through my speaker.",
    "Stage three text to speech is working correctly.",
]

OUTPUT_DIR = Path("output")


def check_wav(path: Path) -> dict:
    """Read WAV properties for validation."""
    with wave.open(str(path), "rb") as wf:
        return {
            "channels": wf.getnchannels(),
            "sample_width": wf.getsampwidth(),
            "frame_rate": wf.getframerate(),
            "n_frames": wf.getnframes(),
            "duration_s": wf.getnframes() / wf.getframerate(),
        }


def main() -> None:
    logger.info("=" * 60)
    logger.info("G1 Voice Assistant – Stage 3: TTS Test")
    logger.info("=" * 60)
    logger.info("TTS backend  : %s", config.tts.backend)
    if config.tts.backend == "piper":
        logger.info("Piper model  : %s", config.tts.piper_model)
        logger.info("Piper binary : %s", config.tts.piper_binary)
    logger.info("")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Initialise TTS ────────────────────────────────────────────────────
    try:
        tts = create_tts_engine(
            backend=config.tts.backend,
            piper_model=config.tts.piper_model,
            piper_models_dir=config.tts.piper_models_dir,
            piper_binary=config.tts.piper_binary,
            piper_length_scale=config.tts.piper_length_scale,
        )
    except (TTSError, ValueError, ImportError) as exc:
        logger.critical("TTS engine init failed: %s", exc)
        sys.exit(1)

    if tts is None:
        logger.critical("TTS_BACKEND=none – nothing to test.")
        sys.exit(1)

    # ── Synthesise each test text ─────────────────────────────────────────
    all_ok = True
    for i, text in enumerate(TEST_TEXTS, start=1):
        out_path = OUTPUT_DIR / f"stage3_test_{i:02d}.wav"
        logger.info("[%d/%d] Synthesising: %.60s", i, len(TEST_TEXTS), text)

        t_start = time.monotonic()
        try:
            wav = tts.synthesize(text, out_path)
            latency = time.monotonic() - t_start

            props = check_wav(wav)
            logger.info(
                "  → %s  [%.2f s latency | %.1f s audio | %d Hz]",
                wav.name, latency, props["duration_s"], props["frame_rate"],
            )
            print(f"\n  Text   : {text}")
            print(f"  WAV    : {wav}")
            print(f"  Latency: {latency:.2f} s")
            print(f"  Audio  : {props['duration_s']:.1f} s @ {props['frame_rate']} Hz\n")

        except TTSError as exc:
            logger.error("  FAILED: %s", exc)
            all_ok = False

    # ── Result ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    if all_ok:
        logger.info("Stage 3 PASSED – WAV files saved in: %s", OUTPUT_DIR)
        logger.info("")
        logger.info("Next: Stage 4 – play the WAV through the G1 speaker:")
        logger.info("    python3 stage4_speaker_test.py")
        logger.info("")
        logger.info("To play manually with aplay:")
        logger.info("    aplay output/stage3_test_01.wav")
        logger.info("    aplay -D hw:1,0 output/stage3_test_01.wav  # APE card")
    else:
        logger.error("Stage 3 had ERRORS – fix TTS before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
