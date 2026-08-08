"""
stage4_speaker_test.py
----------------------
Stage 4 standalone test – audio playback on the G1.

Tests that audio can be played through the G1's speaker.

BEFORE RUNNING:
    1. First run stage3_tts_test.py to generate test WAV files
    2. Check available ALSA devices: aplay -l
    3. Set ALSA_OUTPUT_DEVICE in .env (e.g. hw:1,0 for the G1's APE card)

Usage:
    cd g1_voice_assistant

    # Test with ALSA default device:
    python3 stage4_speaker_test.py

    # Test with specific ALSA device:
    ALSA_OUTPUT_DEVICE=hw:1,0 python3 stage4_speaker_test.py

DIAGNOSING THE G1 AUDIO DEVICE:
    On the G1, run:
        aplay -l
    Look for card 1: APE [NVIDIA Jetson Orin NX APE]

    Then test each device number:
        aplay -D hw:1,0 output/stage3_test_01.wav
        aplay -D hw:1,1 output/stage3_test_01.wav
        ... etc. until audio comes out of the speaker.

    Once you find the right device, set:
        ALSA_OUTPUT_DEVICE=hw:1,<N>
    in your .env file.
"""

from __future__ import annotations

import subprocess
import sys
import wave
import struct
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import config
from logging_utils.logger import get_logger
from robot.g1_speaker import ALSASpeaker, SpeakerError

logger = get_logger(__name__, level=config.logging.level)

OUTPUT_DIR = Path("output")


def generate_test_tone(path: Path, freq_hz: int = 440, duration_s: float = 1.0) -> Path:
    """Generate a simple sine-wave test tone WAV (no external dependencies)."""
    import math
    sample_rate = 22050
    n_samples = int(sample_rate * duration_s)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        samples = []
        for i in range(n_samples):
            # Sine wave at freq_hz, amplitude 60% of max (16-bit = 32767)
            val = int(32767 * 0.6 * math.sin(2 * math.pi * freq_hz * i / sample_rate))
            samples.append(val)
        wf.writeframes(struct.pack("<" + "h" * n_samples, *samples))
    return path


def list_alsa_devices() -> None:
    """Print available ALSA playback devices."""
    try:
        result = subprocess.run(["aplay", "-l"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Available ALSA devices:\n%s", result.stdout.strip())
        else:
            logger.warning("Could not list ALSA devices: %s", result.stderr)
    except FileNotFoundError:
        logger.error("aplay not found. Install with: sudo apt install alsa-utils")


def main() -> None:
    logger.info("=" * 60)
    logger.info("G1 Voice Assistant – Stage 4: Speaker Test")
    logger.info("=" * 60)
    logger.info("Speaker backend : %s", config.speaker.backend)
    logger.info("ALSA device     : %s", config.speaker.alsa_output_device or "default")
    logger.info("")

    # ── List ALSA devices ─────────────────────────────────────────────────
    list_alsa_devices()
    logger.info("")

    # ── Find or generate test audio ───────────────────────────────────────
    # Prefer TTS output from Stage 3; fall back to a generated tone
    test_wavs = sorted(OUTPUT_DIR.glob("stage3_test_*.wav"))
    tone_path = OUTPUT_DIR / "stage4_test_tone.wav"

    if test_wavs:
        test_wav = test_wavs[0]
        logger.info("Using Stage 3 output: %s", test_wav)
    else:
        logger.info("No Stage 3 WAV found – generating 440 Hz test tone")
        test_wav = generate_test_tone(tone_path)
        logger.info("Generated: %s", test_wav)

    # ── Print WAV info ────────────────────────────────────────────────────
    try:
        with wave.open(str(test_wav), "rb") as wf:
            logger.info(
                "WAV: %d Hz, %d ch, %.2f s",
                wf.getframerate(), wf.getnchannels(),
                wf.getnframes() / wf.getframerate(),
            )
    except Exception as exc:
        logger.warning("Could not read WAV info: %s", exc)

    # ── Play via ALSA ─────────────────────────────────────────────────────
    device = config.speaker.alsa_output_device
    speaker = ALSASpeaker(device=device)

    logger.info("")
    logger.info("Playing through speaker (device=%s) …", device or "default")
    logger.info("You should hear audio from the G1's speaker now.")
    logger.info("")

    try:
        speaker.play(test_wav)
        logger.info("=" * 60)
        logger.info("Stage 4 PASSED – audio playback worked!")
        logger.info("")
        logger.info("If you heard the audio, you are ready for Stage 5.")
        logger.info("If NOT heard:")
        logger.info("  1. Run: aplay -l  (to see all ALSA devices)")
        logger.info("  2. Try: aplay -D hw:1,0 %s", test_wav)
        logger.info("          aplay -D hw:1,1 %s", test_wav)
        logger.info("          ... until you find the speaker device")
        logger.info("  3. Set ALSA_OUTPUT_DEVICE=hw:1,<N> in .env")
        logger.info("  4. Re-run this test")
        logger.info("=" * 60)

    except SpeakerError as exc:
        logger.error("Stage 4 FAILED: %s", exc)
        logger.error("")
        logger.error("Diagnosis steps:")
        logger.error("  1. aplay -l               → list devices")
        logger.error("  2. aplay -D hw:1,0 %s", test_wav)
        logger.error("  3. Set ALSA_OUTPUT_DEVICE=hw:1,N in .env")
        logger.error("  4. re-run: python3 stage4_speaker_test.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
