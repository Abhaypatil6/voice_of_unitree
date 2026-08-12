"""
stage3_unitree_tts_test.py
--------------------------
Standalone test for Unitree G1 native TTS.

This script initializes the Unitree SDK audio client, connects to the robot's
native audio service, and sends a short test sentence via TtsMaker.

Usage:
    cd g1_voice_assistant
    TTS_BACKEND=unitree UNITREE_SDK_PATH=/path/to/unitree_sdk2_python \
        UNITREE_NETWORK_INTERFACE=eth0 python3 stage3_unitree_tts_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import config
from logging_utils.logger import get_logger
from tts.tts_engine import create_tts_engine
from tts.base_tts import TTSError

logger = get_logger(__name__, level=config.logging.level)

TEST_TEXT = "Hello, this is a Unitree G1 native TTS test."
OUTPUT_PATH = Path("output/unitree_tts_test.txt")


def main() -> None:
    logger.info("Unitree native TTS test starting")
    logger.info("TTS backend: %s", config.tts.backend)
    logger.info("Unitree SDK path: %s", config.tts.unitree_sdk_path)
    logger.info("Unitree speaker_id: %s", config.tts.unitree_speaker_id)
    logger.info("Unitree network interface: %s", config.speaker.unitree_network_interface)

    try:
        tts = create_tts_engine(
            backend="unitree",
            unitree_sdk_path=config.tts.unitree_sdk_path,
            unitree_speaker_id=config.tts.unitree_speaker_id,
            unitree_network_interface=config.speaker.unitree_network_interface,
        )
    except (TTSError, ValueError, ImportError) as exc:
        logger.error("Failed to initialise Unitree native TTS engine: %s", exc)
        sys.exit(1)

    if tts is None:
        logger.error("Unitree native TTS engine returned None")
        sys.exit(1)

    logger.info("Sending text to Unitree audio service: %s", TEST_TEXT)

    try:
        result_path = tts.synthesize(TEST_TEXT, OUTPUT_PATH)
        logger.info("Unitree TTS invocation succeeded, output record: %s", result_path)
        print("Unitree native TTS test succeeded.")
        print("The G1 should speak the test sentence through its native speaker.")
    except TTSError as exc:
        logger.error("Unitree native TTS test failed: %s", exc)
        print("Unitree native TTS test failed:", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
