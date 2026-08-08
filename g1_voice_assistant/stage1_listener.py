"""
stage1_listener.py
------------------
**Stage 1 standalone runner** – subscribe to /audio_msg and log artifacts.

This script runs ONLY the STT listener.  No LLM, no TTS, no speaker.
It exists to verify:

    1. ROS 2 is reachable from the G1 or development machine
    2. /audio_msg is publishing
    3. JSON parsing is correct
    4. Artifacts are written correctly

Usage (on the G1 or a machine with ROS 2 Foxy):
    source /opt/ros/foxy/setup.bash
    cd /path/to/g1_voice_assistant
    python3 stage1_listener.py

Usage (on the dev machine with ROS 2 Jazzy):
    source /opt/ros/jazzy/setup.bash
    cd /path/to/g1_voice_assistant
    python3 stage1_listener.py

Press Ctrl+C to stop.

Artifacts are saved under:
    conversations/<session_ts>/01_stt/
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict

# ── Add project root to sys.path ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import config
from logging_utils.logger import get_logger
from logging_utils.artifact_logger import ArtifactLogger
from robot.g1_listener import G1Listener
from robot.ros_interface import ROSInterface

logger = get_logger(__name__, level=config.logging.level)


def main() -> None:
    """Stage 1 entry point."""
    logger.info("=" * 60)
    logger.info("G1 Voice Assistant – Stage 1: STT Listener")
    logger.info("=" * 60)
    logger.info("Topic    : %s", config.ros.audio_topic)
    logger.info("Artifacts: %s", config.paths.conversation_dir)
    logger.info("Min len  : %d chars", config.ros.min_text_length)
    logger.info("Min conf : %.2f", config.ros.min_confidence)
    logger.info("")
    logger.info("Waiting for speech on %s …  (Ctrl+C to stop)", config.ros.audio_topic)
    logger.info("=" * 60)

    # ── Create per-session artifact directory ──────────────────────────────
    config.paths.conversation_dir.mkdir(parents=True, exist_ok=True)
    artifact_logger = ArtifactLogger(base_dir=config.paths.conversation_dir)

    # ── Transcript callback (Stage 1: just log, no LLM) ───────────────────
    def on_transcript(payload: Dict[str, Any], transcript: str) -> None:
        logger.info(">>> TRANSCRIPT: %s", transcript)
        # Stage 1 is complete when this fires correctly.
        # Stage 2 will replace this with a Groq call.

    # ── Initialise ROS 2 ──────────────────────────────────────────────────
    ros = ROSInterface(node_name="g1_stage1_listener")
    try:
        ros.init()
    except ImportError as exc:
        logger.critical("ROS 2 not available: %s", exc)
        logger.critical("Source your ROS 2 setup script and retry:")
        logger.critical("    source /opt/ros/foxy/setup.bash   # on G1")
        logger.critical("    source /opt/ros/jazzy/setup.bash  # on Victus")
        sys.exit(1)

    # ── Create G1Listener ─────────────────────────────────────────────────
    listener = G1Listener(
        artifact_logger=artifact_logger,
        on_transcript_callback=on_transcript,
        min_text_length=config.ros.min_text_length,
        min_confidence=config.ros.min_confidence,
        topic=config.ros.audio_topic,
    )

    # ── Subscribe ─────────────────────────────────────────────────────────
    ros.create_subscription(
        topic=config.ros.audio_topic,
        callback=listener.ros_callback,
    )

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
        logger.info("Stage 1 listener stopped.  Artifacts: %s", artifact_logger.session_dir)


if __name__ == "__main__":
    main()
