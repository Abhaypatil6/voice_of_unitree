"""
stage2_groq_test.py
-------------------
**Stage 2 standalone test** – send a fixed prompt to Groq and verify.

This script tests the Groq LLM client in isolation:
  * No ROS 2 required
  * No TTS required
  * No speaker required

It sends a single fixed prompt and prints + saves the response.

Usage:
    cd /path/to/g1_voice_assistant
    python3 stage2_groq_test.py

Requirements:
    pip install groq python-dotenv
    GROQ_API_KEY must be set in .env

The test creates a mini session under:
    conversations/stage2_test_<timestamp>/
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import config
from logging_utils.logger import get_logger
from logging_utils.artifact_logger import ArtifactLogger
from llm.groq_client import GroqClient, LLMError
from conversation.manager import ConversationManager

logger = get_logger(__name__, level=config.logging.level)

# Fixed test prompts
TEST_PROMPTS = [
    "Hello, who are you?",
    "What can you do as a robot?",
    "Tell me something interesting in one sentence.",
]


def main() -> None:
    logger.info("=" * 60)
    logger.info("G1 Voice Assistant – Stage 2: Groq LLM Test")
    logger.info("=" * 60)

    # Validate API key
    if not config.groq.api_key:
        logger.critical(
            "GROQ_API_KEY is not set in .env!\n"
            "Copy .env.example → .env and add your Groq key:\n"
            "    GROQ_API_KEY=gsk_<your-key-here>\n"
            "Get a free key at: https://console.groq.com"
        )
        sys.exit(1)

    logger.info("Model       : %s", config.groq.model)
    logger.info("Max tokens  : %d", config.groq.max_tokens)
    logger.info("Temperature : %.2f", config.groq.temperature)
    logger.info("API key     : gsk_...%s (last 4 chars)", config.groq.api_key[-4:])
    logger.info("")

    # Create session artifacts
    config.paths.conversation_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    art = ArtifactLogger(
        base_dir=config.paths.conversation_dir,
        session_ts=f"stage2_test_{ts}",
    )

    # Initialise Groq client and conversation manager
    try:
        client = GroqClient(
            api_key=config.groq.api_key,
            model=config.groq.model,
            system_prompt=config.groq.system_prompt,
            max_tokens=config.groq.max_tokens,
            temperature=config.groq.temperature,
            artifact_logger=art,
        )
    except (ImportError, ValueError) as exc:
        logger.critical("Cannot create GroqClient: %s", exc)
        sys.exit(1)

    history_mgr = ConversationManager(
        max_history_messages=config.conversation.max_history_messages
    )

    # ── Send each test prompt ──────────────────────────────────────────────
    all_ok = True
    for i, prompt in enumerate(TEST_PROMPTS, start=1):
        logger.info("-" * 50)
        logger.info("[Turn %d] USER: %s", i, prompt)

        try:
            response = client.generate(
                counter=i,
                transcript=prompt,
                history=history_mgr.history,
            )
            history_mgr.add_exchange(prompt, response)

            logger.info("[Turn %d] G1  : %s", i, response)
            print(f"\n  USER: {prompt}")
            print(f"  G1  : {response}\n")

        except LLMError as exc:
            logger.error("[Turn %d] FAILED: %s", i, exc)
            all_ok = False
            # Continue to next prompt (demonstrate error isolation)

    # ── Finalise session ──────────────────────────────────────────────────
    art.finalise()

    logger.info("=" * 60)
    if all_ok:
        logger.info("Stage 2 PASSED – all %d turns succeeded", len(TEST_PROMPTS))
        logger.info("Artifacts saved: %s", art.session_dir)
        logger.info("Next: Stage 3 (TTS)")
    else:
        logger.warning("Stage 2 had ERRORS – check artifacts for details")
        logger.info("Artifacts saved: %s", art.session_dir)
        sys.exit(1)


if __name__ == "__main__":
    main()
