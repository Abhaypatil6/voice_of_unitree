"""
tests/test_g1_native_tts.py
--------------------------
Standalone Unitree G1 native TTS smoke test.

Usage:
    python3 tests/test_g1_native_tts.py <network_interface>

This script uses the official Unitree SDK API exactly as shown in the
Unitree examples. It does not perform any local WAV playback, Piper
synthesis, or speaker/motor control beyond the native audio service.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

SDK_ROOT = Path("/home/unitree/Projects/unitree_sdk2_python")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone Unitree G1 native TTS smoke test"
    )
    parser.add_argument(
        "network_interface",
        help="Network interface to use for Unitree DDS, e.g. eth0",
    )
    args = parser.parse_args()

    if not SDK_ROOT.exists():
        print(f"ERROR: Unitree SDK source not found at {SDK_ROOT}")
        return 2

    sys.path.insert(0, str(SDK_ROOT))
    os.environ.setdefault("PYTHONPATH", str(SDK_ROOT))

    print("Initializing Unitree DDS...")
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
    except ImportError as exc:
        print("ERROR: Failed to import unitree_sdk2py:", exc)
        return 3

    try:
        ChannelFactoryInitialize(0, args.network_interface)
    except Exception as exc:
        print("ERROR: ChannelFactoryInitialize failed:", exc)
        return 4

    print("Initializing G1 AudioClient...")
    try:
        audio_client = AudioClient()
        audio_client.SetTimeout(10.0)
        audio_client.Init()
    except Exception as exc:
        print("ERROR: AudioClient initialization failed:", exc)
        return 5

    print("Getting current volume...")
    try:
        volume = audio_client.GetVolume()
        print(f"Current volume: {volume}")
    except Exception as exc:
        print("ERROR: GetVolume failed:", exc)
        return 6

    print("Setting volume to 85...")
    try:
        audio_client.SetVolume(85)
    except Exception as exc:
        print("ERROR: SetVolume failed:", exc)
        return 7

    print("Sending native TTS request...")
    try:
        code = audio_client.TtsMaker(
            "Hello. This is a test of the Unitree G1 native speaker.",
            0,
        )
    except Exception as exc:
        print("ERROR: TtsMaker failed:", exc)
        return 8

    print(f"TTS return code: {code}")
    print("Waiting for speech to finish...")
    time.sleep(8.0)
    print("Done. Note: return code does not guarantee physical speaker success.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
