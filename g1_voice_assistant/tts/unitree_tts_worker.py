"""
Standalone Unitree G1 native TTS worker.

IMPORTANT:
This process intentionally does NOT import rclpy.

It owns the Unitree DDS connection independently from the
main ROS 2 process.
"""

from __future__ import print_function

import os
import sys


def main():
    if len(sys.argv) < 2:
        print("ERROR: expected Unitree SDK path", flush=True)
        sys.exit(1)

    sdk_path = sys.argv[1]
    network_interface = sys.argv[2] if len(sys.argv) > 2 else "eth0"
    speaker_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    # Make Unitree SDK available.
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)

    # Make user-installed CycloneDDS available.
    user_site = os.path.expanduser(
        "~/.local/lib/python3.8/site-packages"
    )

    if user_site not in sys.path:
        sys.path.insert(0, user_site)

    try:
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
        )

        from unitree_sdk2py.g1.audio.g1_audio_client import (
            AudioClient,
        )

    except Exception as exc:
        print(
            "ERROR: Unitree SDK import failed: {}".format(exc),
            flush=True,
        )
        sys.exit(2)

    try:
        print(
            "Unitree TTS worker: initializing DDS on {}".format(
                network_interface
            ),
            flush=True,
        )

        ChannelFactoryInitialize(
            0,
            network_interface,
        )

        print(
            "Unitree TTS worker: DDS initialized",
            flush=True,
        )

        client = AudioClient()

        client.SetTimeout(10.0)
        client.Init()

        print(
            "Unitree TTS worker: AudioClient initialized",
            flush=True,
        )

    except Exception as exc:
        print(
            "ERROR: Unitree AudioClient initialization failed: {}".format(
                exc
            ),
            flush=True,
        )
        sys.exit(3)

    # Read text commands from stdin.
    #
    # One line = one speech request.
    #
    # Special command:
    #     __QUIT__
    #
    for line in sys.stdin:
        text = line.strip()

        if not text:
            continue

        if text == "__QUIT__":
            break

        try:
            print(
                "TTS_REQUEST",
                flush=True,
            )

            result = client.TtsMaker(
                text,
                speaker_id,
            )

            print(
                "TTS_RESULT {}".format(result),
                flush=True,
            )

        except Exception as exc:
            print(
                "TTS_ERROR {}".format(exc),
                flush=True,
            )


if __name__ == "__main__":
    main()
