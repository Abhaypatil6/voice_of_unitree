"""
Standalone Unitree G1 audio playback worker.

Pipeline:

Piper WAV
    ↓
16-bit PCM
    ↓
Mono
    ↓
16 kHz
    ↓
Unitree AudioClient.PlayStream()
    ↓
G1 speaker

Python 3.8 compatible.
"""

from __future__ import print_function

import os
import sys
import time
import wave
import uuid
import audioop


APP_NAME = "g1_voice_assistant"

# Stable streaming configuration
CHUNK_SIZE = 3200
STREAM_DELAY = 0.10

# G1 speaker volume
G1_VOLUME = 90


def load_pcm_16khz_mono(wav_path):
    """Load WAV and convert it to 16-bit mono 16-kHz PCM."""

    with wave.open(wav_path, "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    print(
        "WAV input: channels={}, width={} bytes, "
        "rate={}, bytes={}".format(
            channels,
            sample_width,
            sample_rate,
            len(frames),
        ),
        flush=True,
    )

    # ---------------------------------------------------------
    # Validate sample width
    # ---------------------------------------------------------

    if sample_width != 2:
        raise RuntimeError(
            "Expected 16-bit PCM WAV, got {}-bit".format(
                sample_width * 8
            )
        )

    # ---------------------------------------------------------
    # Stereo -> mono
    # ---------------------------------------------------------

    if channels == 2:
        frames = audioop.tomono(
            frames,
            2,
            0.5,
            0.5,
        )

    elif channels != 1:
        raise RuntimeError(
            "Unsupported channel count: {}".format(
                channels
            )
        )

    # ---------------------------------------------------------
    # Resample -> 16 kHz
    # ---------------------------------------------------------

    if sample_rate != 16000:
        print(
            "Resampling audio: {} Hz -> 16000 Hz".format(
                sample_rate
            ),
            flush=True,
        )

        frames, _ = audioop.ratecv(
            frames,
            2,
            1,
            sample_rate,
            16000,
            None,
        )

    print(
        "PCM output: channels=1, width=2 bytes, "
        "rate=16000, bytes={}".format(
            len(frames)
        ),
        flush=True,
    )

    return frames


def initialize_unitree(sdk_path, network_interface):
    """Initialize Unitree SDK and return AudioClient."""

    # ---------------------------------------------------------
    # Add SDK path
    # ---------------------------------------------------------

    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)

    # ---------------------------------------------------------
    # Add user Python packages
    # ---------------------------------------------------------

    user_site = os.path.expanduser(
        "~/.local/lib/python3.8/site-packages"
    )

    if user_site not in sys.path:
        sys.path.insert(0, user_site)

    # ---------------------------------------------------------
    # Import Unitree SDK
    # ---------------------------------------------------------

    try:
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
        )

        from unitree_sdk2py.g1.audio.g1_audio_client import (
            AudioClient,
        )

    except Exception as exc:
        print(
            "ERROR: Unitree SDK import failed: {}".format(
                exc
            ),
            flush=True,
        )

        sys.exit(2)

    # ---------------------------------------------------------
    # Initialize DDS
    # ---------------------------------------------------------

    try:
        print(
            "Unitree audio worker: "
            "initializing DDS on {}".format(
                network_interface
            ),
            flush=True,
        )

        ChannelFactoryInitialize(
            0,
            network_interface,
        )

        print(
            "Unitree audio worker: DDS initialized",
            flush=True,
        )

        # -----------------------------------------------------
        # Audio client
        # -----------------------------------------------------

        client = AudioClient()

        client.SetTimeout(10.0)
        client.Init()

        print(
            "Unitree audio worker: "
            "AudioClient initialized",
            flush=True,
        )

        # -----------------------------------------------------
        # Set speaker volume
        # -----------------------------------------------------

        try:
            result = client.SetVolume(G1_VOLUME)

            print(
                "Unitree speaker volume set to {} "
                "(result={})".format(
                    G1_VOLUME,
                    result,
                ),
                flush=True,
            )

            # Verify
            try:
                volume_result = client.GetVolume()

                print(
                    "Unitree speaker volume: {}".format(
                        volume_result
                    ),
                    flush=True,
                )

            except Exception as exc:
                print(
                    "WARNING: Could not verify volume: {}".format(
                        exc
                    ),
                    flush=True,
                )

        except Exception as exc:
            print(
                "WARNING: Could not set speaker volume: {}".format(
                    exc
                ),
                flush=True,
            )

        return client

    except Exception as exc:
        print(
            "ERROR: AudioClient initialization "
            "failed: {}".format(
                exc
            ),
            flush=True,
        )

        sys.exit(3)


def stop_playback(client):
    """Safely stop Unitree playback."""

    try:
        client.PlayStop(APP_NAME)
    except Exception:
        pass


def play_wav(client, wav_path):
    """Play a WAV file through the G1 speaker."""

    if not os.path.isfile(wav_path):
        print(
            "PLAY_ERROR WAV not found: {}".format(
                wav_path
            ),
            flush=True,
        )
        return

    try:
        # -----------------------------------------------------
        # Convert WAV
        # -----------------------------------------------------

        pcm_data = load_pcm_16khz_mono(
            wav_path
        )

        if not pcm_data:
            raise RuntimeError(
                "PCM audio data is empty"
            )

        # -----------------------------------------------------
        # Unique stream ID
        # -----------------------------------------------------

        stream_id = str(uuid.uuid4())

        print(
            "PLAY_START {} bytes".format(
                len(pcm_data)
            ),
            flush=True,
        )

        # -----------------------------------------------------
        # Stream PCM
        # -----------------------------------------------------

        total_chunks = (
            len(pcm_data) + CHUNK_SIZE - 1
        ) // CHUNK_SIZE

        chunk_number = 0

        for offset in range(
            0,
            len(pcm_data),
            CHUNK_SIZE,
        ):
            chunk = pcm_data[
                offset:
                offset + CHUNK_SIZE
            ]

            chunk_number += 1

            result = client.PlayStream(
                APP_NAME,
                stream_id,
                chunk,
            )

            if isinstance(result, tuple):
                ret_code = result[0]
            else:
                ret_code = result

            if ret_code != 0:
                raise RuntimeError(
                    "PlayStream returned {}".format(
                        ret_code
                    )
                )

            # Give the Unitree audio service enough time
            # to consume the current chunk.
            time.sleep(STREAM_DELAY)

        # -----------------------------------------------------
        # Finish playback
        # -----------------------------------------------------

        client.PlayStop(APP_NAME)

        print(
            "PLAY_COMPLETE",
            flush=True,
        )

    except Exception as exc:

        stop_playback(client)

        print(
            "PLAY_ERROR {}".format(
                exc
            ),
            flush=True,
        )


def main():

    # ---------------------------------------------------------
    # Arguments
    # ---------------------------------------------------------

    if len(sys.argv) < 2:
        print(
            "ERROR: expected Unitree SDK path",
            flush=True,
        )

        print(
            "Usage: python3 unitree_audio_worker.py "
            "<sdk_path> [network_interface]",
            flush=True,
        )

        sys.exit(1)

    sdk_path = sys.argv[1]

    network_interface = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "eth0"
    )

    # ---------------------------------------------------------
    # Initialize Unitree
    # ---------------------------------------------------------

    client = initialize_unitree(
        sdk_path,
        network_interface,
    )

    # ---------------------------------------------------------
    # Command loop
    # ---------------------------------------------------------

    for line in sys.stdin:

        line = line.strip()

        if not line:
            continue

        # =====================================================
        # QUIT
        # =====================================================

        if line == "__QUIT__":
            stop_playback(client)
            break

        # =====================================================
        # STOP
        # =====================================================

        if line == "__STOP__":

            try:
                client.PlayStop(
                    APP_NAME
                )

                print(
                    "PLAY_STOPPED",
                    flush=True,
                )

            except Exception as exc:

                print(
                    "PLAY_ERROR {}".format(
                        exc
                    ),
                    flush=True,
                )

            continue

        # =====================================================
        # PLAY
        # =====================================================

        if line.startswith("PLAY "):

            wav_path = line[5:].strip()

            play_wav(
                client,
                wav_path,
            )

            continue

        # =====================================================
        # Unknown command
        # =====================================================

        print(
            "ERROR: unknown command: {}".format(
                line
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()