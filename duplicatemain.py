
import os
import socket
import tempfile
import wave

import numpy as np
import requests
import whisper

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

MULTICAST_GROUP = "239.168.123.161"
PORT = 5555
INTERFACE_IP = "192.168.123.40"
NETWORK_INTERFACE = "eno1"

SAMPLE_RATE = 16000
CHUNK_SECONDS = 4
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:8b"


def ask_qwen(text):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": (
                "You are a friendly Unitree G1 robot voice assistant. "
                "Reply naturally in one or two short sentences. "
                "The user said: " + text
            ),
            "think": False,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def main():
    print("Initializing Unitree communication...")
    ChannelFactoryInitialize(0, NETWORK_INTERFACE)

    audio_client = AudioClient()
    audio_client.SetTimeout(10.0)
    audio_client.Init()
    print("G1 speaker ready.")

    print("Loading Whisper base model...")
    whisper_model = whisper.load_model("base")
    print("Whisper loaded.")

    sock = socket.socket(
        socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
    )
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", PORT))

    membership = (
        socket.inet_aton(MULTICAST_GROUP)
        + socket.inet_aton(INTERFACE_IP)
    )
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)

    print("Listening through G1's microphone.")
    print("Replies will play through G1's speaker.")
    print("Press Ctrl+C to stop.")

    audio_buffer = bytearray()
    target_bytes = SAMPLE_RATE * 2 * CHUNK_SECONDS

    try:
        while True:
            packet, address = sock.recvfrom(65535)
            audio_buffer.extend(packet)

            while len(audio_buffer) >= target_bytes:
                chunk = bytes(audio_buffer[:target_bytes])
                del audio_buffer[:target_bytes]

                samples = np.frombuffer(chunk, dtype="<i2")

                # Skip chunks that are mostly quiet.
                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                if rms < 500:
                    continue

                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as temp:
                    wav_path = temp.name

                try:
                    with wave.open(wav_path, "wb") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(SAMPLE_RATE)
                        wav_file.writeframes(chunk)

                    result = whisper_model.transcribe(
                        wav_path,
                        fp16=False,
                        language="en",
                        condition_on_previous_text=False,
                        no_speech_threshold=0.6,
                        logprob_threshold=-1.0,
                    )

                    recognized_text = result["text"].strip()
                    segments = result.get("segments", [])

                    # Ignore empty, very short, or low-confidence results.
                    reliable_segments = [
                        seg for seg in segments
                        if seg.get("no_speech_prob", 1.0) < 0.6
                        and seg.get("avg_logprob", -10.0) > -1.0
                    ]

                    if not recognized_text or not reliable_segments:
                        continue

                    if len(recognized_text.split()) < 2:
                        print("Ignoring a very short transcription.")
                        continue

                    print(f"\nYou said: {recognized_text}")
                    print("Qwen is thinking...")

                    try:
                        reply = ask_qwen(recognized_text)
                        if not reply:
                            print("Qwen returned an empty reply.")
                            continue

                        print(f"G1 says: {reply}")
                        print("Playing reply through G1 speaker...")

                        tts_result = audio_client.TtsMaker(reply, 1)
                        if tts_result != 0:
                            print(f"G1 TTS returned: {tts_result}")

                    except requests.RequestException as error:
                        print(f"Ollama request failed: {error}")
                    except Exception as error:
                        print(f"G1 speaker error: {error}")

                finally:
                    os.unlink(wav_path)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()


# cd ~/Projects/unitree_sdk2_python && PYTHONPATH="$PWD" python3 ~/Projects/Voice/g1_realtime_voice/main.py

# cd ~/Projects/unitree_sdk2_python && PYTHONPATH="$PWD" python3 ~/Projects/Voice/g1_realtime_voice/main.py