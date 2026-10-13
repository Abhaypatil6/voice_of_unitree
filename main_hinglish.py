import os
import re
import socket
import tempfile
import wave
import time
import asyncio
import subprocess
import uuid
import inspect

import numpy as np
import requests
import whisper
import edge_tts

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient


# ============================================================
# CONFIGURATION
# ============================================================

MULTICAST_GROUP = "239.168.123.161"
PORT = 5555
INTERFACE_IP = "192.168.123.40"
NETWORK_INTERFACE = "eno1"

SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1
CHUNK_SECONDS = 4

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:8b"
OLLAMA_TIMEOUT = 120

WHISPER_MODEL = "base"
MIN_RMS = 500
MIN_WORDS = 2
MAX_REPLY_WORDS = 35

VOICES = {
    "en": "en-IN-PrabhatNeural",
    "hi": "hi-IN-MadhurNeural",
}

G1_AUDIO_TIMEOUT = 10.0

# G1 playback expects 16 kHz, mono, signed 16-bit PCM.
PCM_RATE = 16000
PCM_CHANNELS = 1
PCM_WIDTH_BYTES = 2
PLAY_CHUNK_BYTES = PCM_RATE * PCM_CHANNELS * PCM_WIDTH_BYTES * 3

# Common Romanized Hindi words. This is a lightweight heuristic:
# English and Hinglish can overlap, so not every short sentence can
# be classified perfectly from words alone.
HINDI_WORDS = {
    "main", "mein", "mujhe", "mujhko", "mera", "meri", "mere",
    "tum", "tumhe", "tumhara", "tumhari", "aap", "aapko",
    "kaise", "kya", "kyun", "kyu", "hai", "hain", "ho", "tha",
    "thi", "the", "nahi", "nahin", "haan", "accha", "achha",
    "batao", "bolo", "bol", "chahiye", "kar", "karo", "kya",
    "ka", "ki", "ke", "ko", "se", "aur", "mera", "yaar",
    "bhai", "bilkul", "thoda", "bahut", "acha", "kaise",
    "kahan", "kab", "abhi", "matlab", "sunao", "samjhao",
    "karna", "karta", "karti", "karega", "raha", "rahi",
    "rahe", "hoon", "hun", "hum", "ham", "aaj", "kal",
    "acchi", "achha", "achhe", "dhanyavaad", "shukriya"
}


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_reply_language(text):
    """Return 'hi' for Hindi/Hinglish and 'en' for English."""
    text = text.strip()
    if not text:
        return "en"

    # If Whisper returns Devanagari, treat it as Hindi.
    if any("\u0900" <= ch <= "\u097F" for ch in text):
        return "hi"

    # Normalize punctuation before checking Romanized Hindi words.
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return "en"

    hindi_hits = sum(1 for word in words if word in HINDI_WORDS)

    # Require a Hindi marker for very short text; for longer text,
    # two markers give a stronger indication of Hinglish.
    if hindi_hits >= 1 and (len(words) <= 5 or hindi_hits >= 2):
        return "hi"

    return "en"


# ============================================================
# OLLAMA RESPONSE
# ============================================================

def ask_qwen(text, language):
    if language == "en":
        language_instruction = """
Reply in natural, conversational English only.
Do not use Hindi or Hinglish. Keep the response friendly and concise.
"""
    else:
        language_instruction = """
Reply in natural, casual Hindi/Hinglish using Roman letters.
Mix simple Hindi and English naturally, as people commonly speak in India.
Do not use Devanagari characters. Keep the response friendly and concise.
"""

    prompt = f"""
You are a friendly Unitree G1 humanoid robot assistant talking to a person in India.
{language_instruction}
Answer the person's actual question directly.
Usually use one or two short sentences.
Do not use emojis, markdown, or stage directions.
Never claim that you performed an action unless you did.

Person: {text}
Assistant:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "think": False,
            "stream": False,
        },
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()

    reply = response.json().get("response", "").strip()
    reply = reply.replace("**", "").replace("*", "")
    reply = reply.replace("```", "")
    reply = " ".join(reply.split())

    words = reply.split()
    if len(words) > MAX_REPLY_WORDS:
        reply = " ".join(words[:MAX_REPLY_WORDS]).rstrip(" ,;:-") + "."

    return reply


# ============================================================
# G1 AUDIO INITIALIZATION
# ============================================================

def initialize_g1_audio():
    print("\nInitializing Unitree communication...", flush=True)
    ChannelFactoryInitialize(0, NETWORK_INTERFACE)

    audio_client = AudioClient()
    audio_client.SetTimeout(G1_AUDIO_TIMEOUT)
    audio_client.Init()

    print("G1 AudioClient initialized.", flush=True)

    try:
        print("G1 volume response:", audio_client.GetVolume(), flush=True)
    except Exception as error:
        print(f"Could not read G1 volume: {error}", flush=True)

    if not hasattr(audio_client, "PlayStream"):
        raise RuntimeError(
            "This installed AudioClient has no PlayStream method. "
            "Check the playback API in your installed Unitree SDK."
        )

    if not hasattr(audio_client, "PlayStop"):
        print("Warning: PlayStop is not available in this SDK version.", flush=True)

    try:
        print("PlayStream signature:", inspect.signature(audio_client.PlayStream), flush=True)
    except (TypeError, ValueError):
        pass

    return audio_client


# ============================================================
# EDGE TTS GENERATION
# ============================================================

async def generate_edge_voice(text, mp3_path, language):
    voice = VOICES.get(language, VOICES["en"])
    communicator = edge_tts.Communicate(text=text, voice=voice)
    await communicator.save(mp3_path)


def generate_edge_voice_with_retry(text, mp3_path, language, attempts=3):
    """Retry temporary Edge TTS connection failures."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            asyncio.run(generate_edge_voice(text, mp3_path, language))
            return
        except Exception as error:
            last_error = error
            print(
                f"Edge TTS attempt {attempt}/{attempts} failed: {error}",
                flush=True,
            )
            if attempt < attempts:
                time.sleep(2)

    raise RuntimeError(f"Edge TTS failed after {attempts} attempts: {last_error}")


def convert_mp3_to_pcm(mp3_path, wav_path):
    """Convert Edge MP3 to 16 kHz, mono, signed 16-bit PCM WAV."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", mp3_path,
            "-ar", str(PCM_RATE),
            "-ac", str(PCM_CHANNELS),
            "-c:a", "pcm_s16le",
            wav_path,
        ],
        check=True,
        timeout=60,
    )

    with wave.open(wav_path, "rb") as wav_file:
        if wav_file.getframerate() != PCM_RATE:
            raise ValueError("Unexpected WAV sample rate.")
        if wav_file.getnchannels() != PCM_CHANNELS:
            raise ValueError("Unexpected WAV channel count.")
        if wav_file.getsampwidth() != PCM_WIDTH_BYTES:
            raise ValueError("Unexpected WAV sample width.")
        pcm_data = wav_file.readframes(wav_file.getnframes())

    return pcm_data


# ============================================================
# G1 SPEAKER PLAYBACK
# ============================================================

def stop_g1_stream(audio_client, app_name):
    if hasattr(audio_client, "PlayStop"):
        try:
            audio_client.PlayStop(app_name)
        except Exception as error:
            print(f"PlayStop warning: {error}", flush=True)


def speak_reply(audio_client, reply, language):
    if not reply:
        print("No reply to speak.", flush=True)
        return False

    voice = VOICES.get(language, VOICES["en"])
    print(f"\nG1 says ({voice}): {reply}", flush=True)

    app_name = "edge_tts"
    stream_id = uuid.uuid4().hex

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            mp3_path = os.path.join(temp_dir, "reply.mp3")
            wav_path = os.path.join(temp_dir, "reply.wav")

            print("Generating Edge TTS audio...", flush=True)
            generate_edge_voice_with_retry(reply, mp3_path, language)

            print("Converting speech to PCM...", flush=True)
            pcm_data = convert_mp3_to_pcm(mp3_path, wav_path)

            if not pcm_data:
                print("Edge TTS produced empty audio.", flush=True)
                return False

            bytes_per_second = PCM_RATE * PCM_CHANNELS * PCM_WIDTH_BYTES
            duration = len(pcm_data) / bytes_per_second
            print(f"Streaming {duration:.1f} seconds to G1...", flush=True)

            offset = 0
            while offset < len(pcm_data):
                chunk = pcm_data[offset:offset + PLAY_CHUNK_BYTES]

                result = audio_client.PlayStream(app_name, stream_id, chunk)
                status = result[0] if isinstance(result, tuple) else result

                if status not in (None, 0):
                    print(f"PlayStream returned an error: {result}", flush=True)
                    stop_g1_stream(audio_client, app_name)
                    return False

                offset += len(chunk)
                time.sleep(len(chunk) / bytes_per_second)

            print("Finished sending Edge TTS audio.", flush=True)
            return True

    except Exception as error:
        print(f"Edge TTS / G1 playback error: {error}", flush=True)
        stop_g1_stream(audio_client, app_name)
        return False


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def transcribe_audio(whisper_model, audio_bytes):
    wav_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            wav_path = temp_file.name

        with wave.open(wav_path, "wb") as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_bytes)

        # Do not force English: allow Whisper to detect English or Hindi.
        result = whisper_model.transcribe(
            wav_path,
            fp16=False,
            language=None,
            task="transcribe",
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
        )

        recognized_text = result.get("text", "").strip()
        segments = result.get("segments", [])

        reliable_segments = [
            segment for segment in segments
            if segment.get("no_speech_prob", 1.0) < 0.6
            and segment.get("avg_logprob", -10.0) > -1.0
        ]

        if not recognized_text or not reliable_segments:
            return ""

        if len(recognized_text.split()) < MIN_WORDS:
            print("Ignoring very short transcription.", flush=True)
            return ""

        return recognized_text

    except Exception as error:
        print(f"Whisper transcription error: {error}", flush=True)
        return ""

    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass


# ============================================================
# G1 MICROPHONE MULTICAST
# ============================================================

def create_microphone_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", PORT))

    membership = socket.inet_aton(MULTICAST_GROUP) + socket.inet_aton(INTERFACE_IP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)

    sock.settimeout(1.0)
    return sock


# ============================================================
# MAIN ASSISTANT LOOP
# ============================================================

def main():
    audio_client = None
    sock = None

    try:
        audio_client = initialize_g1_audio()

        print("\nLoading Whisper model...", flush=True)
        whisper_model = whisper.load_model(WHISPER_MODEL)
        print("Whisper loaded.", flush=True)

        print("\nChecking Ollama...", flush=True)
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        print("Ollama is reachable.", flush=True)

        sock = create_microphone_socket()

        print("\n" + "=" * 55, flush=True)
        print(" UNITREE G1 AUTO-LANGUAGE VOICE ASSISTANT", flush=True)
        print("=" * 55, flush=True)
        print(f"Network interface: {NETWORK_INTERFACE}", flush=True)
        print(f"Microphone group: {MULTICAST_GROUP}:{PORT}", flush=True)
        print(f"Qwen model: {MODEL_NAME}", flush=True)
        print(f"English voice: {VOICES['en']}", flush=True)
        print(f"Hindi/Hinglish voice: {VOICES['hi']}", flush=True)
        print("Speak in English, Hindi, or Hinglish.", flush=True)
        print("Press Ctrl+C to stop.", flush=True)

        audio_buffer = bytearray()
        target_bytes = SAMPLE_RATE * SAMPLE_WIDTH_BYTES * CHANNELS * CHUNK_SECONDS
        last_transcription = ""
        last_transcription_time = 0.0

        while True:
            try:
                packet, address = sock.recvfrom(65535)
            except socket.timeout:
                continue

            if not packet:
                continue

            audio_buffer.extend(packet)

            while len(audio_buffer) >= target_bytes:
                chunk = bytes(audio_buffer[:target_bytes])
                del audio_buffer[:target_bytes]

                samples = np.frombuffer(chunk, dtype="<i2")
                if samples.size == 0:
                    continue

                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                if rms < MIN_RMS:
                    continue

                recognized_text = transcribe_audio(whisper_model, chunk)
                if not recognized_text:
                    continue

                now = time.monotonic()
                if (
                    recognized_text.lower() == last_transcription.lower()
                    and now - last_transcription_time < 8
                ):
                    print("Ignoring duplicate transcription.", flush=True)
                    continue

                last_transcription = recognized_text
                last_transcription_time = now

                print(f"\nYou said: {recognized_text}", flush=True)

                language = detect_reply_language(recognized_text)
                print(f"Detected reply language: {language}", flush=True)
                print("Qwen is thinking...", flush=True)

                try:
                    reply = ask_qwen(recognized_text, language)

                    if not reply:
                        print("Qwen returned an empty reply.", flush=True)
                        continue

                    speak_reply(audio_client, reply, language)

                    # Discard stale audio captured while processing/speaking.
                    audio_buffer.clear()

                except requests.Timeout:
                    print("Ollama timed out.", flush=True)
                except requests.RequestException as error:
                    print(f"Ollama request failed: {error}", flush=True)
                except Exception as error:
                    print(f"Assistant error: {error}", flush=True)

    except KeyboardInterrupt:
        print("\nStopping assistant...", flush=True)
    except Exception as error:
        print(f"Startup or runtime error: {error}", flush=True)
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        print("Assistant stopped.", flush=True)


if __name__ == "__main__":
    main()

