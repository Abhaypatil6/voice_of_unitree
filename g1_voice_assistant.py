#!/usr/bin/env python3
"""
Unitree G1 English voice assistant.

Pipeline
    G1 mic (UDP multicast)
      -> VAD utterance segmentation (no fixed 4 s chunks)
      -> Whisper (English only)
      -> Ollama chat, streamed sentence by sentence
      -> Edge TTS -> ffmpeg (in memory) -> 16 kHz mono PCM
      -> G1 speaker (AudioClient.PlayStream), paced against a play clock

The LLM, TTS and playback stages run concurrently, so the robot starts
speaking after the FIRST sentence is ready instead of after the whole reply.
Short questions get short answers; detailed questions and stories get as many
sentences as they need - there is no fixed sentence or word limit.

Requirements
    pip install numpy requests openai-whisper edge-tts
    (optional) pip install webrtcvad   # better voice activity detection
    ffmpeg on PATH, Ollama running with the model pulled.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import math
import queue
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np
import requests

log = logging.getLogger("g1voice")


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class Config:
    # --- network / microphone -------------------------------------------
    multicast_group: str = "239.168.123.161"
    port: int = 5555
    interface_ip: str = "192.168.123.40"
    network_interface: str = "eno1"
    sample_rate: int = 16000  # G1 mic: 16 kHz, mono, s16le

    # --- voice activity detection ---------------------------------------
    frame_ms: int = 30  # 10/20/30 ms (webrtcvad limits)
    min_rms: float = 350.0  # absolute loudness floor for speech
    noise_factor: float = 3.0  # speech must be N x the noise floor
    start_frames: int = 3  # consecutive voiced frames to start
    preroll_ms: int = 300  # audio kept from before speech start
    silence_end_ms: int = 700  # silence that ends an utterance
    min_speech_ms: int = 350  # shorter blips are discarded
    max_utterance_s: float = 15.0  # hard cap per utterance
    vad_aggressiveness: int = 2  # webrtcvad 0-3 (only if installed)

    # --- speech recognition ---------------------------------------------
    whisper_model: str = "small.en"  # English-only model: faster and more accurate
    min_avg_logprob: float = -1.2
    max_compression_ratio: float = 2.4
    no_speech_threshold: float = 0.6

    # --- language model -------------------------------------------------
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:8b"
    llm_connect_timeout: float = 5.0
    llm_read_timeout: float = 60.0
    llm_temperature: float = 0.6
    llm_num_predict: int = 3000  # safety ceiling only; long stories fit easily
    llm_num_ctx: int = 8192  # room for long answers plus history
    keep_alive: str = "30m"
    history_turns: int = 4  # remembered user/assistant pairs
    history_reply_chars: int = 1500  # long replies are trimmed in memory

    # --- text to speech -------------------------------------------------
    voice: str = "en-IN-PrabhatNeural"  # any English Edge TTS voice
    tts_rate: str = "+0%"  # e.g. "+10%" for faster speech
    tts_attempts: int = 3
    tts_timeout: float = 20.0

    # --- G1 playback ----------------------------------------------------
    g1_timeout: float = 10.0
    volume: Optional[int] = None  # 0-100, None = leave unchanged
    play_chunk_seconds: float = 1.0
    play_lead_seconds: float = 1.5  # how far ahead of the speaker we run
    pre_silence_ms: int = 120  # avoids a clipped first syllable
    echo_tail_seconds: float = 0.6  # mute mic a bit after speaking
    use_led: bool = True

    # --- behaviour ------------------------------------------------------
    wake_words: tuple = ()  # e.g. ("robot", "g1", "रोबोट"); empty = always on
    wake_window_s: float = 20.0  # stay awake this long after a wake word


# ============================================================
# TEXT HELPERS
# ============================================================

# Whisper hallucinates these on near-silence.
_HALLUCINATION = re.compile(
    r"(thanks? for watching|subtitles? by|amara\.org|www\.|"
    r"please subscribe|like and subscribe)",
    re.IGNORECASE,
)
_HALLUCINATION_EXACT = {"you", ".", "..", "...", "bye."}


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_MARKDOWN = re.compile(r"[*_`#>~|]+")
_URL = re.compile(r"https?://\S+")
_STAGE_DIRECTION = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_EMOJI = re.compile("[\U0001f000-\U0001faff\u2600-\u27bf\ufe0f\u200d]")


def clean_for_speech(text: str) -> str:
    text = _THINK_BLOCK.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _STAGE_DIRECTION.sub(" ", text)
    text = _EMOJI.sub("", text)
    text = _MARKDOWN.sub("", text)
    return " ".join(text.split())


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


class SentenceSplitter:
    """Turns a token stream into complete sentences as soon as they exist."""

    MAX_UNBROKEN_CHARS = 140

    def __init__(self) -> None:
        self.buffer = ""

    def feed(self, chunk: str) -> list[str]:
        self.buffer += chunk
        self.buffer = _THINK_BLOCK.sub("", self.buffer)
        if "<think>" in self.buffer:  # unfinished reasoning block
            return []
        parts = _SENTENCE_END.split(self.buffer)
        complete, self.buffer = parts[:-1], parts[-1]

        # Very long run without punctuation: break at the last comma.
        if len(self.buffer) > self.MAX_UNBROKEN_CHARS:
            cut = self.buffer.rfind(", ")
            if cut > 40:
                complete.append(self.buffer[: cut + 1])
                self.buffer = self.buffer[cut + 2 :]
        return [s.strip() for s in complete if s.strip()]

    def flush(self) -> list[str]:
        rest = _THINK_BLOCK.sub("", self.buffer).strip()
        self.buffer = ""
        return [rest] if rest and "<think>" not in rest else []


# ============================================================
# VOICE ACTIVITY DETECTION / UTTERANCE SEGMENTATION
# ============================================================


class UtteranceSegmenter:
    """
    Feeds on fixed-size int16 frames and emits complete utterances
    (pre-roll + speech + short trailing silence) as int16 arrays.
    Uses an adaptive noise floor, plus webrtcvad when it is installed.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.frame_len = cfg.sample_rate * cfg.frame_ms // 1000
        self.noise = cfg.min_rms / cfg.noise_factor
        self.preroll: collections.deque = collections.deque(
            maxlen=math.ceil(cfg.preroll_ms / cfg.frame_ms) + cfg.start_frames
        )
        self.vad = None
        try:
            import webrtcvad  # type: ignore

            self.vad = webrtcvad.Vad(cfg.vad_aggressiveness)
        except Exception:
            pass
        self.reset()

    def reset(self) -> None:
        self.preroll.clear()
        self.in_speech = False
        self.voiced_run = 0
        self.speech_frames = 0
        self.silence_frames = 0
        self.frames: list[np.ndarray] = []

    @property
    def threshold(self) -> float:
        return max(self.cfg.min_rms, self.noise * self.cfg.noise_factor)

    def _voiced(self, frame: np.ndarray, rms: float) -> bool:
        if rms <= self.threshold:
            return False
        if self.vad is not None:
            try:
                return bool(self.vad.is_speech(frame.tobytes(), self.cfg.sample_rate))
            except Exception:
                return True
        return True

    def push(self, frame: np.ndarray) -> Optional[np.ndarray]:
        cfg = self.cfg
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        voiced = self._voiced(frame, rms)

        if not self.in_speech:
            self.preroll.append(frame)
            if voiced:
                self.voiced_run += 1
            else:
                self.voiced_run = 0
                self.noise = 0.95 * self.noise + 0.05 * rms  # track background
            if self.voiced_run >= cfg.start_frames:
                self.in_speech = True
                self.frames = list(self.preroll)
                self.speech_frames = self.voiced_run
                self.silence_frames = 0
            return None

        self.frames.append(frame)
        if voiced:
            self.speech_frames += 1
            self.silence_frames = 0
        else:
            self.silence_frames += 1

        ended = self.silence_frames * cfg.frame_ms >= cfg.silence_end_ms
        too_long = len(self.frames) * cfg.frame_ms >= cfg.max_utterance_s * 1000
        if not (ended or too_long):
            return None

        utterance = np.concatenate(self.frames)
        long_enough = self.speech_frames * cfg.frame_ms >= cfg.min_speech_ms
        self.reset()
        return utterance if long_enough else None


class MicListener(threading.Thread):
    """
    Dedicated thread: keeps reading the multicast socket so the kernel buffer
    never overflows, segments speech, and discards everything while muted
    (so the robot never hears itself).
    """

    def __init__(self, cfg: Config, out: "queue.Queue[np.ndarray]") -> None:
        super().__init__(name="mic-listener", daemon=True)
        self.cfg = cfg
        self.out = out
        self.segmenter = UtteranceSegmenter(cfg)
        self.packets = 0
        self.last_packet = 0.0
        self._stop_evt = threading.Event()
        self._muted = threading.Event()
        self.sock = self._open_socket()

    def _open_socket(self) -> socket.socket:
        cfg = self.cfg
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        except OSError:
            pass
        sock.bind(("", cfg.port))
        membership = socket.inet_aton(cfg.multicast_group) + socket.inet_aton(
            cfg.interface_ip
        )
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
        sock.settimeout(0.5)
        return sock

    @property
    def is_muted(self) -> bool:
        return self._muted.is_set()

    def mute(self) -> None:
        self._muted.set()

    def unmute(self) -> None:
        self._muted.clear()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        frame_bytes = self.segmenter.frame_len * 2
        buf = bytearray()
        while not self._stop_evt.is_set():
            try:
                packet, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            if not packet:
                continue
            self.packets += 1
            self.last_packet = time.monotonic()

            if self._muted.is_set():
                buf.clear()
                self.segmenter.reset()
                continue

            buf.extend(packet)
            while len(buf) >= frame_bytes:
                frame = np.frombuffer(bytes(buf[:frame_bytes]), dtype="<i2")
                del buf[:frame_bytes]
                utterance = self.segmenter.push(frame)
                if utterance is not None:
                    self.out.put(utterance)
        try:
            self.sock.close()
        except OSError:
            pass


# ============================================================
# SPEECH RECOGNITION
# ============================================================


class Transcriber:
    def __init__(self, cfg: Config) -> None:
        import torch
        import whisper

        self.cfg = cfg
        self.whisper = whisper
        self.fp16 = torch.cuda.is_available()
        log.info("Loading Whisper '%s' (fp16=%s)...", cfg.whisper_model, self.fp16)
        self.model = whisper.load_model(cfg.whisper_model)

    def warm_up(self) -> None:
        try:
            self.transcribe(np.zeros(self.cfg.sample_rate, dtype="<i2"))
        except Exception as error:
            log.debug("Whisper warm-up failed: %s", error)

    def transcribe(self, samples: np.ndarray) -> str:
        """Return the English transcript, or '' if nothing reliable was heard."""
        cfg = self.cfg
        audio = samples.astype(np.float32) / 32768.0

        result = self.model.transcribe(
            audio,
            language="en",  # English only
            task="transcribe",
            fp16=self.fp16,
            temperature=(0.0, 0.2, 0.4),
            condition_on_previous_text=False,
            no_speech_threshold=cfg.no_speech_threshold,
            logprob_threshold=cfg.min_avg_logprob,
            compression_ratio_threshold=cfg.max_compression_ratio,
            verbose=None,
        )

        good = [
            seg
            for seg in result.get("segments", [])
            if seg.get("no_speech_prob", 1.0) < cfg.no_speech_threshold
            and seg.get("avg_logprob", -10.0) > cfg.min_avg_logprob
            and seg.get("compression_ratio", 0.0) < cfg.max_compression_ratio
        ]
        # Only text from reliable segments is used.
        text = " ".join(seg["text"].strip() for seg in good).strip()

        if not text or not re.search(r"\w", text):
            return ""
        if text.lower() in _HALLUCINATION_EXACT or _HALLUCINATION.search(text):
            log.debug("Dropped likely hallucination: %r", text)
            return ""
        return text


# ============================================================
# LANGUAGE MODEL (Ollama /api/chat, streamed)
# ============================================================


def build_system_prompt() -> str:
    return (
        "You are a friendly, warm Unitree G1 humanoid robot assistant. Your reply "
        "is spoken aloud by a speech synthesizer, so write plain flowing sentences: "
        "no emojis, markdown, bullet points, numbered lists, headings or stage "
        "directions, and spell out symbols and numbers when needed. "
        "Match the length of your answer to the question. "
        "For a simple or direct question, answer directly in one or two short "
        "sentences. "
        "If the person asks for detail, an explanation, a comparison, a lesson or "
        "'how' and 'why' something works, give a thorough, well organised answer "
        "that covers everything important, using as many sentences as it takes. "
        "If the person asks for a story, tell the complete story from beginning to "
        "end with characters, a setting, a problem, a climax and a satisfying "
        "ending, using as many sentences as the story needs, and never stop "
        "halfway or cut it short. "
        "You can currently only talk - never claim to have moved, picked up, or "
        "done anything physical. The person's words come from speech recognition "
        "and may contain mistakes: interpret them generously, and politely ask "
        "them to repeat if a message is truly unintelligible. "
        "Always reply in natural, conversational English only. "
        "/no_think"
    )


class OllamaChat:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.session = requests.Session()
        self.history: collections.deque = collections.deque(
            maxlen=cfg.history_turns * 2
        )

    def check(self) -> None:
        resp = self.session.get(f"{self.cfg.ollama_url}/api/tags", timeout=5)
        resp.raise_for_status()
        names = {m.get("name", "") for m in resp.json().get("models", [])}
        wanted = self.cfg.llm_model
        if wanted not in names and f"{wanted}:latest" not in names:
            raise RuntimeError(
                f"Ollama model '{wanted}' not found. Run: ollama pull {wanted}"
            )

    def warm_up(self) -> None:
        """Load the model into memory now, not on the first question."""
        try:
            self.session.post(
                f"{self.cfg.ollama_url}/api/chat",
                json={
                    "model": self.cfg.llm_model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": False,
                    "think": False,
                    "keep_alive": self.cfg.keep_alive,
                    "options": {"num_predict": 1},
                },
                timeout=120,
            )
        except requests.RequestException as error:
            log.debug("Ollama warm-up failed: %s", error)

    def stream_sentences(self, text: str) -> Iterator[str]:
        """Yield the reply one clean sentence at a time, with no length cap."""
        cfg = self.cfg
        messages = [{"role": "system", "content": build_system_prompt()}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": text})

        payload = {
            "model": cfg.llm_model,
            "messages": messages,
            "stream": True,
            "think": False,
            "keep_alive": cfg.keep_alive,
            "options": {
                "temperature": cfg.llm_temperature,
                "num_predict": cfg.llm_num_predict,
                "num_ctx": cfg.llm_num_ctx,
            },
        }

        splitter = SentenceSplitter()

        with self.session.post(
            f"{cfg.ollama_url}/api/chat",
            json=payload,
            stream=True,
            timeout=(cfg.llm_connect_timeout, cfg.llm_read_timeout),
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                chunk = data.get("message", {}).get("content", "")
                for raw in splitter.feed(chunk):
                    sentence = clean_for_speech(raw)
                    if re.search(r"\w", sentence):
                        yield sentence
                if data.get("done"):
                    break
            for raw in splitter.flush():
                sentence = clean_for_speech(raw)
                if re.search(r"\w", sentence):
                    yield sentence

    def remember(self, user_text: str, reply: str) -> None:
        limit = self.cfg.history_reply_chars
        if len(reply) > limit:
            reply = reply[:limit].rsplit(" ", 1)[0] + " ..."
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})


# ============================================================
# TEXT TO SPEECH (Edge TTS -> 16 kHz mono PCM, all in memory)
# ============================================================


class EdgeSynth:
    """Used from a single worker thread; keeps one event loop alive."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def _fetch_mp3(self, text: str, voice: str) -> bytes:
        import edge_tts

        communicate = edge_tts.Communicate(text, voice, rate=self.cfg.tts_rate)
        audio = bytearray()
        async for item in communicate.stream():
            if item["type"] == "audio":
                audio += item["data"]
        return bytes(audio)

    def synthesize(self, text: str) -> bytes:
        cfg = self.cfg
        voice = cfg.voice
        if self._loop is None:
            self._loop = asyncio.new_event_loop()

        last_error: Optional[Exception] = None
        for attempt in range(1, cfg.tts_attempts + 1):
            try:
                mp3 = self._loop.run_until_complete(
                    asyncio.wait_for(self._fetch_mp3(text, voice), cfg.tts_timeout)
                )
                if not mp3:
                    raise RuntimeError("Edge TTS returned no audio")
                return self._to_pcm(mp3)
            except Exception as error:
                last_error = error
                log.warning(
                    "Edge TTS attempt %d/%d failed: %s",
                    attempt,
                    cfg.tts_attempts,
                    error,
                )
                if attempt < cfg.tts_attempts:
                    time.sleep(0.5 * attempt)
        raise RuntimeError(f"Edge TTS failed: {last_error}")

    def _to_pcm(self, mp3: bytes) -> bytes:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-f",
                "s16le",
                "-ar",
                str(self.cfg.sample_rate),
                "-ac",
                "1",
                "pipe:1",
            ],
            input=mp3,
            capture_output=True,
            timeout=30,
            check=True,
        )
        if not proc.stdout:
            raise RuntimeError("ffmpeg produced no audio")
        return proc.stdout[: len(proc.stdout) // 2 * 2]


# ============================================================
# G1 SPEAKER
# ============================================================

LED_COLORS = {
    "idle": (0, 0, 0),
    "thinking": (0, 0, 255),
    "speaking": (0, 255, 0),
    "error": (255, 0, 0),
}


class G1Speaker:
    APP_NAME = "g1_voice"

    def __init__(self, cfg: Config) -> None:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

        self.cfg = cfg
        log.info("Initializing Unitree communication on %s...", cfg.network_interface)
        ChannelFactoryInitialize(0, cfg.network_interface)
        self.client = AudioClient()
        self.client.SetTimeout(cfg.g1_timeout)
        self.client.Init()

        if not hasattr(self.client, "PlayStream"):
            raise RuntimeError(
                "Installed AudioClient has no PlayStream method; check your SDK version."
            )

        if cfg.volume is not None:
            try:
                self.client.SetVolume(int(cfg.volume))
            except Exception as error:
                log.warning("Could not set volume: %s", error)
        try:
            log.info("G1 volume: %s", self.client.GetVolume())
        except Exception as error:
            log.warning("Could not read G1 volume: %s", error)

        self.bytes_per_second = cfg.sample_rate * 2
        self.stream_id = uuid.uuid4().hex
        self._audio_end = 0.0
        self._first_chunk = True
        self.abort = threading.Event()  # set to cut speech short

    # ---- LED -----------------------------------------------------------
    def set_led(self, state: str) -> None:
        if not self.cfg.use_led or not hasattr(self.client, "LedControl"):
            return
        try:
            self.client.LedControl(*LED_COLORS[state])
        except Exception:
            pass

    # ---- streaming playback --------------------------------------------
    def start_reply(self) -> None:
        self.stream_id = uuid.uuid4().hex  # fresh stream per reply
        self._audio_end = 0.0
        self._first_chunk = True
        self.abort.clear()

    def play(self, pcm: bytes) -> None:
        """Send PCM and stay at most `play_lead_seconds` ahead of real time."""
        cfg = self.cfg
        if self._first_chunk:
            pcm = (
                b"\x00" * (cfg.sample_rate * 2 * cfg.pre_silence_ms // 1000 // 2 * 2)
                + pcm
            )
            self._first_chunk = False

        chunk_bytes = max(
            2, int(self.bytes_per_second * cfg.play_chunk_seconds) // 2 * 2
        )
        for offset in range(0, len(pcm), chunk_bytes):
            if self.abort.is_set():
                return
            chunk = pcm[offset : offset + chunk_bytes]
            result = self.client.PlayStream(self.APP_NAME, self.stream_id, chunk)
            code = result[0] if isinstance(result, tuple) else result
            if code not in (None, 0):
                raise RuntimeError(f"PlayStream returned error code {code}")

            now = time.monotonic()
            self._audio_end = (
                max(self._audio_end, now) + len(chunk) / self.bytes_per_second
            )
            wait = self._audio_end - now - cfg.play_lead_seconds
            if wait > 0:
                self.abort.wait(wait)  # interruptible sleep

    def finish(self) -> None:
        """Wait for queued audio to be played, then close the stream."""
        remaining = self._audio_end - time.monotonic()
        if remaining > 0:
            self.abort.wait(remaining)
        self.stop()

    def stop(self) -> None:
        if hasattr(self.client, "PlayStop"):
            try:
                self.client.PlayStop(self.APP_NAME)
            except Exception as error:
                log.debug("PlayStop warning: %s", error)

    def say_builtin(self, text: str) -> bool:
        """Offline English fallback using the robot's own TTS."""
        if not hasattr(self.client, "TtsMaker"):
            return False
        try:
            self.client.TtsMaker(text, 1)  # speaker id 1 = English
            self.abort.wait(max(1.0, 0.4 * len(text.split())))
            return True
        except Exception as error:
            log.warning("Built-in TTS failed: %s", error)
            return False


# ============================================================
# ASSISTANT
# ============================================================

_DONE = object()

ERROR_PHRASE = "Sorry, I'm having trouble thinking right now. Please try again."


class VoiceAssistant:
    """
    Orchestrates listen -> transcribe -> think -> speak.

    `on_event` (optional) receives plain dicts, used by the web UI:
        {"type": "state", "state": <idle|loading|listening|transcribing|
                                    thinking|speaking|stopping|error>, "detail": str}
        {"type": "user",  "turn": n, "text": str}
        {"type": "robot", "turn": n, "text": <one sentence>, "error": bool}
        {"type": "robot_done", "turn": n}

    Models are loaded once (load) and kept in memory, so a session can be
    stopped and started again quickly.
    """

    def __init__(
        self, cfg: Config, on_event: Optional[Callable[[dict], None]] = None
    ) -> None:
        self.cfg = cfg
        self.on_event = on_event
        self.utterances: "queue.Queue[np.ndarray]" = queue.Queue()
        self.awake_until = 0.0
        self.turn = 0
        self.listener: Optional[MicListener] = None
        self.speaker: Optional[G1Speaker] = None
        self.llm: Optional[OllamaChat] = None
        self.stt: Optional[Transcriber] = None
        self.tts: Optional[EdgeSynth] = None
        self._loaded = False
        self._stop = threading.Event()

    # ---- events / control ----------------------------------------------
    def emit(self, kind: str, **data) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event({"type": kind, **data})
        except Exception as error:
            log.debug("Event callback failed: %s", error)

    def set_state(self, state: str, detail: str = "") -> None:
        if self._stop.is_set() and state not in ("idle", "error"):
            return  # keep showing "stopping"
        self.emit("state", state=state, detail=detail)

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def reset_stop(self) -> None:
        self._stop.clear()
        if self.speaker is not None:
            self.speaker.abort.clear()

    def request_stop(self) -> None:
        """Ask the running session to finish; cuts any speech short."""
        self._stop.set()
        if self.speaker is not None:
            self.speaker.abort.set()
            self.speaker.stop()

    # ---- startup -------------------------------------------------------
    def load(self) -> None:
        """Heavy one-time initialisation (idempotent)."""
        if self._loaded:
            return
        cfg = self.cfg

        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "ffmpeg not found on PATH (needed to convert TTS audio)."
            )

        self.set_state("loading", "Checking Ollama...")
        self.llm = OllamaChat(cfg)
        self.llm.check()
        log.info("Ollama OK, model '%s' available.", cfg.llm_model)

        try:
            socket.create_connection(
                ("speech.platform.bing.com", 443), timeout=3
            ).close()
        except OSError:
            log.warning(
                "Edge TTS server unreachable (no internet?). "
                "Only the robot's built-in English voice will work."
            )

        self.set_state("loading", "Loading speech recognition...")
        self.stt = Transcriber(cfg)
        self.tts = EdgeSynth(cfg)

        self.set_state("loading", "Connecting to the G1...")
        self.speaker = G1Speaker(cfg)

        self.set_state("loading", "Warming up models...")
        threading.Thread(target=self.llm.warm_up, daemon=True).start()
        self.stt.warm_up()
        self._loaded = True

    def begin_listening(self) -> None:
        cfg = self.cfg
        self._flush_utterances()
        self.listener = MicListener(cfg, self.utterances)
        self.listener.start()
        self.speaker.set_led("idle")

        log.info("=" * 55)
        log.info(" UNITREE G1 VOICE ASSISTANT (English)")
        log.info("=" * 55)
        log.info(
            "Mic %s:%d via %s | LLM %s | Whisper %s",
            cfg.multicast_group,
            cfg.port,
            cfg.network_interface,
            cfg.llm_model,
            cfg.whisper_model,
        )
        if cfg.wake_words:
            log.info("Wake words: %s", ", ".join(cfg.wake_words))
        log.info("Listening...")

    def end_listening(self) -> None:
        listener, self.listener = self.listener, None
        if listener is not None:
            listener.stop()
            listener.join(timeout=3)  # releases the multicast socket
        if self.speaker is not None:
            self.speaker.stop()
            self.speaker.set_led("idle")

    def start(self) -> None:
        """Convenience for the command-line entry point."""
        self.load()
        self.begin_listening()

    def shutdown(self) -> None:
        self.request_stop()
        self.end_listening()

    # ---- main loop -----------------------------------------------------
    def _flush_utterances(self) -> None:
        while True:
            try:
                self.utterances.get_nowait()
            except queue.Empty:
                return

    def run(self) -> None:
        """Blocks until request_stop() is called."""
        started = time.monotonic()
        last_warn = 0.0
        while not self._stop.is_set():
            try:
                samples = self.utterances.get(timeout=0.5)
            except queue.Empty:
                if self.listener is None or not self.listener.is_alive():
                    raise RuntimeError("Microphone listener stopped unexpectedly.")
                now = time.monotonic()
                silent_for = now - (self.listener.last_packet or started)
                if silent_for > 5 and now - last_warn > 30:
                    log.warning(
                        "No microphone packets for %.0f s - check the network "
                        "interface, multicast route and interface IP.",
                        silent_for,
                    )
                    last_warn = now
                continue

            if self._stop.is_set():
                break
            self.set_state("transcribing")
            try:
                self.handle(samples)
            except Exception as error:
                log.exception("Unhandled error while handling an utterance: %s", error)
                self.speaker.set_led("error")
            finally:
                self._resume_listening()

    def _resume_listening(self) -> None:
        # Only after a spoken turn: drop what was captured meanwhile (our own
        # voice), wait out the echo tail, then listen again.
        if self.listener is not None and self.listener.is_muted:
            self._flush_utterances()
            self._stop.wait(self.cfg.echo_tail_seconds)
            self.listener.unmute()
        if self.speaker is not None:
            self.speaker.set_led("idle")
        self.set_state("listening")

    # ---- one conversational turn ---------------------------------------
    def _wake_ok(self, text: str) -> bool:
        cfg = self.cfg
        if not cfg.wake_words:
            return True
        now = time.monotonic()
        lowered = text.lower()
        if (
            any(word.lower() in lowered for word in cfg.wake_words)
            or now < self.awake_until
        ):
            self.awake_until = now + cfg.wake_window_s
            return True
        log.debug("Ignored (no wake word): %s", text)
        return False

    def handle(self, samples: np.ndarray) -> None:
        t0 = time.monotonic()
        text = self.stt.transcribe(samples)
        t_stt = time.monotonic() - t0
        if not text or not self._wake_ok(text):
            return

        log.info("You said (STT %.1fs): %s", t_stt, text)

        self.turn += 1
        self.emit("user", turn=self.turn, text=text)

        self.listener.mute()  # robot must not hear itself
        self.set_state("thinking")
        self.speaker.set_led("thinking")
        self.respond(text, t0)

    def respond(self, text: str, t_start: float) -> None:
        turn = self.turn
        sentences: "queue.Queue" = queue.Queue()
        audio: "queue.Queue" = queue.Queue(maxsize=3)
        stop = threading.Event()
        reply_parts: list[str] = []
        llm_failed = threading.Event()
        play_failed = False

        def put_audio(item) -> None:
            while not stop.is_set():
                try:
                    audio.put(item, timeout=0.2)
                    return
                except queue.Full:
                    continue

        def llm_worker() -> None:
            try:
                stream = self.llm.stream_sentences(text)
                try:
                    for sentence in stream:
                        if stop.is_set():
                            break
                        reply_parts.append(sentence)
                        sentences.put(sentence)
                finally:
                    stream.close()  # aborts generation if we stopped early
            except Exception as error:
                llm_failed.set()
                log.error("LLM error: %s", error)
            finally:
                sentences.put(_DONE)

        def tts_worker() -> None:
            while True:
                sentence = sentences.get()
                if sentence is _DONE:
                    put_audio(_DONE)
                    return
                if stop.is_set():
                    continue
                try:
                    pcm = self.tts.synthesize(sentence)
                except Exception as error:
                    log.error("TTS error: %s", error)
                    pcm = None
                put_audio((sentence, pcm))

        threads = [
            threading.Thread(target=llm_worker, name="llm", daemon=True),
            threading.Thread(target=tts_worker, name="tts", daemon=True),
        ]
        for thread in threads:
            thread.start()

        spoke = False
        self.speaker.start_reply()
        while True:
            if self._stop.is_set():  # Stop button pressed
                stop.set()
                self.speaker.stop()
                break
            try:
                item = audio.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is _DONE:
                break

            sentence, pcm = item
            try:
                if pcm:
                    if not spoke:
                        log.info(
                            "First audio ready after %.1fs", time.monotonic() - t_start
                        )
                        self.set_state("speaking")
                        self.speaker.set_led("speaking")
                    self.emit(
                        "robot", turn=turn, text=sentence
                    )  # shown as it is spoken
                    self.speaker.play(pcm)
                    spoke = True
                    log.info("G1 says: %s", sentence)
                elif self.speaker.say_builtin(sentence):
                    self.emit("robot", turn=turn, text=sentence)
                    spoke = True
                    log.info("G1 says (built-in voice): %s", sentence)
            except Exception as error:
                log.error("Playback error: %s", error)
                play_failed = True
                stop.set()
                self.speaker.stop()
                break

        for thread in threads:
            thread.join(timeout=0.5 if stop.is_set() else 5)

        if spoke and not stop.is_set():
            self.speaker.finish()
        self.emit("robot_done", turn=turn)

        if self._stop.is_set():
            return
        if reply_parts and not llm_failed.is_set() and not play_failed:
            self.llm.remember(text, " ".join(reply_parts))
        elif not spoke and not play_failed:
            self._say_error(turn)

    def _say_error(self, turn: int) -> None:
        if self._stop.is_set():
            return
        self.speaker.set_led("error")
        self.emit("robot", turn=turn, text=ERROR_PHRASE, error=True)
        try:
            pcm = self.tts.synthesize(ERROR_PHRASE)
            self.speaker.start_reply()
            self.speaker.play(pcm)
            self.speaker.finish()
        except Exception:
            self.speaker.say_builtin(ERROR_PHRASE)
        finally:
            self.emit("robot_done", turn=turn)


# ============================================================
# ENTRY POINT
# ============================================================


def build_parser(cfg: Optional[Config] = None) -> argparse.ArgumentParser:
    cfg = cfg or Config()
    p = argparse.ArgumentParser(description="Unitree G1 English voice assistant")
    p.add_argument(
        "--interface", help=f"network interface (default {cfg.network_interface})"
    )
    p.add_argument(
        "--interface-ip",
        help=f"local IP on that interface (default {cfg.interface_ip})",
    )
    p.add_argument(
        "--whisper-model", help=f"tiny.en/base.en/small.en/medium.en (default {cfg.whisper_model})"
    )
    p.add_argument("--llm-model", help=f"Ollama model (default {cfg.llm_model})")
    p.add_argument("--voice", help=f"Edge TTS English voice (default {cfg.voice})")
    p.add_argument(
        "--wake-word", action="append", help="repeatable; enables wake-word gating"
    )
    p.add_argument("--volume", type=int, help="G1 speaker volume 0-100")
    p.add_argument("--no-led", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def apply_args(cfg: Config, args: argparse.Namespace) -> None:
    if args.interface:
        cfg.network_interface = args.interface
    if args.interface_ip:
        cfg.interface_ip = args.interface_ip
    if args.whisper_model:
        cfg.whisper_model = args.whisper_model
    if args.llm_model:
        cfg.llm_model = args.llm_model
    if args.voice:
        cfg.voice = args.voice
    if args.wake_word:
        cfg.wake_words = tuple(args.wake_word)
    if args.volume is not None:
        cfg.volume = args.volume
    if args.no_led:
        cfg.use_led = False


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> Config:
    cfg = Config()
    args = build_parser(cfg).parse_args()
    apply_args(cfg, args)
    setup_logging(args.verbose)
    return cfg


def _raise_keyboard_interrupt(*_args) -> None:
    raise KeyboardInterrupt


def main() -> None:
    cfg = parse_args()
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    assistant = VoiceAssistant(cfg)
    try:
        assistant.start()
        assistant.run()
    except KeyboardInterrupt:
        log.info("Stopping assistant...")
    except Exception as error:
        log.exception("Fatal error: %s", error)
    finally:
        assistant.shutdown()
        log.info("Assistant stopped.")


if __name__ == "__main__":
    main()