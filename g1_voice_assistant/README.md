# G1 Voice Assistant

Real-time voice conversation system for the **Unitree G1 humanoid robot**.

```
🎤 G1 Mic → Unitree STT → /audio_msg → Groq LLM → Piper TTS → 🔊 G1 Speaker
```

---

## Quick Start

### On the G1

```bash
# 1. Source ROS 2
source /opt/ros/foxy/setup.bash

# 2. Enter project directory
cd ~/g1_voice_assistant

# 3. Set up environment
cp .env.example .env
nano .env   # fill in GROQ_API_KEY and ALSA_OUTPUT_DEVICE

# 4. Install Python deps
pip3 install -r requirements.txt

# 5. Test each stage before running the full pipeline
python3 stage1_listener.py      # Stage 1: verify /audio_msg arrives
python3 stage2_groq_test.py     # Stage 2: verify Groq responds
python3 stage3_tts_test.py      # Stage 3: verify TTS synthesises audio
python3 stage4_speaker_test.py  # Stage 4: verify speaker plays audio

# 6. Run the full pipeline
python3 main.py
```

---

## Architecture

```
/audio_msg (ROS 2)
      │
      ▼
 G1Listener          ← filters noise, deduplicates by index
      │ artifact: 01_stt/raw_XXXXXX.json
      │           01_stt/transcript_XXXXXX.txt
      ▼
GroqClient           ← Groq LLM API (internet required)
      │ artifact: 02_llm/request_XXXXXX.json
      │           02_llm/response_XXXXXX.json
      ▼
TTSEngine            ← Piper (offline) or espeak-ng fallback
      │ artifact: 03_tts/metadata_XXXXXX.json
      │           output/response_XXXXXX.wav
      ▼
G1Speaker            ← aplay (ALSA) or sounddevice
      │ artifact: 04_speaker/playback_XXXXXX.json
      ▼
🔊 G1 Speaker
```

Every stage fails **independently** — if Groq fails, the STT artifact is still saved.

---

## Configuration (`.env`)

```env
# Required
GROQ_API_KEY=gsk_...

# Groq model (free tier)
GROQ_MODEL=llama-3.1-8b-instant

# Assistant persona
SYSTEM_PROMPT=You are G1, a helpful voice assistant built into a Unitree G1 humanoid robot. Keep answers concise and conversational. No markdown, no bullet points, plain spoken sentences only.

# TTS: piper (preferred) or espeak (fallback)
TTS_BACKEND=piper
PIPER_MODEL=en_US-ryan-high
PIPER_MODELS_DIR=tts/models
PIPER_BINARY=piper

# Speaker: alsa (for G1) or local (for dev laptop)
SPEAKER_BACKEND=alsa
ALSA_OUTPUT_DEVICE=hw:1,0    # find with: aplay -l on the G1

# ROS topic
ROS_AUDIO_TOPIC=/audio_msg

# Conversation history
MAX_HISTORY_MESSAGES=10
```

---

## Piper TTS Setup on the G1

Piper is a fast, offline, neural TTS that runs on the Jetson's CPU.

```bash
# 1. Download the ARM64 binary
cd ~
wget https://github.com/rhasspy/piper/releases/latest/download/piper_linux_aarch64.tar.gz
tar -xzf piper_linux_aarch64.tar.gz
sudo cp piper/piper /usr/local/bin/
piper --version   # verify

# 2. Download voice model (en_US-ryan-high = clear male voice)
mkdir -p ~/g1_voice_assistant/tts/models
cd ~/g1_voice_assistant/tts/models

# Model file
wget "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx"

# Config file (required)
wget "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx.json"

# 3. Test synthesis
echo "Hello, I am G1." | piper --model tts/models/en_US-ryan-high.onnx --output_file /tmp/test.wav
aplay /tmp/test.wav
```

---

## Finding the Correct ALSA Device on the G1

```bash
# List all playback devices
aplay -l

# The G1 uses the NVIDIA APE card. Try each device:
aplay -D hw:1,0 /tmp/test.wav
aplay -D hw:1,1 /tmp/test.wav
# ... until you hear audio from the robot's speaker

# Set the working device in .env:
# ALSA_OUTPUT_DEVICE=hw:1,<N>
```

---

## Stage Testing

Run each stage in isolation to verify before the full pipeline:

| Script | Tests | Network? | ROS? |
|--------|-------|----------|------|
| `stage1_listener.py` | /audio_msg → artifacts | No | **Yes** |
| `stage2_groq_test.py` | Groq 3-turn conversation | **Yes** | No |
| `stage3_tts_test.py` | TTS → WAV files | No | No |
| `stage4_speaker_test.py` | WAV → ALSA playback | No | No |
| `main.py` | Full pipeline | **Yes** | **Yes** |

---

## Artifact Structure

Every run creates a timestamped session directory:

```
conversations/
└── 2026-08-08_12-30-15/
    ├── session.json            ← session metadata
    ├── 01_stt/
    │   ├── raw_000001.json     ← full /audio_msg payload (never modified)
    │   └── transcript_000001.txt
    ├── 02_llm/
    │   ├── request_000001.json ← prompt + history (no API key)
    │   └── response_000001.json ← response + latency + token usage
    ├── 03_tts/
    │   └── metadata_000001.json
    └── 04_speaker/
        └── playback_000001.json
```

---

## Running Tests (no ROS or API key needed)

```bash
cd g1_voice_assistant
python3 -m pytest tests/ -v
# Expected: 43 passed
```

---

## Deploying to G1

```bash
# From the dev machine (HP Victus)
rsync -avz --exclude='.env' --exclude='conversations/' --exclude='__pycache__' \
    g1_voice_assistant/ unitree@192.168.123.164:~/g1_voice_assistant/

# Then on the G1, set up .env separately:
ssh unitree@192.168.123.164
cp ~/g1_voice_assistant/.env.example ~/g1_voice_assistant/.env
nano ~/g1_voice_assistant/.env  # add GROQ_API_KEY and ALSA_OUTPUT_DEVICE
```

---

## Safety

This project is **voice conversation only**. It does NOT:
- Publish to `/lowcmd` or any motor-control topic
- Modify robot locomotion, balance, or arm control
- Change ROS_DOMAIN_ID or network configuration
- Interfere with existing Unitree services

---

## Hardware

| Component | Value |
|-----------|-------|
| Robot | Unitree G1 |
| Compute | NVIDIA Jetson Orin NX |
| Architecture | aarch64 |
| OS | Ubuntu 20.04.6 LTS |
| ROS | Foxy |
| Python | 3.8.10 |
| STT | Unitree built-in (`/audio_msg`) |
| LLM | Groq API (`llama-3.1-8b-instant`) |
| TTS | Piper (offline, CPU) |
| Speaker | ALSA / aplay |

---

## Troubleshooting

**No transcripts received:**
```bash
ros2 topic echo /audio_msg   # verify publisher is active
ros2 topic info /audio_msg   # check publisher count
```

**Groq errors:**
- Check `conversations/<session>/02_llm/response_*.json` → `"error"` field
- Verify `GROQ_API_KEY` is set in `.env`

**No audio from speaker:**
```bash
aplay -l                              # list devices
aplay -D hw:1,0 output/*.wav          # test device 0
aplay -D hw:1,1 output/*.wav          # test device 1
# Update ALSA_OUTPUT_DEVICE in .env
```

**TTS not working:**
```bash
piper --version                       # verify binary
ls tts/models/                        # verify .onnx files
TTS_BACKEND=espeak python3 stage3_tts_test.py  # test fallback
```
