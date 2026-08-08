#!/bin/bash
# scripts/install_piper_g1.sh
# ---------------------------
# Install Piper TTS binary and the en_US-ryan-high voice on the G1.
#
# Run this ON THE G1 (not the dev machine):
#   cd ~/g1_voice_assistant
#   bash scripts/install_piper_g1.sh
#
# Requirements: curl or wget, internet access from the G1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${PROJECT_DIR}/tts/models"
INSTALL_BIN="/usr/local/bin/piper"

echo "================================================================"
echo " Piper TTS Installer for Unitree G1 (linux_aarch64)"
echo "================================================================"
echo ""

# ── Check architecture ────────────────────────────────────────────────────────
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "WARNING: This script is designed for aarch64 (ARM64)."
    echo "         Detected: $ARCH"
    echo "         You may need to download a different binary."
    echo ""
fi

# ── Install piper binary ──────────────────────────────────────────────────────
echo "[1/3] Installing Piper binary …"

if command -v piper &> /dev/null; then
    echo "   Piper already installed: $(piper --version)"
else
    PIPER_URL="https://github.com/rhasspy/piper/releases/latest/download/piper_linux_aarch64.tar.gz"
    TMP_DIR=$(mktemp -d)

    echo "   Downloading: $PIPER_URL"
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "${TMP_DIR}/piper.tar.gz" "$PIPER_URL"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "${TMP_DIR}/piper.tar.gz" "$PIPER_URL"
    else
        echo "ERROR: Neither wget nor curl found. Install one first."
        exit 1
    fi

    echo "   Extracting …"
    tar -xzf "${TMP_DIR}/piper.tar.gz" -C "$TMP_DIR"

    echo "   Installing to ${INSTALL_BIN} …"
    if [ -f "${TMP_DIR}/piper/piper" ]; then
        sudo cp "${TMP_DIR}/piper/piper" "$INSTALL_BIN"
        sudo chmod +x "$INSTALL_BIN"
    else
        # Some releases put the binary directly in the tar root
        sudo cp "${TMP_DIR}/piper" "$INSTALL_BIN" 2>/dev/null || \
            sudo find "$TMP_DIR" -name "piper" -type f -exec cp {} "$INSTALL_BIN" \;
        sudo chmod +x "$INSTALL_BIN"
    fi

    rm -rf "$TMP_DIR"
    echo "   Piper installed: $(piper --version)"
fi

echo ""

# ── Download voice model ──────────────────────────────────────────────────────
echo "[2/3] Downloading en_US-ryan-high voice model …"
mkdir -p "$MODELS_DIR"

VOICE_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high"
ONNX_FILE="${MODELS_DIR}/en_US-ryan-high.onnx"
JSON_FILE="${MODELS_DIR}/en_US-ryan-high.onnx.json"

if [ -f "$ONNX_FILE" ] && [ -f "$JSON_FILE" ]; then
    echo "   Voice model already present."
else
    echo "   Downloading .onnx (may take a minute) …"
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$ONNX_FILE" "${VOICE_BASE}/en_US-ryan-high.onnx"
        wget -q --show-progress -O "$JSON_FILE" "${VOICE_BASE}/en_US-ryan-high.onnx.json"
    else
        curl -L --progress-bar -o "$ONNX_FILE" "${VOICE_BASE}/en_US-ryan-high.onnx"
        curl -L --progress-bar -o "$JSON_FILE" "${VOICE_BASE}/en_US-ryan-high.onnx.json"
    fi
    echo "   Model downloaded: $(du -sh "$ONNX_FILE" | cut -f1)"
fi

echo ""

# ── Smoke test ────────────────────────────────────────────────────────────────
echo "[3/3] Smoke test …"
TEST_WAV="/tmp/piper_test.wav"
echo "Hello, I am G1, your Unitree robot assistant." | \
    piper --model "$ONNX_FILE" --output_file "$TEST_WAV"

if [ -f "$TEST_WAV" ] && [ -s "$TEST_WAV" ]; then
    SIZE=$(du -sh "$TEST_WAV" | cut -f1)
    echo "   WAV generated: ${TEST_WAV} (${SIZE})"
    echo ""
    echo "   Playing test audio …"
    aplay "$TEST_WAV" 2>/dev/null || echo "   (aplay failed – will try ALSA_OUTPUT_DEVICE next)"
    rm -f "$TEST_WAV"
else
    echo "ERROR: Piper produced no output. Check the model files."
    exit 1
fi

echo ""
echo "================================================================"
echo " Piper TTS installation complete!"
echo ""
echo " Voice model: ${MODELS_DIR}/en_US-ryan-high.onnx"
echo ""
echo " Now update .env:"
echo "   TTS_BACKEND=piper"
echo "   PIPER_MODEL=en_US-ryan-high"
echo "   PIPER_MODELS_DIR=tts/models"
echo ""
echo " Then run: python3 stage3_tts_test.py"
echo "================================================================"
