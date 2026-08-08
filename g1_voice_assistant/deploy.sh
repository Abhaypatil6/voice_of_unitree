#!/bin/bash
# deploy.sh
# ---------
# Deploy g1_voice_assistant to the Unitree G1 robot.
#
# Usage (from the dev machine, inside voice_of_unitree/):
#   chmod +x g1_voice_assistant/deploy.sh
#   ./g1_voice_assistant/deploy.sh
#
# The script will NOT copy .env (keep API keys off the robot's SSH transfer).
# Set GROQ_API_KEY directly on the G1 after deploying.

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
G1_USER="${G1_USER:-unitree}"
G1_HOST="${G1_HOST:-192.168.123.164}"
G1_DEST="${G1_DEST:-~/g1_voice_assistant}"
LOCAL_SRC="$(cd "$(dirname "$0")" && pwd)"

echo "================================================================"
echo " G1 Voice Assistant – Deploy"
echo "================================================================"
echo " Source : $LOCAL_SRC"
echo " Target : ${G1_USER}@${G1_HOST}:${G1_DEST}"
echo "================================================================"
echo ""

# ── Check SSH connectivity ──────────────────────────────────────────────────
echo "[1/4] Checking SSH connectivity to G1 (${G1_HOST}) …"
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes \
        "${G1_USER}@${G1_HOST}" "echo OK" 2>/dev/null; then
    echo ""
    echo "ERROR: Cannot connect to ${G1_USER}@${G1_HOST}"
    echo ""
    echo "Check that:"
    echo "  1. The G1 is powered on and on the same network"
    echo "  2. You can ping it:  ping ${G1_HOST}"
    echo "  3. SSH key is set up:  ssh-copy-id ${G1_USER}@${G1_HOST}"
    echo "  4. Or use Ethernet (192.168.123.164) instead of Wi-Fi (192.168.0.57)"
    exit 1
fi
echo "   Connected ✓"
echo ""

# ── Sync files ─────────────────────────────────────────────────────────────
echo "[2/4] Syncing files …"
rsync -avz --progress \
    --exclude='.env' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache/' \
    --exclude='conversations/' \
    --exclude='output/' \
    --exclude='logs/' \
    --exclude='.git/' \
    --exclude='*.egg-info/' \
    "${LOCAL_SRC}/" \
    "${G1_USER}@${G1_HOST}:${G1_DEST}/"

echo "   Sync complete ✓"
echo ""

# ── Create required directories on G1 ──────────────────────────────────────
echo "[3/4] Creating runtime directories on G1 …"
ssh "${G1_USER}@${G1_HOST}" \
    "mkdir -p ${G1_DEST}/conversations ${G1_DEST}/output ${G1_DEST}/logs ${G1_DEST}/tts/models"
echo "   Directories created ✓"
echo ""

# ── Post-deploy instructions ────────────────────────────────────────────────
echo "[4/4] Post-deploy checklist:"
echo ""
echo "   On the G1, run:"
echo ""
echo "   ssh ${G1_USER}@${G1_HOST}"
echo ""
echo "   # 1. Set up environment"
echo "   cd ${G1_DEST}"
echo "   cp .env.example .env"
echo "   nano .env    # set GROQ_API_KEY and ALSA_OUTPUT_DEVICE"
echo ""
echo "   # 2. Install Python dependencies"
echo "   pip3 install -r requirements.txt"
echo ""
echo "   # 3. (Optional) Install Piper TTS"
echo "   bash scripts/install_piper_g1.sh"
echo ""
echo "   # 4. Source ROS 2 and test Stage 1"
echo "   source /opt/ros/foxy/setup.bash"
echo "   python3 stage1_listener.py"
echo ""
echo "   # 5. Test the full pipeline"
echo "   python3 main.py"
echo ""
echo "================================================================"
echo " Deploy DONE"
echo "================================================================"
