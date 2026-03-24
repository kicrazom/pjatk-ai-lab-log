#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
BACKEND="$SCRIPT_DIR/backend"

echo "┌──────────────────────────────────────────────┐"
echo "│  AI Workstation Dashboard — Setup             │"
echo "└──────────────────────────────────────────────┘"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 not found."
    echo "   Install: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi
echo "✓ Python: $(python3 --version)"

# Ensure python3-venv is available
if ! python3 -m venv --help &>/dev/null; then
    echo ""
    echo "❌ python3-venv not installed."
    echo "   Install: sudo apt install python3-venv"
    exit 1
fi

# Create venv if missing
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "→ Creating virtual environment in .venv/ ..."
    python3 -m venv "$VENV_DIR"
    echo "✓ venv created"
else
    echo "✓ venv already exists"
fi

# Install / update deps
echo ""
echo "→ Installing Python dependencies into venv..."
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -r "$BACKEND/requirements.txt" -q
echo "✓ Dependencies installed"

# Check rocm-smi
echo ""
if command -v rocm-smi &>/dev/null; then
    echo "✓ rocm-smi: $(which rocm-smi)"
elif [ -f /opt/rocm/bin/rocm-smi ]; then
    echo "✓ rocm-smi: /opt/rocm/bin/rocm-smi"
else
    echo "⚠ rocm-smi not found — GPU panel will show 'unavailable'"
    echo "  Install: sudo apt install rocm-smi-lib"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete!"
echo ""
echo "  To start:  ./start.sh"
echo "  To enable at boot: sudo cp ai-dashboard.service /etc/systemd/system/"
echo "                     sudo systemctl daemon-reload"
echo "                     sudo systemctl enable --now ai-dashboard"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
