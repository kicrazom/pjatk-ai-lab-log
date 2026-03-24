#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
BACKEND="$SCRIPT_DIR/backend"
PYTHON="$VENV_DIR/bin/python3"

# Auto-setup if venv doesn't exist yet
if [ ! -f "$PYTHON" ]; then
    echo "⚠ venv not found — running setup first..."
    echo ""
    bash "$SCRIPT_DIR/setup.sh"
    echo ""
fi

echo "┌──────────────────────────────────────────────┐"
echo "│  AI Workstation Dashboard                     │"
echo "│  http://localhost:8000                        │"
echo "│  http://$(hostname -I | awk '{print $1}'):8000                │"
echo "│  Press Ctrl+C to stop                         │"
echo "└──────────────────────────────────────────────┘"
echo ""

cd "$BACKEND"
exec "$PYTHON" server.py
