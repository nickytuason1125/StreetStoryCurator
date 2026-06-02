#!/usr/bin/env bash
# Magnum Engine — Street Photo Curation AI
# Lightweight launcher: activates venv → installs deps → runs setup_wizard.py
set -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "  ============================================================"
echo "   Magnum Engine — Street Photo Curation AI"
echo "  ============================================================"
echo ""

# ── Python check ──────────────────────────────────────────────────────────────
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "  [ERROR] Python not found."
    echo "          Install Python 3.10+ via your package manager or https://python.org"
    exit 1
fi

# ── Create venv if absent ─────────────────────────────────────────────────────
if [ ! -f "venv/bin/activate" ]; then
    echo "  Creating Python virtual environment..."
    "$PYTHON" -m venv venv
    echo "  Virtual environment created."
    echo ""
fi

# ── Activate venv ─────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source venv/bin/activate

# ── Run setup wizard ──────────────────────────────────────────────────────────
python scripts/setup_wizard.py "$@"
