#!/bin/bash
# Street Story Curator — Mac launcher (double-click in Finder)

cd "$(dirname "$0")"
VENV="$(pwd)/venv"

alert() {
    osascript -e "display alert \"Street Story Curator\" message \"$1\" buttons {\"OK\"}" 2>/dev/null || echo "$1"
}

# ── Python check ──────────────────────────────────────────────────
PYTHON=""
for try_py in python3.12 python3.11 python3.10 python3; do
    if command -v "$try_py" &>/dev/null; then
        PYTHON="$try_py"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    alert "Python 3.10 or newer is required.\n\nDownload from python.org then re-run this launcher."
    open "https://www.python.org/downloads/"
    exit 1
fi

# ── First-run setup ───────────────────────────────────────────────
if [ ! -f "$VENV/bin/python" ]; then
    echo ""
    echo "First launch: setting up environment."
    echo "This downloads ~1.5 GB of libraries and takes 5-10 minutes."
    echo "Subsequent launches will be instant."
    echo ""

    "$PYTHON" -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip -q

    echo "[1/3] Installing PyTorch..."
    "$VENV/bin/pip" install torch torchvision -q
    if [ $? -ne 0 ]; then
        echo "ERROR: PyTorch install failed. Check your internet connection."
        read -p "Press Enter to exit."
        exit 1
    fi

    echo "[2/3] Installing CLIP..."
    "$VENV/bin/pip" install \
        "clip @ git+https://github.com/openai/CLIP.git" -q 2>/dev/null || \
        echo "WARNING: CLIP install failed (git may not be available). Some features may be limited."

    echo "[3/3] Installing remaining dependencies..."
    "$VENV/bin/pip" install -r requirements.txt -q
    if [ $? -ne 0 ]; then
        echo "ERROR: Dependency install failed."
        read -p "Press Enter to exit."
        exit 1
    fi

    echo ""
    echo "Setup complete."
    echo ""
fi

# ── Build frontend if dist is missing ────────────────────────────
if [ ! -f "frontend/dist/index.html" ]; then
    if command -v npm &>/dev/null; then
        echo "Building UI..."
        (cd frontend && npm install -q && npm run build -q)
    else
        echo "WARNING: npm not found. Install Node.js from https://nodejs.org"
        echo "Then re-run this launcher."
        read -p "Press Enter to exit."
        exit 1
    fi
fi

# ── Launch ────────────────────────────────────────────────────────
"$VENV/bin/python" src/local_launcher.py &
