#!/usr/bin/env python3
"""
Magnum Engine — Onboarding & Launch Wizard

Sequence
────────
1.  Print banner.
2.  Run all system checks silently → display full checklist at once.
3.  If anything is missing, show ACTION REQUIRED with auto-countdown.
4.  Pull missing models (live progress).
5.  Build frontend if dist/ absent.
6.  Kill stale port, launch FastAPI server.
7.  Open http://localhost:8000 in the default browser.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
import webbrowser
from pathlib import Path

# ── Project root (this file lives in <root>/scripts/) ────────────────────────
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# No-window flag for all subprocesses on Windows — prevents CMD flashes.
_NO_WIN = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

# ── ANSI colours ──────────────────────────────────────────────────────────────

def _enable_win_ansi() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetConsoleMode(k32.GetStdHandle(-11), 0x0007)
    except Exception:
        pass

_enable_win_ansi()

for _s in (sys.stdout, sys.stderr):
    try:
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

R  = "\033[91m"
Y  = "\033[93m"
G  = "\033[92m"
C  = "\033[96m"
B  = "\033[94m"
BD = "\033[1m"
DM = "\033[2m"
RS = "\033[0m"

SEP = f"  {DM}{'─' * 54}{RS}"


# ── Model registry ────────────────────────────────────────────────────────────
REQUIRED_MODELS: dict[str, dict] = {
    "qwen2.5vl:3b":  {"gb": 2.0, "role": "Pixel Inspector  (VLM ingestion)"},
    "deepseek-r1:8b": {"gb": 4.9, "role": "Story Curator + Jury Critique"},
    "llama3.2":       {"gb": 2.0, "role": "Competition Curator"},
}

DEPRECATED_LOCAL: dict[str, float] = {
    "phi4-mini-reasoning-q4.gguf": 2.4,
}
DEPRECATED_OLLAMA: list[str] = [
    "llava", "llava:latest", "phi3", "phi3.5",
    "mistral", "deepseek-r1:1.5b",
]

OLLAMA_BASE = "http://localhost:11434"
SERVER_PORT = 8000
SERVER_HOST = "127.0.0.1"
SERVER_URL  = f"http://{SERVER_HOST}:{SERVER_PORT}"
COUNTDOWN_S = 10   # seconds before auto-accepting download


# ── Helpers ───────────────────────────────────────────────────────────────────

def _http_get(url: str, timeout: int = 5) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def _clear_line() -> None:
    sys.stdout.write("\033[2K\r")
    sys.stdout.flush()


# ── Checklist helpers ─────────────────────────────────────────────────────────

def _ok(label: str, detail: str = "") -> str:
    d = f"  {DM}{detail}{RS}" if detail else ""
    return f"  {G}  ✓{RS}  {label}{d}"

def _warn(label: str, detail: str = "") -> str:
    d = f"  {DM}{detail}{RS}" if detail else ""
    return f"  {Y}  ⚠{RS}  {label}{d}"

def _fail(label: str, detail: str = "") -> str:
    d = f"  {DM}{detail}{RS}" if detail else ""
    return f"  {R}  ✗{RS}  {label}{d}"

def _need(label: str, detail: str = "") -> str:
    d = f"  {DM}{detail}{RS}" if detail else ""
    return f"  {Y}  →{RS}  {BD}{label}{RS}{d}"


# ── Banner ────────────────────────────────────────────────────────────────────

def print_banner() -> None:
    print()
    print(f"  {BD}{C}+======================================================+{RS}")
    print(f"  {BD}{C}|       M A G N U M   E N G I N E                     |{RS}")
    print(f"  {BD}{C}|       Street Photography Curation AI  v4.0           |{RS}")
    print(f"  {BD}{C}+======================================================+{RS}")
    print(f"  {DM}  100% Offline  ·  SigLIP-2  ·  DeepSeek-R1  ·  LanceDB{RS}")
    print()


# ── Silent checks (run before printing anything) ──────────────────────────────

def _check_dependencies() -> tuple[bool, str]:
    req = ROOT / "requirements.txt"
    if not req.exists():
        return True, "requirements.txt not found — skipped"
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req), "--quiet"],
        capture_output=True, text=True, creationflags=_NO_WIN,
    )
    return r.returncode == 0, r.stderr.strip()[:120] if r.returncode != 0 else ""


def _check_ollama() -> bool:
    return _http_get(f"{OLLAMA_BASE}/api/tags", timeout=3) is not None


def _get_installed_models() -> set[str]:
    data = _http_get(f"{OLLAMA_BASE}/api/tags", timeout=5)
    if data is None:
        return set()
    try:
        return {m["name"] for m in json.loads(data).get("models", [])}
    except Exception:
        return set()


def _check_models(installed: set[str]) -> dict[str, bool]:
    status: dict[str, bool] = {}
    for model in REQUIRED_MODELS:
        base = model.split(":")[0]
        found = any(
            inst == model or inst.startswith(model + ":") or inst.startswith(base + ":")
            for inst in installed
        )
        status[model] = found
    return status


def _check_frontend() -> bool:
    return (ROOT / "frontend" / "dist" / "index.html").exists()


# ── Full checklist display ────────────────────────────────────────────────────

def print_system_checklist(
    deps_ok: bool,
    deps_err: str,
    ollama_ok: bool,
    model_status: dict[str, bool],
    frontend_ok: bool,
) -> list[str]:
    """
    Print the full system checklist in one go.
    Returns list of missing model names.
    """
    print(f"  {BD}System Checklist{RS}")
    print(SEP)

    # 1. Python dependencies
    if deps_ok:
        print(_ok("Python dependencies"))
    else:
        print(_warn("Python dependencies", deps_err or "some packages may be missing"))

    # 2. Ollama
    if ollama_ok:
        print(_ok("Ollama", f"running at {OLLAMA_BASE}"))
    else:
        print(_fail("Ollama", "not running — install from https://ollama.com/download"))

    # 3. AI Models
    missing: list[str] = []
    for model, info in REQUIRED_MODELS.items():
        ok = model_status.get(model, False)
        label = f"{model:<22}  ~{info['gb']:.1f} GB   {info['role']}"
        if ok:
            print(_ok(label))
        else:
            print(_need(f"MISSING  {label}"))
            missing.append(model)

    # 4. Frontend
    if frontend_ok:
        print(_ok("Frontend build", "dist/ ready"))
    else:
        print(_warn("Frontend build", "will build now"))

    print(SEP)
    return missing


# ── Consent gate with countdown ───────────────────────────────────────────────

def _timed_input(prompt: str, timeout: int) -> str:
    """
    Show prompt and read one line.  On Windows (no select on stdin) we spin in a
    thread so the countdown can tick while waiting for input.
    """
    result: list[str] = []
    answered = threading.Event()

    def _reader():
        try:
            result.append(input())
        except Exception:
            result.append("")
        answered.set()

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    sys.stdout.write(prompt)
    sys.stdout.flush()

    for remaining in range(timeout, 0, -1):
        if answered.wait(timeout=1):
            break
        sys.stdout.write(f"\r{prompt}{remaining:2d}s ")
        sys.stdout.flush()

    if not answered.is_set():
        sys.stdout.write("\n")
        return ""          # timeout → treat as Enter (yes)

    return result[0] if result else ""


def download_consent_gate(missing: list[str]) -> bool:
    if not missing:
        return False

    total_gb = sum(REQUIRED_MODELS[m]["gb"] for m in missing)
    color    = R if total_gb > 6 else Y if total_gb > 3 else C

    print()
    print(f"  {color}{BD}ACTION REQUIRED — {len(missing)} model(s) need downloading:{RS}")
    for m in missing:
        print(f"    {Y}→{RS}  {m:<24}  ~{REQUIRED_MODELS[m]['gb']:.1f} GB")
    print(f"  {color}{BD}  Total: ~{total_gb:.1f} GB{RS}  {DM}(stored in Ollama cache, reused across runs){RS}")
    print()

    try:
        answer = _timed_input(
            f"  Download and continue? {BD}[Y/n]{RS} (auto-yes in {COUNTDOWN_S}s) ",
            COUNTDOWN_S,
        ).strip().lower()
    except KeyboardInterrupt:
        print("\n  Aborted.")
        sys.exit(0)

    return answer in ("", "y", "yes")


# ── Pull models ───────────────────────────────────────────────────────────────

def pull_models(missing: list[str]) -> None:
    if not missing:
        return
    print(f"\n  {BD}Downloading {len(missing)} model(s) — this may take several minutes.{RS}\n")
    for model in missing:
        print(f"  {C}→ ollama pull {model}{RS}")
        try:
            proc = subprocess.Popen(
                ["ollama", "pull", model],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, creationflags=_NO_WIN,
            )
            for line in proc.stdout:  # type: ignore[union-attr]
                s = line.rstrip()
                if not s:
                    continue
                if "pulling" in s.lower() or "%" in s:
                    sys.stdout.write(f"\r    {DM}{s[:78]:<78}{RS}")
                    sys.stdout.flush()
                else:
                    _clear_line()
                    print(f"    {s}")
            proc.wait()
            _clear_line()
            if proc.returncode == 0:
                print(f"  {G}  ✓ {model} ready{RS}\n")
            else:
                print(f"  {R}  ✗ {model} pull failed (exit {proc.returncode}){RS}\n")
        except FileNotFoundError:
            print(f"  {R}  ✗ `ollama` not found in PATH — is Ollama installed?{RS}")
            sys.exit(1)
        except Exception as e:
            print(f"  {R}  ✗ Failed to pull {model}: {e}{RS}")


# ── Legacy cleanup ────────────────────────────────────────────────────────────

def legacy_cleanup(installed: set[str]) -> None:
    models_dir = ROOT / "models"
    found_any  = False

    for fname, size_gb in DEPRECATED_LOCAL.items():
        target = models_dir / fname
        if not target.exists():
            continue
        found_any = True
        try:
            target.unlink()
            print(f"  {G}  ✓ Removed legacy file: {fname} (~{size_gb:.1f} GB freed){RS}")
        except Exception as e:
            print(f"  {Y}  ⚠ Could not remove {fname}: {e}{RS}")

    for model in DEPRECATED_OLLAMA:
        base = model.split(":")[0]
        present = any(inst == model or inst.startswith(base + ":") for inst in installed)
        if not present:
            continue
        found_any = True
        try:
            r = subprocess.run(["ollama", "rm", model],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=_NO_WIN)
            if r.returncode == 0:
                print(f"  {G}  ✓ Removed legacy engine: {model}{RS}")
        except Exception:
            pass

    if found_any:
        print()


# ── Frontend build ────────────────────────────────────────────────────────────

def ensure_frontend() -> None:
    fe_dir    = ROOT / "frontend"
    dist_index = fe_dir / "dist" / "index.html"

    if not fe_dir.exists():
        print(f"  {R}  frontend/ directory not found — skipping{RS}")
        return

    # Rebuild whenever any .tsx/.ts/.css source file is newer than the built index.html.
    # This guarantees that launch.bat always ships the latest code.
    def _needs_build() -> bool:
        if not dist_index.exists():
            return True
        built_mtime = dist_index.stat().st_mtime
        src_dir = fe_dir / "src"
        if not src_dir.exists():
            return False
        for f in src_dir.rglob("*"):
            if f.suffix in (".tsx", ".ts", ".css", ".html") and f.stat().st_mtime > built_mtime:
                return True
        return False

    if not _needs_build():
        print(f"  {G}  ✓ Frontend up to date{RS}")
        return

    print(f"\n  {Y}  Building frontend (source changed)…{RS}")
    try:
        subprocess.run(["npm", "install", "--silent"], cwd=str(fe_dir), check=True, creationflags=_NO_WIN)
        subprocess.run(["npm", "run", "build"],        cwd=str(fe_dir), check=True, creationflags=_NO_WIN)
        print(f"  {G}  ✓ Frontend built{RS}")
    except Exception as e:
        print(f"  {R}  ✗ Frontend build failed: {e}{RS}")
        print(f"  {DM}    Run `cd frontend && npm run build` manually.{RS}")


# ── Server ────────────────────────────────────────────────────────────────────

def _kill_port(port: int) -> None:
    try:
        if sys.platform == "win32":
            out = subprocess.check_output(
                f'netstat -aon | findstr ":{port}" | findstr "LISTENING"',
                shell=True, text=True, stderr=subprocess.DEVNULL, creationflags=_NO_WIN,
            )
            for line in out.strip().splitlines():
                pid = line.strip().split()[-1]
                subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True, creationflags=_NO_WIN)
        else:
            subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True)
    except Exception:
        pass


def launch_server() -> subprocess.Popen:
    venv_py = ROOT / "venv" / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    python  = str(venv_py) if venv_py.exists() else sys.executable
    server  = ROOT / "server.py"

    print(f"\n  {BD}Launching server…{RS}  {DM}{python} {server.name}{RS}\n")
    _kill_port(SERVER_PORT)

    _cflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    return subprocess.Popen(
        [python, str(server)],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        creationflags=_cflags,
    )


def wait_for_server(proc: subprocess.Popen, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    sys.stdout.write(f"  {DM}  Waiting for server")
    sys.stdout.flush()
    while time.time() < deadline:
        if proc.stdout:
            line = proc.stdout.readline()
            if line and line.strip():
                # Print key startup lines inline
                stripped = line.strip()
                if any(k in stripped for k in ("Uvicorn", "Application startup", "ERROR", "error")):
                    _clear_line()
                    print(f"  {DM}  {stripped}{RS}")
                    sys.stdout.write(f"  {DM}  Waiting for server")
                    sys.stdout.flush()
        data = _http_get(SERVER_URL, timeout=1)
        if data is not None:
            sys.stdout.write(f" {G}ready!{RS}\n")
            return True
        if proc.poll() is not None:
            sys.stdout.write(f"\n  {R}  ✗ Server exited unexpectedly (code {proc.returncode}){RS}\n")
            sys.exit(1)
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(0.6)

    sys.stdout.write(f" {Y}(timeout — may still be starting){RS}\n")
    return False


def print_success_banner() -> None:
    url_pad = f"{SERVER_URL:<43}"
    print(f"""
  {BD}{G}+======================================================+
  |   Magnum Engine is running!                          |
  |                                                      |
  |   Open:  {url_pad}|
  |                                                      |
  |   Press Ctrl+C in this window to stop the server.   |
  +======================================================+{RS}
""")


# ── Ollama wait (interactive) ─────────────────────────────────────────────────

def wait_for_ollama_interactive() -> None:
    """Called only when Ollama is offline — pauses and gives install instructions."""
    print()
    print(f"  {R}{BD}  ✗ Ollama is not running (or not installed).{RS}")
    print()
    print(f"  {BD}Install steps:{RS}")
    print(f"    1. Download Ollama from  {C}https://ollama.com/download{RS}")
    print(f"    2. Run the installer — Ollama starts automatically in the tray.")
    print(f"    3. Come back here and press {BD}Enter{RS} to retry.")
    print()

    while True:
        try:
            input("  Press Enter to retry… ")
        except KeyboardInterrupt:
            print("\n  Aborted.")
            sys.exit(1)
        if _http_get(f"{OLLAMA_BASE}/api/tags", timeout=5) is not None:
            print(f"  {G}  ✓ Ollama is running{RS}\n")
            return
        print(f"  {R}  Still not reachable — check Ollama is running in the system tray.{RS}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print_banner()

    # ── Run all checks silently first ─────────────────────────────────────────
    print(f"  {DM}  Running system checks…{RS}", end="", flush=True)

    deps_ok, deps_err = _check_dependencies()
    ollama_ok         = _check_ollama()
    installed         = _get_installed_models() if ollama_ok else set()
    model_status      = _check_models(installed)
    frontend_ok       = _check_frontend()

    _clear_line()

    # ── Print full checklist at once ──────────────────────────────────────────
    missing = print_system_checklist(deps_ok, deps_err, ollama_ok, model_status, frontend_ok)

    # ── Block if Ollama is offline ────────────────────────────────────────────
    if not ollama_ok:
        wait_for_ollama_interactive()
        # Re-check models after Ollama comes up
        installed    = _get_installed_models()
        model_status = _check_models(installed)
        missing      = [m for m, ok in model_status.items() if not ok]

    # ── Legacy cleanup (silent) ───────────────────────────────────────────────
    legacy_cleanup(installed)

    # ── Download missing models ───────────────────────────────────────────────
    if missing:
        if download_consent_gate(missing):
            pull_models(missing)
        else:
            print(f"\n  {Y}  Skipping downloads.{RS}")
            print(f"  {DM}  Jury Critique and Story features will be unavailable until models are installed.{RS}\n")
    else:
        print(f"  {G}  All models present — nothing to download.{RS}\n")

    # ── Build frontend if needed ──────────────────────────────────────────────
    ensure_frontend()

    # ── Launch ────────────────────────────────────────────────────────────────
    server_proc = launch_server()
    wait_for_server(server_proc)

    print_success_banner()
    webbrowser.open(SERVER_URL)

    # Keep wizard alive; stream server output to the window
    try:
        for line in server_proc.stdout:  # type: ignore[union-attr]
            print(f"  {DM}  {line.rstrip()}{RS}")
    except KeyboardInterrupt:
        print(f"\n\n  {Y}  Stopping server…{RS}")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print(f"  {G}  Server stopped. Goodbye.{RS}\n")


if __name__ == "__main__":
    main()
