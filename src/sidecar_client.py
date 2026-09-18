"""sidecar_client — the one client for the VLM/text sidecar process (2026-09-16).

critique_engine (vision GGUF) and local_llm (text GGUF) both talk to ONE
disposable sidecar process (src/vlm_sidecar.py) that hosts both llama.cpp
models. This module owns everything lifecycle-related so the two callers
cannot drift:

    sidecar_client.infer(payload)      → dict from /infer, one respawn-retry
    sidecar_client.ensure()            → adopt a healthy sidecar or spawn one
    sidecar_client.evict(model=None)   → targeted unload ("vision"/"text");
                                         None ⇒ the whole sidecar exits
    sidecar_client.kill()              → hard stop (unhealthy/stale sidecar)
    sidecar_client.sweep_orphan()      → boot hygiene: kill provably orphaned

Design: process exit is the ONLY complete RAM release on Windows (llama.cpp
arenas, mmapped weights and the CRT heap survive in-process unloads), so the
sidecar is designed to die — on its own idle timer, on a targeted evict that
leaves nothing loaded, or on an explicit kill.
"""
import json
import os
import socket
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_MARKER = _ROOT / "cache" / "vlm_sidecar.json"
_LOG = _ROOT / "cache" / "vlm_sidecar.log"

_port: int = 0
_ensure_lock = threading.Lock()


def marker_path() -> Path:
    return _MARKER


def read_marker() -> "dict | None":
    try:
        return json.loads(_MARKER.read_text(encoding="utf-8"))
    except Exception:
        return None


def probe_health(port: int, timeout: float = 2.0) -> "dict | None":
    try:
        import requests as _rq
        r = _rq.get(f"http://127.0.0.1:{port}/health", timeout=timeout)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None


def pid_alive(pid: int) -> bool:
    try:
        import psutil as _ps
        return _ps.pid_exists(pid)
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False


def kill() -> None:
    """Terminate the sidecar recorded in the marker (best-effort) and drop it."""
    m = read_marker()
    if m:
        try:
            import requests as _rq
            _rq.post(f"http://127.0.0.1:{int(m['port'])}/evict", json={}, timeout=3)
        except Exception:
            try:
                os.kill(int(m["pid"]), 9)
            except Exception:
                pass
    try:
        _MARKER.unlink(missing_ok=True)
    except Exception:
        pass


def spawn() -> bool:
    """Launch a detached sidecar on a free port. Source-tree only — the frozen
    bundle has no separate interpreter to spawn; the packaged app already runs
    its models in isolated subprocesses via the grade worker."""
    import sys as _sys
    if getattr(_sys, "frozen", False):
        print("[sidecar] frozen build — sidecar unavailable")
        return False
    import subprocess as _sp
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    flags = 0x08000000 if os.name == "nt" else 0
    try:
        log = open(_LOG, "ab")
        _sp.Popen(
            [_sys.executable, str(_ROOT / "src" / "vlm_sidecar.py"),
             "--port", str(port), "--parent", str(os.getpid())],
            cwd=str(_ROOT), stdout=log, stderr=log, creationflags=flags,
        )
        return True
    except Exception as e:
        print(f"[sidecar] spawn failed: {e}")
        return False


def ensure() -> bool:
    """Adopt a healthy sidecar, or spawn one and wait for it to come up.

    Any caller (vision or text) gets the SAME process — whoever spawns first
    owns the marker; the other adopts via the health probe.
    """
    global _port
    with _ensure_lock:
        m = read_marker()
        if m:
            h = probe_health(int(m["port"]))
            if h is not None:
                _port = int(m["port"])
                return True
            # Stale marker: sidecar gone or unhealthy. A healthy sidecar
            # probes OK above and is adopted — never killed here.
            kill()
        if not spawn():
            return False
        for _ in range(40):                 # ~10 s for marker + bind
            m2 = read_marker()
            if m2 and probe_health(int(m2["port"])) is not None:
                _port = int(m2["port"])
                return True
            time.sleep(0.25)
        print("[sidecar] did not come up in time")
        return False


def infer(payload: dict, read_timeout: float = 300.0) -> "dict | None":
    """One /infer round-trip with a single respawn-retry (covers a sidecar
    that died between the health probe and the infer). Returns the response
    dict, or None when the sidecar could not be brought up at all."""
    global _port
    for attempt in (1, 2):
        if not ensure():
            return None
        try:
            import requests as _rq
            r = _rq.post(f"http://127.0.0.1:{_port}/infer", json=payload,
                         timeout=(5, read_timeout))
            if r.ok:
                return r.json()
            print(f"[sidecar] HTTP {r.status_code}")
            return None
        except Exception as e:
            print(f"[sidecar] infer failed ({e})"
                  + (" — respawning" if attempt == 1 else ""))
            kill()
    return None


def preload(model: str, read_timeout: float = 600.0) -> bool:
    """Load `model` ("vision"/"text") in the sidecar WITHOUT generating.
    Warm-ahead support — returns True when the model came up resident."""
    global _port
    if not ensure():
        return False
    try:
        import requests as _rq
        r = _rq.post(f"http://127.0.0.1:{_port}/preload",
                     json={"model": model}, timeout=(5, read_timeout))
        if r.ok:
            j = r.json()
            return bool(j.get("ok"))
        print(f"[sidecar] preload HTTP {r.status_code}")
    except Exception as e:
        print(f"[sidecar] preload failed ({e})")
    return False


def evict(model: "str | None" = None) -> None:
    """Targeted unload. `model` ("vision"/"text") drops that one model; when
    nothing remains loaded the sidecar exits itself. None ⇒ exit regardless."""
    m = read_marker()
    if not m:
        return
    try:
        import requests as _rq
        _rq.post(f"http://127.0.0.1:{int(m['port'])}/evict",
                 json={"model": model} if model else {}, timeout=3)
    except Exception:
        try:
            os.kill(int(m["pid"]), 9)
        except Exception:
            pass


def sweep_orphan() -> bool:
    """Boot hygiene: kill a sidecar whose owning server is gone. Returns True
    when an orphan was terminated. A sidecar whose parent is alive (another
    running FirstCut instance) is left strictly alone."""
    m = read_marker()
    if not m:
        return False
    parent = int(m.get("parent_pid", 0))
    pid = int(m.get("pid", 0))
    if parent == os.getpid():
        return False                        # ours; ensure() owns it
    if parent > 0 and pid_alive(parent):
        return False                        # a live peer's sidecar — leave it
    try:
        os.kill(pid, 9)
        print(f"[sidecar] startup: swept orphan sidecar (pid {pid}, "
              f"parent {parent} gone)", flush=True)
    except Exception:
        pass
    try:
        _MARKER.unlink(missing_ok=True)
    except Exception:
        pass
    return True