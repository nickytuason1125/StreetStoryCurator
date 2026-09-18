"""vlm_sidecar — disposable worker process hosting the Qwen-VL GGUF (2026-09-16).

The 4 GB llama.cpp vision model used to live inside the server process, where
"in-process unload" could never reliably return its RAM: llama.cpp arenas,
the CRT heap and mmapped weights survive even after the Llama object is
dropped (measured: 6 GB working set / 10 GB committed on a "model unloaded"
server). Process exit is the ONLY complete release on Windows — the kernel
tears down the whole address space — so the model now runs here, in a
process designed to die.

Lifecycle:
    spawned by critique_engine (its client) on first real vision call
    → serves /infer until idle for VLM_SIDECAR_IDLE_EXIT_S (default 600 s,
      or 120 s if never used at all) → os._exit(0), everything freed.
    → the server's residency reaper can also force an earlier exit via
      critique_engine.unload() → POST /evict.

Supervision: a marker file (cache/vlm_sidecar.json) records {pid, port,
parent_pid}. critique_engine.sweep_orphan_sidecar() — called at server boot —
kills any sidecar whose parent server is gone, so an orphan can never sit on
gigabytes unnoticed.

Usage:
    python vlm_sidecar.py --port <port> --parent <server_pid>
"""
import sys, os, json, time, threading, argparse, socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_MARKER_PATH = os.path.join(_ROOT, "cache", "vlm_sidecar.json")
_IDLE_EXIT_S = float(os.environ.get("VLM_SIDECAR_IDLE_EXIT_S", "600"))
_UNUSED_EXIT_S = float(os.environ.get("VLM_SIDECAR_UNUSED_EXIT_S", "120"))

_vision = None          # Qwen-VL GGUF (multimodal, image+text)
_text = None            # text-LLM GGUF (local_llm's model)
_grammars: dict = {}    # grammar_src / json_schema → LlamaGrammar
_last_used = 0.0        # monotonic ts of last /infer
_started = 0.0          # monotonic ts of process start
_lock = threading.Lock()
_busy = False           # True while an /infer or /preload is in flight — the
                        # server's pressure watchdog must NOT evict mid-job
                        # (an evict here kills the job AND forces a full cold
                        # reload — the "stuck at 33%" thrash of 2026-09-16).


def _touch():
    global _last_used
    _last_used = time.monotonic()
    _refresh_marker()


def _refresh_marker():
    try:
        with open(_MARKER_PATH, "r", encoding="utf-8") as f:
            m = json.load(f)
        if m.get("pid") == os.getpid():
            m["last_used"] = time.time()
            with open(_MARKER_PATH, "w", encoding="utf-8") as f:
                json.dump(m, f)
    except Exception:
        pass


def _idle_seconds() -> float:
    return time.monotonic() - max(_last_used, _started)


def _load_vision():
    """Qwen-VL multimodal GGUF (critique_engine's model)."""
    global _vision
    if _vision is not None:
        return _vision
    import model_registry as _mr
    gguf = _mr.vision_gguf_path()
    mmproj = _mr.vision_mmproj_path()
    if not gguf.exists() or not mmproj.exists():
        return None
    from llama_cpp import Llama
    try:
        from llama_cpp.llama_chat_format import Qwen2VLChatHandler as _Handler
    except ImportError:
        from llama_cpp.llama_chat_format import Llava15ChatHandler as _Handler
    chat_handler = _Handler(clip_model_path=str(mmproj))
    _n_threads = min(os.cpu_count() or 4, 8)
    print(f"[sidecar] Loading Qwen-VL GGUF  threads={_n_threads}  ctx=2048", flush=True)
    _vision = Llama(
        model_path=str(gguf),
        chat_handler=chat_handler,
        n_ctx=2048,
        n_gpu_layers=-1,
        n_threads=_n_threads,
        verbose=False,
    )
    print("[sidecar] Qwen-VL GGUF ready.", flush=True)
    return _vision


# local_llm's offload ladder, moved verbatim: a build without enough VRAM
# fails at construction and backs off one rung at a time. 0 (pure CPU) last.
_GPU_LAYER_LADDER = (-1, 20, 10, 0)


def _load_text():
    """Text-LLM GGUF (local_llm's model), same offload ladder as in-process."""
    global _text
    if _text is not None:
        return _text
    import model_registry as _mr
    path = _mr.text_gguf_path()
    if not path.exists():
        return None
    try:
        from tier_select import has_gpu
        has_cuda = has_gpu()
    except Exception:
        has_cuda = False
    from llama_cpp import Llama
    ladder = _GPU_LAYER_LADDER if has_cuda else (0,)
    for n_gpu in ladder:
        try:
            _text = Llama(
                model_path=str(path),
                n_ctx=4096,
                n_gpu_layers=n_gpu,
                # flash_attn halves the KV cache, but some builds reject it on
                # a pure-CPU context. The n_gpu=0 rung is the last resort.
                flash_attn=(n_gpu != 0),
                n_threads=min(os.cpu_count() or 4, 8),
                verbose=False,
            )
            print(f"[sidecar] text GGUF {path.name} loaded (n_gpu_layers={n_gpu})",
                  flush=True)
            return _text
        except Exception as e:
            print(f"[sidecar] text load failed at n_gpu_layers={n_gpu} ({e}) — backing off",
                  flush=True)
    _text = None
    return None


def _get_grammar(src: str, from_json: bool = False):
    key = f"{'json:' if from_json else ''}{src}"
    g = _grammars.get(key)
    if g is not None:
        return g
    from llama_cpp import LlamaGrammar
    g = (LlamaGrammar.from_json_schema(src) if from_json
         else LlamaGrammar.from_string(src))
    _grammars[key] = g
    return g


def _infer(d: dict) -> dict:
    with _lock:
        kind = d.get("kind", "vision")
        if kind == "text":
            llm = _load_text()
            if llm is None:
                return {"ok": False, "error": "text model files missing"}
        else:
            llm = _load_vision()
            if llm is None:
                return {"ok": False, "error": "vision model files missing"}
        kwargs = dict(
            messages=d["messages"],
            temperature=float(d.get("temperature", 0.1)),
            max_tokens=int(d.get("max_tokens", 200)),
        )
        gsrc = d.get("grammar_src")
        schema = d.get("json_schema")
        if gsrc:
            try:
                kwargs["grammar"] = _get_grammar(gsrc)
            except Exception as e:
                print(f"[sidecar] grammar build failed ({e})", flush=True)
                return {"ok": False, "grammar_failed": True}
        elif schema is not None:
            try:
                import json as _json
                kwargs["grammar"] = _get_grammar(_json.dumps(schema), from_json=True)
            except Exception as e:
                print(f"[sidecar] grammar build failed ({e})", flush=True)
                return {"ok": False, "grammar_failed": True}
        output = llm.create_chat_completion(**kwargs)
        return {"ok": True,
                "text": (output["choices"][0]["message"]["content"] or "")}


def _idle_watch():
    while True:
        time.sleep(15)
        idle = _idle_seconds()
        loaded = (_vision is not None) or (_text is not None)
        exit_s = _IDLE_EXIT_S if loaded else _UNUSED_EXIT_S
        if idle > exit_s:
            print(f"[sidecar] idle {idle:.0f} s — exiting (all RAM returns to the OS)",
                  flush=True)
            os._exit(0)


def _drop(model: "str | None") -> bool:
    """Targeted evict. Returns True when the whole process should exit."""
    if model == "vision":
        globals()["_vision"] = None
    elif model == "text":
        globals()["_text"] = None
    import gc
    gc.collect()
    return (model is None) or ((_vision is None) and (_text is None))


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):        # silence per-request noise
        pass

    def _json(self, code: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True,
                             "loaded": {"vision": _vision is not None,
                                        "text": _text is not None},
                             "busy": _busy,
                             "idle_s": round(_idle_seconds(), 1)})
        else:
            self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._json(400, {"ok": False, "error": f"bad body: {e}"})
            return
        if self.path == "/infer":
            global _busy
            _touch()
            _busy = True
            try:
                out = _infer(d)
                self._json(200, out)
            except Exception as e:
                self._json(200, {"ok": False, "error": f"{type(e).__name__}: {e}"})
            finally:
                _busy = False
        if self.path == "/preload":
            # Warm-ahead support: load a model WITHOUT generating. The server
            # calls this when the user shows browsing intent, so the first
            # real critique doesn't pay the cold load.
            _touch()
            model = d.get("model", "vision")
            _busy = True
            try:
                with _lock:
                    llm = _load_vision() if model == "vision" else _load_text()
                self._json(200, {"ok": llm is not None, "model": model})
            except Exception as e:
                self._json(200, {"ok": False, "error": f"{type(e).__name__}: {e}"})
            finally:
                _busy = False
        elif self.path == "/evict":
            # The whole point: exit, don't unload-in-place. A targeted evict
            # ("model": "vision"/"text") only exits when nothing remains
            # loaded; a bare evict always exits. Response first, then
            # teardown — the client reads the ack before we vanish.
            try:
                self._json(200, {"ok": True, "bye": True})
                self.wfile.flush()
            except Exception:
                pass
            if _drop(d.get("model")):
                print("[sidecar] evict — exiting", flush=True)
                os._exit(0)
            print(f"[sidecar] evicted {d.get('model')} — still serving", flush=True)
        else:
            self._json(404, {"ok": False, "error": "not found"})


def main():
    global _started
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--parent", type=int, required=True)
    a = ap.parse_args()
    _started = time.monotonic()

    os.makedirs(os.path.dirname(_MARKER_PATH), exist_ok=True)
    with open(_MARKER_PATH, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "port": a.port,
                   "parent_pid": a.parent, "started": time.time()}, f)

    threading.Thread(target=_idle_watch, daemon=True).start()
    print(f"[sidecar] up on 127.0.0.1:{a.port} (parent {a.parent}); "
          f"idle exit {_IDLE_EXIT_S:.0f}s / unused {_UNUSED_EXIT_S:.0f}s", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), _Handler).serve_forever()


if __name__ == "__main__":
    main()
