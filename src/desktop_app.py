import os, sys, time, threading, socket, webbrowser, logging
from pathlib import Path

_log = logging.getLogger(__name__)


def find_free_port(start=7860, end=7875):
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError("No free ports available in range 7860-7875")


def run_gradio(port):
    try:
        import gradio as gr
        from app import build_ui, _CSS, _allowed_paths
        demo = build_ui()
        demo.queue(default_concurrency_limit=2).launch(
            server_name="127.0.0.1", server_port=port,
            quiet=True, inbrowser=False, prevent_thread_lock=False,
            theme=gr.themes.Soft(), css=_CSS,
            allowed_paths=_allowed_paths(),
            inline=False,
        )
    except Exception as exc:
        _log.error("Gradio launch failed: %s", exc)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    port = find_free_port()
    _log.info("Street Story Curator — starting on port %s...", port)

    # Add src/ to sys.path and anchor cwd at project root before the thread starts
    _src = Path(__file__).resolve().parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    os.chdir(_src.parent)   # ./cache, ./output, ./models resolve from here

    t = threading.Thread(target=run_gradio, args=(port,), daemon=True)
    t.start()

    # Poll until Gradio is accepting connections (up to 6 seconds)
    url = f"http://127.0.0.1:{port}"
    for _ in range(15):
        try:
            import urllib.request
            urllib.request.urlopen(url, timeout=1)
            break
        except Exception:
            time.sleep(0.4)
    else:
        _log.warning("Server slow to start — opening anyway...")

    _log.info("Opening %s", url)
    webbrowser.open(url)
    _log.info("Keep this window open. Press Ctrl+C or close to stop.")
    try:
        input()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
