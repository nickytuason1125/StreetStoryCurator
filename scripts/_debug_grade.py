import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8000"
FOLDER = ROOT.parent / "Sample_Street"

import requests

server = subprocess.Popen([sys.executable, str(ROOT / "server.py")], cwd=str(ROOT),
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(60):
        time.sleep(2)
        try:
            if requests.get(f"{BASE}/api/health", timeout=2).status_code == 200:
                print("server up")
                break
        except Exception:
            continue

    body = {"folder_paths": [str(FOLDER)], "folder_path": str(FOLDER),
            "preset": "classic_street", "force_rescan": True,
            "scan_mode": False, "deep_grade": False, "mogco_target": 0}
    with requests.post(f"{BASE}/api/grade/v2/stream", json=body, stream=True,
                       timeout=(10, 900)) as resp:
        print("status:", resp.status_code)
        n = 0
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            n += 1
            if n <= 6 or "error" in line.lower() or "done" in line.lower():
                print(f"[{n}] {line[:220]}")
            if n > 4000:
                break
finally:
    server.terminate()
    try:
        server.wait(timeout=10)
    except Exception:
        server.kill()
