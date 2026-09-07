"""Ad-hoc validation for the thumb dedup + removable-volume prewarm changes.

Run:  venv\\Scripts\\python.exe scripts\\check_thumb_dedup.py
"""
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("FIRSTCUT_LITE", "1")

import server_impl as _si      # noqa: E402,F401  (must load before routers: mount_all)
import routers.library as lib   # noqa: E402

fails = []
t0 = time.time()


def stamp():
    return f"[{time.time() - t0:6.2f}s]"


def check(name, ok, detail=""):
    print(f"{stamp()} {'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        fails.append(name)


# --- 1. _is_slow_volume: card reader vs fixed drive ------------------------
check("F: (card reader) detected removable", lib._is_slow_volume("F:\\DCIM\\100MSDCF") is True)
check("C: (fixed) not removable", lib._is_slow_volume("C:\\Users") is False)

# --- 2. dedup: two concurrent callers, one decode ---------------------------
calls = []
log = []
gate = threading.Event()


def fake_decode(path, low_priority=False):
    dest = lib.THUMB_DIR / lib._thumb_cache_name(Path(path).resolve())
    if dest.exists():
        return                      # real decode re-checks the cache first
    calls.append((path, low_priority))
    log.append(f"{stamp()} decode enter lp={low_priority}")
    gate.wait(timeout=10)         # hold the decode open
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"fake")
    log.append(f"{stamp()} decode exit lp={low_priority}")


lib._gen_one_thumb_decode = fake_decode

p = str(Path(os.environ.get("TEMP", os.environ.get("TMP", "."))) / "dedup_test_standin.jpg")


def runner(low_priority, hold, ev_name):
    lib._gen_one_thumb(p, low_priority)
    log.append(f"{stamp()} runner-{ev_name} done")


t1 = threading.Thread(target=runner, args=(True, None, "prewarm"))
t1.start()
time.sleep(0.3)                   # let t1 register as owner
check("owner registered", len(lib._thumb_inflight) == 1, f"inflight={list(lib._thumb_inflight)}")
t2 = threading.Thread(target=runner, args=(False, None, "ondemand"))
t2.start()
time.sleep(0.5)
check("duplicate waited (no 2nd decode yet)", len(calls) == 1, f"calls={len(calls)}")
gate.set()
t1.join(15); t2.join(15)
for line in log:
    print(line, flush=True)
check("exactly one decode for two callers", len(calls) == 1, f"calls={len(calls)}")
check("threads finished", not t1.is_alive() and not t2.is_alive())
check("in-flight registry cleaned up", len(lib._thumb_inflight) == 0,
      f"inflight={list(lib._thumb_inflight)}")

# --- 3. on-demand caller falls through when owner dropped the work ---------
calls.clear()
lib._thumb_inflight.clear()


def fake_decode_drop(path, low_priority=False):
    calls.append(path)            # RAM gate drop: registers, produces nothing


lib._gen_one_thumb_decode = fake_decode_drop
lib._gen_one_thumb(p, True)       # owner: registers, produces nothing, exits
check("drop owner registered then exited", len(lib._thumb_inflight) == 0)
lib._gen_one_thumb(p, False)      # on-demand caller: must fall through and decode
check("on-demand regenerates after RAM-gate drop", len(calls) == 2, f"calls={len(calls)}")

# --- 4. registry empty after runs -------------------------------------------
check("registry empty after runs", len(lib._thumb_inflight) == 0)

# --- cleanup: the fake decode wrote a cache entry for the stand-in path -----
try:
    (lib.THUMB_DIR / lib._thumb_cache_name(Path(p).resolve())).unlink()
    print(f"{stamp()} cleaned up fake cache entry")
except FileNotFoundError:
    pass

print()
print("RESULT:", "ALL PASS" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)

