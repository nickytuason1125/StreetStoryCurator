# PORTING.md — macOS / macOS ARM status

Updated 2026-09. Windows 10+ remains the primary, fully-verified platform.
macOS 13+ (Intel and Apple Silicon) is **ported at the code level** and runs
the CPU tier; it has NOT yet been verified on real Mac hardware — the
checklist below is what a first run on a Mac must confirm.

## What was already portable (audit 2026-09)

- `suppress_console.py` — self-disables on non-Windows (`sys.platform` guard)
- `src/win_job.py` — degrades to plain subprocess without pywin32 (documented
  trade-off: on macOS a killed server can orphan grade children; `pkill -f
  grade_runner` is the manual cleanup, a process-group port is the upgrade)
- `routers/grading.py`, `routers/extras.py`, `scripts/setup_wizard.py`,
  `scripts/benchmark.py`, `scripts/build_engine_cpu.py` — creationflags guarded
- `requirements.txt` — `pywin32` carries a `sys_platform == "win32"` marker
- `src/native_launcher.py` — primary window is pywebview (Cocoa on macOS)

## Fixed in this pass (were hard crashes or silent misbehavior on macOS)

- `routers/export.py`, `src/local_launcher.py` ×2 — unconditional
  `subprocess.CREATE_NO_WINDOW` (AttributeError on POSIX)
- `src/native_launcher.py` — browser discovery now finds /Applications apps;
  isolated profile dirs moved to `platform_compat.app_data_dir()`
- `server_impl.py`, `src/machine_profile.py` — data dirs: macOS uses
  `~/Library/Application Support/FirstCut`
- `scripts/backfill_face_embeddings.py` — RAM fallback guarded to win32

## New in this pass

- `src/platform_compat.py` — the single OS-knowledge module (app dirs,
  no-window flag, browser app-mode, reveal-in-Finder). New code asks it;
  nothing should reach for `ctypes.windll`/`LOCALAPPDATA` directly again.
- `run_macos.sh` — bootstrap: venv, MPS-capable torch, CPU onnxruntime,
  requirements (env markers skip Windows-only deps), frontend build, launch.
- `src/encode_worker.py` — `FIRSTCUT_TORCH_DEVICE=mps` opt-in for Apple
  GPU encoding. Default stays CPU until verified: a wrong device answer
  silently changes every embedding.

## Real-Mac verification checklist (before calling macOS supported)

1. `./run_macos.sh` on Apple Silicon and Intel — venv builds, server binds,
   window opens (pywebview/Cocoa), browser fallback works.
2. Grade a small folder; compare `encoder_source` + spot-check grades against
   a Windows run of the same photos (kernels differ; grades may differ
   marginally — confirm no systematic drift).
3. Kill the server mid-grade; confirm the checkpoint + Resume recover (the
   win_job orphan protection is absent on POSIX by design).
4. `FIRSTCUT_TORCH_DEVICE=mps` bench: encode output must match CPU within
   float tolerance before making MPS default; check the ~4 GB model load
   against macOS unified-memory pressure.
5. YuNet/rawpy/exiftool-equivalents: confirm `face_signals`, RAW decode
   (`.ARW/.RAF/.DNG`), and XMP sidecar writing on APFS.
6. llama-cpp-python source build (needs CMake + Xcode CLT) — jury/critique
   features are dead without it.

## Known not-yet-cross-platform

- Installer: Windows has Setup.ps1 + packaged build; macOS ships as
  `run_macos.sh` (a .app bundle is future work).
- Edge app-mode windowing is Windows-first-class; macOS falls back to
  pywebview (primary) or a normal Chrome/Edge window.
- `scripts/setup_wizard.py` Windows checks (venv layout, PowerShell) — the
  macOS path is `run_macos.sh` instead.
