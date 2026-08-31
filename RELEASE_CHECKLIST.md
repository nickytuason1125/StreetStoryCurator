# FirstCut Release Checklist

The complete path from source tree to a public download page. Items marked
● require a decision or purchase; everything else is a command.

## 1. Pre-build (5 min)

- [ ] `venv\Scripts\python.exe -m pytest tests\ -q` — full suite green
- [ ] `cd frontend && npm test` — frontend gates green
- [ ] Version bump if needed: `tauri.conf.json` version + `Cargo.toml` version
- [ ] Disk check: **≥ 12 GB free on C:** (engine bundle + NSIS compression head-room)

## 2. Build the engine (10–20 min)

- [ ] `venv\Scripts\pyinstaller.exe FirstCut.spec --noconfirm`
- [ ] Output: `dist\FirstCut\` (~12.5 GB — CUDA torch, FastAPI, all deps)
- [ ] Smoke the engine exe standalone if paranoid: run it, probe :8000/api/health

## 3. Stage the engine into the Tauri resources (5 min)

```powershell
robocopy "dist\FirstCut" `
  "frontend\src-tauri\resources\binaries\curator-api-x86_64-pc-windows-msvc" `
  /E /MOVE /NFL /NDL /NJH /NP
Rename-Item "frontend\src-tauri\resources\binaries\curator-api-x86_64-pc-windows-msvc\FirstCut.exe" "curator-api.exe"
```

(The /MOVE frees the dist copy; the resources copy is what the installer packs.)

## 4. Build the installer (20–40 min — LZMA on 12.5 GB)

- [ ] `cd frontend && npm run tauri:build`
- [ ] Output: `src-tauri\target\release\bundle\nsis\FirstCut_1.0.0_x64-setup.exe`
- [ ] Sanity: installer should be **several GB**, not single-digit MB (single-digit = engine missing)

## 5. Sign (● requires the certificate)

- [ ] `scripts\sign_release.ps1 -Mode Store -Thumbprint <SHA1>`
      or `-Mode TrustedSigning -Account ... -Profile ... -DmdfPath ...`
- [ ] Verify: `signtool verify /pa /all <installer>`
- Without a cert: skip, but every user gets the SmartScreen 2-click bypass
  ("More info → Run anyway") — acceptable for beta, not for public launch

## 6. Install test (30 min)

- [ ] Run the installer (double-click, or `/S` for silent)
- [ ] First-run: welcome → engine boots → model banner (if any missing) → download → grade a folder → star → export
- [ ] `scripts\launch_check.ps1 -Runs 3` — no browser tabs, isolated webview, health green
- [ ] Uninstall → reinstall → repeat once (clean-cycle proof)

## 7. Distribute

- [ ] Upload `FirstCut_1.0.0_x64-setup.exe` + SHA-256 checksum
- [ ] Download page text: system requirements (Windows 10+, 8 GB RAM min, GPU
      recommended), offline statement, privacy statement (nothing leaves the machine)
- [ ] Beta users: include the SmartScreen 2-click note until reputation builds

## 8. Post-release

- [ ] `scripts\first_run_smoke.py` after every future engine change
- [ ] MasterJudge refits ride the champion/challenger gate automatically
- [ ] Winget manifest + Microsoft Store — the two channels that bypass
      SmartScreen friction entirely — are the natural v1.1 distribution moves
