# Building Lumara for Distribution

## Prerequisites (all platforms)

```bash
# 1. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 2. Install runtime dependencies
pip install -r requirements.txt

# 3. Install CLIP (not on PyPI — install from GitHub)
pip install git+https://github.com/openai/CLIP.git

# 4. Install PyInstaller
pip install pyinstaller
```

> **Important:** PyInstaller must be installed inside the same `venv` that contains
> all project dependencies. Building with the system Python will produce a broken bundle.

---

## Windows

```bash
pyinstaller Lumara.spec
```

Output: `dist/Lumara.exe`

### VS Code shortcut

Press **`Ctrl+Shift+B`** to run the default *Build Executable (PyInstaller)* task defined
in `.vscode/tasks.json`.

---

## macOS

```bash
pyinstaller Lumara.spec
```

Output: `dist/Lumara` (single binary) or `dist/Lumara.app`
(app bundle — omit `--onefile` to produce the `.app` directory).

> macOS Gatekeeper will quarantine unsigned binaries. To let users run them:
> ```bash
> xattr -cr dist/Lumara.app
> codesign --force --deep --sign - dist/Lumara.app
> ```
> For distribution outside the App Store an Apple Developer ID is required.

---

## Linux

```bash
pyinstaller Lumara.spec
```

Output: `dist/Lumara` (ELF binary, no extension)

> Linux builds require `python3-gi`, `gir1.2-webkit2-4.0`, and `libgtk-3-dev` for
> pywebview's GTK backend. Install with:
> ```bash
> sudo apt install python3-gi gir1.2-webkit2-4.0 libgtk-3-dev
> ```

---

## Attaching binaries to GitHub Releases

### Option A — GitHub CLI (`gh`)

```bash
# Tag the release
git tag v1.0.0
git push origin v1.0.0

# Create the release and attach the binary in one command
gh release create v1.0.0 dist/Lumara.exe \
  --title "Lumara v1.0.0" \
  --notes "First public release. Offline street photo grading & sequencing."
```

To attach multiple platform binaries to the **same** release
(after building separately on each OS):

```bash
# From Windows — attach .exe
gh release upload v1.0.0 dist/Lumara.exe

# From macOS — attach macOS binary
gh release upload v1.0.0 "dist/Lumara-macos"

# From Linux — attach Linux binary
gh release upload v1.0.0 "dist/Lumara-linux"
```

### Option B — GitHub Actions (automated, recommended)

Create `.github/workflows/release.yml` to build on all three runners and
upload assets automatically when a tag is pushed. Each job runs:

```yaml
- uses: actions/checkout@v4
- uses: actions/setup-python@v5
  with: { python-version: "3.11" }
- run: pip install -r requirements.txt
        git+https://github.com/openai/CLIP.git
        pyinstaller Lumara.spec
- uses: actions/upload-release-asset@v1
  with:
    asset_path: dist/Lumara${{ matrix.ext }}
```

---

## Cross-platform limitations

| Limitation | Detail |
|---|---|
| **Must build on target OS** | PyInstaller bundles the OS's own Python runtime and shared libraries. A Windows build **cannot** produce a macOS or Linux binary, and vice versa. |
| **Architecture** | An x86-64 build will not run on Apple Silicon (arm64) unless the Python and all C-extensions (torch, cv2) are also arm64. Build on an M-series Mac for arm64. |
| **PyTorch size** | The bundled `.exe` / binary will be **1–2 GB** due to PyTorch. Use `--exclude-module` to strip unused torch backends if size is a concern. |
| **CLIP weights not bundled** | Model weights (~330 MB) are **not** included in the binary. On first run the app downloads them to `./models/` or falls back to `~/.cache/clip/`. Ship a separate installer or advise users to pre-run the app once with network access. |
| **Windows Defender** | Unsigned one-file executables may be flagged. Code-sign with a trusted certificate for production distribution. |
| **macOS notarisation** | From macOS 10.15+ all distributed binaries must be notarised by Apple or Gatekeeper will block them for end-users. |
| **Linux display server** | pywebview requires an X11 or Wayland session. Headless / SSH environments will fail to open the window. |
