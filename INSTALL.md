# Street Story Curator — Installation Guide

Street Story Curator is a desktop app for grading, sequencing, and curating street photography. It runs fully offline on your computer — no internet required after setup.

---

## What You Need First

Before installing the app, make sure these two programs are on your computer. Both are free.

| Program | Why | Download |
|---------|-----|----------|
| **Python 3.12** | Runs the app engine | https://www.python.org/downloads/release/python-31210/ |
| **Git** | Required to install the AI grading model | https://git-scm.com/download/win |
| **Node.js (LTS)** | Builds the interface (first run only) | https://nodejs.org |

### Installing Python (Windows)
1. Go to https://www.python.org/downloads/release/python-31210/ and click **Download Python 3.12**
2. Run the installer
3. **Important:** Tick the box that says **"Add Python to PATH"** before clicking Install
4. Click Install Now

### Installing Python (Mac)
1. Go to https://www.python.org/downloads/ and download the macOS installer
2. Run the `.pkg` file and follow the steps

### Installing Git (Windows)
1. Go to https://git-scm.com/download/win and download the installer
2. Run it with all default settings

### Installing Node.js
1. Go to https://nodejs.org and click **Download LTS**
2. Run the installer with all default settings

---

## Windows — Installation

1. Download and unzip the `street-story-curator` folder anywhere on your computer (Desktop is fine)
2. Open the folder
3. Double-click **`Install Shortcut.bat`**
   - This places a **Street Story Curator** shortcut on your Desktop
4. Double-click the **Street Story Curator** shortcut on your Desktop

**On first launch only**, the app will automatically download and install its libraries (~1.5 GB). This takes about 5–10 minutes depending on your internet speed. A window will show the progress. Once it finishes, the app opens and every launch after this is instant.

---

## Mac — Installation

1. Download and unzip the `street-story-curator` folder anywhere on your computer
2. Open **Terminal** (search for it in Spotlight with `Cmd + Space`)
3. Run this command to make the launcher executable (do this once only):
   ```
   chmod +x ~/Desktop/street-story-curator/Start.command
   ```
   Replace `~/Desktop/street-story-curator/` with the actual path if you put it somewhere else.
4. In Finder, open the `street-story-curator` folder and double-click **`Start.command`**

**On first launch only**, the app installs its libraries (~1.5 GB). This takes about 5–10 minutes. The Terminal window shows the progress. Once it finishes, the app opens. Every launch after this is instant.

> **Mac security warning:** If macOS says "Start.command cannot be opened because it is from an unidentified developer", right-click the file and choose **Open**, then click **Open** again in the dialog.

---

## After Installation

- **Windows:** Double-click the **Street Story Curator** shortcut on your Desktop
- **Mac:** Double-click **`Start.command`** inside the project folder (or drag it to your Dock)

The app opens a window with a dark interface. From there:

1. Click **Browse** or paste a folder path to load your photos
2. Click **Grade Folder** to analyse and grade your images
3. Use the **Sequence** tab to build and save story sequences

---

## Troubleshooting

**The app doesn't open / nothing happens**
- Make sure Python was installed with "Add Python to PATH" checked
- Try running `Start.bat` (Windows) or `Start.command` (Mac) directly — it will show any error messages

**"Python is not installed" message on Windows**
- Uninstall Python and reinstall, making sure to tick "Add Python to PATH"

**First-time setup fails with a network error**
- Check your internet connection — the first launch downloads libraries
- Try again; the download resumes where it left off

**App opens but photos don't display**
- Make sure the folder you selected contains `.jpg`, `.jpeg`, `.png`, `.arw`, `.cr2`, or `.nef` files

**Mac: "Operation not permitted" error**
- Go to System Settings → Privacy & Security → Files and Folders and grant Terminal access

---

## Uninstalling

1. Delete the `street-story-curator` folder
2. Delete the **Street Story Curator** shortcut from your Desktop (Windows)
3. That's it — nothing is installed system-wide
