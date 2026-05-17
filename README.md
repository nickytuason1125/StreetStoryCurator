# Street Story Curator

AI-powered street photography editor. Grades, sequences, and curates your photos using a local vision pipeline — no cloud, no subscription, fully offline after setup.

![Grade + Sequence](https://img.shields.io/badge/runs%20offline-✓-brightgreen) ![Python 3.12](https://img.shields.io/badge/python-3.12-blue) ![Windows](https://img.shields.io/badge/platform-Windows-lightgrey)

---

## What it does

- **Grades** every photo in a folder — Strong / Mid / Weak — using SigLIP-2 embeddings + UniQA vision quality
- **Sequences** your best shots into a story arc with NSGA-III multi-objective optimisation
- **Creative direction** via Phi-4-mini reasoning: editorial feedback tuned to your shooting brief
- **Learns your taste** through a PersonalHead MLP that adapts from your grade corrections
- Runs entirely on your machine — no internet needed after first setup

---

## Quick Install — Windows

> Requires **Python 3.12**, **Node.js LTS**, and **Git**. See [Prerequisites](#prerequisites) below.

1. Click **Code → Download ZIP** on this page, extract anywhere (e.g. Desktop)
2. Open the extracted folder
3. Double-click **`Setup.bat`**

The wizard opens, checks your system, installs everything, and places a shortcut on your Desktop. First run downloads ~4 GB of libraries and AI models — takes 10–20 minutes depending on your connection. Every launch after that is instant.

**GPU users** (NVIDIA): the wizard auto-detects your GPU and installs the CUDA-accelerated build of PyTorch. Grading a folder of 50 photos takes ~30 seconds on a GTX 1070 or newer.

**CPU-only**: works fine — grading is slower (~3–5 min per 50 photos).

---

## Prerequisites

Install these three free programs before running Setup:

| Program | Why | Download |
|---|---|---|
| **Python 3.12** | Runs the app engine | https://www.python.org/downloads/release/python-31210/ |
| **Node.js LTS** | Builds the interface (first run only) | https://nodejs.org |
| **Git** | Downloads AI model weights | https://git-scm.com/download/win |

**Python install tip:** tick **"Add Python to PATH"** during the Python installer — it won't work without it.

---

## After Install

- Launch via the **Street Story Curator** shortcut on your Desktop
- Or double-click `run_local.bat` inside the folder

---

## AI Stack (Frontier 2026)

| Component | Model | Purpose |
|---|---|---|
| Embedding | SigLIP-2 ViT-g/14 NaFlex (1536-d) | Semantic similarity + brief alignment |
| Composition | YOLO11s-seg | Person detection + OTS portrait routing |
| Quality | UniQA (pyiqa) | Unified technical + aesthetic score |
| Chiaroscuro | DINOv2 ViT-S/14 | Intentional shadow/highlight detection |
| Sequencing | NSGA-III (pymoo) | Multi-objective story arc optimisation |
| Reasoning | Phi-4-mini-reasoning (GGUF) | Purist editorial feedback |
| Preference | PersonalHead MLP 1536→256→1 | Online taste learning via DPO |

All models run locally. Model weights are downloaded on first launch via HuggingFace / ultralytics (cached in `~/.cache/huggingface` and `models/`).

---

## Folder Layout

```
street-story-curator/
├── Setup.bat              ← run this to install
├── run_local.bat          ← launch after install
├── server.py              ← FastAPI backend
├── src/                   ← pipeline modules
│   ├── grade_pipeline_v2.py
│   ├── vision_grading_heads.py
│   ├── vision_composition_heads.py
│   ├── specvlm_pipeline.py
│   └── ...
├── frontend/              ← React/Vite/TypeScript UI
├── models/                ← YOLO weights (gitignored, downloaded at runtime)
└── cache/                 ← LanceDB + catalog (gitignored)
```

---

## Updating

```
git pull
```

Then launch normally — the setup stamp (`venv/.setup_ok`) means the installer won't re-run. If a dependency changed, delete `venv/.setup_ok` and run `Setup.bat` again.

---

## License

MIT
