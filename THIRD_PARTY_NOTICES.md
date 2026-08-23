# Third-party notices

FrameGrade is assembled from third-party libraries and pre-trained model weights.
This file inventories them. It exists because the app is headed for public
download, and a download distributes — or causes the user's machine to fetch —
every item below.

**Status: the library column is settled. The model weights column is not.**
Model licences are the part that can actually stop a release, and several of the
ones here are more restrictive than their libraries. Nothing in this file should
be read as legal advice; the items marked **VERIFY** need a decision before the
first public build.

---

## Blocking issue — resolve before any public release

### Ultralytics YOLO is AGPL-3.0

`ultralytics` is imported by five modules:

- `src/dfine_detector.py`
- `src/iqa_worker.py`
- `src/vision_grading_heads.py`
- `src/yolo_auditor.py`
- `src/vision_composition_heads.py`

AGPL-3.0 is a strong copyleft licence. Distributing a closed-source application
that links it obliges you to offer the complete corresponding source of the whole
work under the same terms. This is not a formality that a notices file discharges.

Three ways out, in the order I would consider them:

1. **Drop YOLO for D-FINE.** `models/dfine_nano` is already installed and
   `src/dfine_detector.py` already wraps it. D-FINE is Apache-2.0. This is the
   cheapest exit if the YOLO paths are refinements rather than load-bearing.
2. **Buy an Ultralytics Enterprise licence.** Keeps the code as-is; costs money.
3. **Open-source FrameGrade under AGPL-3.0.** Free, and forecloses a closed
   commercial release later.

This is a business decision, not an engineering one, so it is flagged rather than
resolved here.

---

## Model weights

These are not bundled in the installer — the first-run downloader fetches them
from Hugging Face onto the user's machine. That distinction matters legally
(the user obtains them from the upstream host, under the upstream terms) but it
does not remove the obligation to be accurate about what the app requires.

| Model | Used for | Upstream | Licence |
|---|---|---|---|
| SigLIP-2 ViT-gopt-16-384 | Pro-tier image + text embeddings | `timm/ViT-gopt-16-SigLIP2-384` | Apache-2.0 |
| SigLIP-2 ViT-L-16-384 | Balanced-tier embeddings | `timm/ViT-L-16-SigLIP2-384` | Apache-2.0 |
| SigLIP-2 ViT-B-16-384 | Fast-tier embeddings (the CPU path) | `timm/ViT-B-16-SigLIP2-384` | Apache-2.0 |
| CLIP ViT-B/32 | niche auto-detect, Story/Competition vectors | `openai/CLIP` | MIT |
| D-FINE nano | person / subject detection | D-FINE authors | Apache-2.0 |
| YuNet face detector | face and subject-focus signals | OpenCV Zoo | MIT |
| YOLO11s / YOLO26n | detection and audit heads | Ultralytics | **AGPL-3.0 — see above** |
| Qwen2.5-VL-3B-Instruct | opt-in Deep Grade, annotations | Alibaba | **VERIFY** |
| Qwen3-4B (GGUF) | Story/Competition selection, Judge's Verdict, RAG extraction | Alibaba, via `bartowski/Qwen_Qwen3-4B-GGUF` | Apache-2.0 |
| TOPIQ NR (via `pyiqa`) | technical quality head | `chaofengc/IQA-PyTorch` | **VERIFY** |
| DINOv2-ViT-S/14 | ChiaroscuroHead vision probe | Meta | **VERIFY** |

**On the VERIFY rows.** Each of these needs its model card read before release,
for a specific reason:

- **Qwen2.5-VL-3B-Instruct** — the Qwen 2.5-VL family does not use one licence
  across sizes; some sizes ship under a research-only licence while others are
  Apache-2.0. Confirm which applies to the 3B Instruct checkpoint specifically.
  If it is research-only, Deep Grade cannot ship as a commercial feature; grading
  is unaffected, because SigLIP zero-shot is the default path and Qwen is opt-in.
- **DeepSeek-R1-Distill was REMOVED on 2026-08-22** and its weights deleted, so
  this row is gone rather than resolved. It was replaced by Qwen3-4B, which is
  cleanly Apache-2.0 and needs no verification — one fewer licence question at
  release. The reason for the swap was not licensing: on the 16 GB target laptop
  the 8B never loaded at all (needs ~6.6 GB free; 2.3-4.0 GB was measured), so
  Story mode had been silently falling back to a score sort.
- **TOPIQ / pyiqa** — the `pyiqa` package and the pre-trained IQA weights it
  downloads are licensed separately, and several IQA checkpoints in that ecosystem
  are non-commercial. TOPIQ is on the default grading path, so this one is not
  optional.
- **DINOv2** — check the checkpoint's terms, not just the DINOv2 repository's.

---

## Python libraries

Installed by `requirements.txt` and the installer. Permissive throughout; no
copyleft in this list.

| Package | Licence |
|---|---|
| PyTorch, torchvision | BSD-3-Clause |
| transformers, tokenizers, accelerate, peft, safetensors, huggingface-hub, timm | Apache-2.0 |
| open_clip_torch | MIT |
| onnxruntime | MIT |
| FastAPI, uvicorn, pywebview | MIT |
| lancedb, pyarrow | Apache-2.0 |
| pymoo | Apache-2.0 |
| scipy, scikit-learn, numpy | BSD-3-Clause |
| opencv-python | Apache-2.0 |
| Pillow | MIT-CMU |
| pillow-heif | BSD-3-Clause |
| rawpy | MIT (wraps LibRaw — LGPL-2.1 / CDDL dual) |
| PyTurboJPEG | MIT (wraps libjpeg-turbo — IJG / BSD) |
| pypdfium2 | Apache-2.0 / BSD-3-Clause |
| llama-cpp-python | MIT |
| bitsandbytes | MIT |
| torchao | BSD-3-Clause |
| duckdb | MIT |
| piexif, exifread | MIT |
| requests | Apache-2.0 |
| psutil | BSD-3-Clause |
| pywin32 | PSF |
| openai-clip | MIT |

`pypdfium2` was chosen over PyMuPDF specifically to avoid PyMuPDF's
AGPL-3.0/commercial dual licence. Keep it that way.

---

## Bundled sample images

`dataset_images/` contains 100 JPEGs used by `scripts/build_lean_checkpoint.py`
and `scripts/setup_siglip2_hf.py` to validate a converted checkpoint against
known-good reference embeddings, and as the default input for
`curator_pipeline.py` and `essay_selector.py`.

These appear to be the maintainer's own photographs. **Publishing the repository
or shipping the release zip publishes them.** If that is not intended, replace
them with a handful of licence-clear images — the validators need only eight.
