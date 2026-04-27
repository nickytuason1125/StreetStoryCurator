import os, requests, threading
from pathlib import Path

MODEL_DIR = Path("models/onnx")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Lock to prevent race conditions during model download
_MODEL_DOWNLOAD_LOCK = threading.Lock()

def download(url, dest):
    if not dest.exists():
        print(f"📦 Downloading {dest.name}...")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

def get_sessions():
    with _MODEL_DOWNLOAD_LOCK:
        # DINOv2-small INT8 quantized: 23 MB — composition features & embeddings (onnx-community, public)
        # model.onnx is graph-only (external data); model_quantized.onnx is fully self-contained
        comp_url = "https://huggingface.co/onnx-community/dinov2-small/resolve/main/onnx/model_quantized.onnx"
        # MobileViT-small quantized: 21 MB — aesthetic/quality proxy, CPU-light (Xenova, public)
        aesthetic_url = "https://huggingface.co/Xenova/mobilevit-small/resolve/main/onnx/model.onnx"

        comp_path = MODEL_DIR / "dinov2_small.onnx"
        est_path  = MODEL_DIR / "mobilevit_aesthetic.onnx"

        download(comp_url, comp_path)
        download(aesthetic_url, est_path)

    import onnxruntime as ort   # deferred — keeps module import instant

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads        = os.cpu_count() or 1
    sess_options.inter_op_num_threads        = 1
    sess_options.graph_optimization_level    = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode              = ort.ExecutionMode.ORT_SEQUENTIAL

    providers = ["CPUExecutionProvider"]
    sessions = {
        "composition": ort.InferenceSession(str(comp_path), sess_options=sess_options, providers=providers),
        "aesthetic":   ort.InferenceSession(str(est_path),  sess_options=sess_options, providers=providers),
    }

    nima_path = MODEL_DIR / "nima.onnx"
    if nima_path.exists():
        try:
            sessions["nima"] = ort.InferenceSession(str(nima_path), sess_options=sess_options, providers=providers)
        except Exception as e:
            print(f"Warning: NIMA ONNX found but failed to load ({e}). Skipping.")

    return sessions
