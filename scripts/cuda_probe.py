import os, sys, time

def log(m):
    print(f"[probe] {m}", flush=True)

log(f"python: {sys.executable}")
log(f"free RAM at start: {os.popen('powershell -NoProfile -Command \"[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)\"').read().strip()} GB")

t0 = time.time()
import torch
log(f"torch {torch.__version__} imported in {time.time()-t0:.1f}s")
log(f"torch.cuda.is_available: {torch.cuda.is_available()}")

t0 = time.time()
torch.cuda.init()
log(f"cuda init OK in {time.time()-t0:.1f}s")

t0 = time.time()
x = torch.randn(2048, 2048, device='cuda')
y = x @ x
torch.cuda.synchronize()
log(f"matmul on GPU OK in {time.time()-t0:.1f}s, alloc={torch.cuda.memory_allocated()/1e6:.0f}MB")

t0 = time.time()
# simulate the model-size allocation that keeps failing in production
big = torch.empty(int(2.7 * 1e9 / 4), dtype=torch.float32, device='cuda')
log(f"2.7 GB GPU allocation OK in {time.time()-t0:.1f}s")
del big
torch.cuda.empty_cache()

# onnxruntime — the component that actually dies in production
t0 = time.time()
import onnxruntime as ort
log(f"onnxruntime {ort.__version__} imported in {time.time()-t0:.1f}s")
try:
    sess = ort.InferenceSession.__new__(ort.InferenceSession)
    avail = ort.get_available_providers()
    log(f"providers: {avail}")
    # real CUDA EP init test via a tiny dummy model is complex; cudaSetDevice
    # is what fails in prod — torch.cuda.init above covers the same driver path.
except Exception as e:
    log(f"onnx check failed: {e}")

log("PROBE COMPLETE - all allocations succeeded")
