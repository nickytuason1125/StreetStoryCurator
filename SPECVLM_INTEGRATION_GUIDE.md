# SpecVLM Integration Guide

## Overview

This guide shows how to integrate the new SpecVLM pipeline into the Street Story Curator codebase while maintaining backward compatibility with existing code.

## New Modules Created

| Module | Purpose |
|--------|---------|
| [`specvlm_pipeline.py`](street-story-curator/src/specvlm_pipeline.py) | Draft-and-Verify pipeline with Priority-Gate Controller |
| [`siglip2_encoder.py`](street-story-curator/src/siglip2_encoder.py) | SigLIP-2 ViT-g/14 NaFlex encoder (FP8, 1536-d) |
| [`priority_gate.py`](street-story-curator/src/priority_gate.py) | Confidence-based gate for speculative decoding |
| [`vram_manager.py`](street-story-curator/src/vram_manager.py) | VRAM cleanup utilities for laptop GPUs |
| [`nsga3_sequencer.py`](street-story-curator/src/nsga3_sequencer.py) | NSGA-III multi-objective photo sequencer |
| [`tensorrt_engine.py`](street-story-curator/src/tensorrt_engine.py) | TensorRT-LLM inference engine |
| [`lance_migration.py`](street-story-curator/src/lance_migration.py) | Embedding migration to LanceDB IVF-PQ |
| [`deepseek_model.py`](street-story-curator/src/deepseek_model.py) | DeepSeek-R1 model wrappers |

## Usage Examples

### 1. Basic SpecVLM Grading

```python
from specvlm_pipeline import SpecVLMPipeline

# Create pipeline instance
pipeline = SpecVLMPipeline()

# Grade a single image
result = pipeline.grade_images(["path/to/image.jpg"])

# Access results
print(f"Score: {result[0].score}")
print(f"Confidence: {result[0].confidence}")
print(f"Is verified: {result[0].is_verified}")
print(f"Reasoning: {result[0].reasoning_log}")
```

### 2. Using SigLIP-2 Encoder

```python
from siglip2_encoder import SigLIP2Encoder

# Create encoder (native aspect ratio)
encoder = SigLIP2Encoder()

# Encode multiple images
embeddings = encoder.encode_images([
    "img1.jpg",
    "img2.jpg",
    "img3.jpg",
])

# embeddings.shape = (3, 1536) - 1536-d embeddings (vs 1152-d for So400M)
```

### 3. Priority-Gate Controller

```python
from priority_gate import PriorityGate, AdaptivePriorityGate

# Basic priority gate
gate = PriorityGate(threshold=0.88)

if gate.should_skip(confidence=0.92):
    print("Skip verification - draft is confident enough")
else:
    print("Trigger 7B verifier for correction")

# Adaptive priority gate (adjusts based on VRAM)
adaptive_gate = AdaptivePriorityGate()
adaptive_gate.adjust_for_vram(vram_available=2_000_000_000, vram_total=6_000_000_000)
```

### 4. VRAM Manager

```python
from vram_manager import VRAMManager

# Clear VRAM between phases
VRAMManager.clear_between_phases()

# Check VRAM usage
usage = VRAMManager.get_vram_usage()
print(f"Allocated: {usage['allocated'] / 1e9:.2f}GB")
print(f"Available: {usage['total'] - usage['allocated'] / 1e9:.2f}GB")

# Ensure sufficient VRAM before loading model
if VRAMManager.ensure_sufficient_vram(min_required=3_000_000_000):
    # Safe to load model
    pass
```

### 5. NSGA-III Sequencer

```python
from nsga3_sequencer import run_nsga3_sequence

# Prepare candidates with embeddings and VLM scores
candidates = [
    {"path": "img1.jpg", "score": 0.85, "embedding": emb1, "reasoning_log": "..." },
    {"path": "img2.jpg", "score": 0.78, "embedding": emb2, "reasoning_log": "..." },
    # ...
]

# Optimize sequence
sequence = run_nsga3_sequence(candidates, target=5)

# 4 objectives: Reasoning_Accuracy, Semantic_Vibe, Portfolio_Diversity, Aspect_Ratio_Balance
```

### 6. LanceDB Migration

```python
from lance_migration import migrate_from_sqlite, migrate_from_faiss, create_ivf_pq_index

# Migrate from SQLite
migrate_from_sqlite("cache/legacy.db", batch_size=1000)

# Create IVF-PQ index for fast search
create_ivf_pq_index(num_partitions=16, num_sub_vectors=96)

# Search by embedding
results = search_by_embedding(query_embedding, k=10, nprobes=10)
```

## Migration Path

### Phase 1: Integrate SpecVLM Pipeline (Current)

1. Update `server.py` to use SpecVLM instead of Q-Align
2. Update `grade_pipeline_v2.py` to use SigLIP-2 encoder
3. Keep `lightweight_analyzer.py` for fallback grading

### Phase 2: Remove Legacy Models

1. Remove DINOv2, MobileViT, NIMA from `model_loader.py`
2. Update `lightweight_analyzer.py` to remove ONNX model references
3. Clean up import statements

### Phase 3: Full SpecVLM Integration

1. Replace `grade_pipeline_v2.py` with SpecVLM pipeline
2. Use SigLIP-2 embeddings throughout
3. Update grading thresholds based on new score distribution

## Backward Compatibility

The existing `lightweight_analyzer.py` remains unchanged and can be used alongside the new SpecVLM pipeline:

```python
# Old way (still works)
from lightweight_analyzer import LightweightStreetScorer
old_scanner = LightweightStreetScorer()
old_result = old_scanner._analyze("path/to/image.jpg")

# New way (SpecVLM)
from specvlm_pipeline import SpecVLMPipeline
new_pipeline = SpecVLMPipeline()
new_result = new_pipeline.grade_images(["path/to/image.jpg"])
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Inference per image | <800ms (vs ~2.5s) |
| VRAM usage | ~3.5GB (vs ~4.5GB) |
| Sequence generation | ~8s (vs ~15s) |
| Embedding dimension | 1536-d (vs 1152-d) |

## Troubleshooting

### CUDA Out of Memory
```python
from vram_manager import VRAMManager
VRAMManager.clear_between_phases()
```

### Slow Inference
- Verify TensorRT engine is built
- Check batch size (use 1-4 for laptop GPUs)
- Enable fused attention in TensorRT engine

### Grade Distribution Shift
- New 0.0-1.0 scale based on VLM reasoning
- Adjust thresholds: Strong >0.65, Mid 0.40-0.65, Weak <0.40
- Review `SpecVLMResult` confidence scores for low-confidence cases

## Deployment Checklist

- [ ] Install new dependencies (`requirements.txt` updated)
- [ ] Download SpecVLM models (DeepSeek-R1-Distill-Qwen-1.5B, 7B)
- [ ] Build TensorRT engines (optional, for sub-800ms inference)
- [ ] Run embedding migration to LanceDB IVF-PQ
- [ ] Update server endpoints to use SpecVLM
- [ ] Test with sample images
- [ ] Adjust grading thresholds if needed
- [ ] Deploy with fallback to legacy analyzer
