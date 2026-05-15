"""
Online DPO (Direct Preference Optimization) trainer for the 1.5B draft model.

Each bucket-move event (Mid → Strong, etc.) queues a (winner, loser) preference
pair. A single DPO gradient step is taken per pair using a QLoRA Rank-4 adapter
on top of the INT4 quantized base model.

Reference policy: base model with LoRA adapters disabled (no second model copy).
Adapter footprint: ~6 MB (Rank 4, four attention projections, 28 layers).
VRAM peak during training: ~1.2 GB. Guard: pauses if VRAM > 5.2 GB used.
Adapter persists to models/dpo_adapter/ and is loaded read-only by SpecVLM.
"""
from __future__ import annotations

import gc
import json
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# ── Paths ──────────────────────────────────────────────────────────────────────
ADAPTER_PATH    = Path("models/dpo_adapter")
STYLE_LOG_PATH  = ADAPTER_PATH / "style_log.json"
MODEL_CACHE_DIR = Path("models/deepseek/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B")

# ── Hyperparameters ────────────────────────────────────────────────────────────
LORA_RANK     = 4
LORA_ALPHA    = 8
LORA_DROPOUT  = 0.05
DPO_LR        = 5e-5
DPO_BETA      = 0.1
VRAM_LIMIT_GB = 5.2
MAX_SEQ_LEN   = 256
SLEEP_IDLE    = 30     # seconds to sleep when queue is empty
MAX_BATCH     = 8      # events processed per model-load cycle

# ── Grade → target score ───────────────────────────────────────────────────────
_GRADE_SCORE = {
    "Strong ✅": 0.72,
    "Mid ⚠️":   0.50,
    "Weak ❌":   0.28,
}
_GRADE_RANK = {"Strong ✅": 2, "Mid ⚠️": 1, "Weak ❌": 0}


# ── VRAM helpers ───────────────────────────────────────────────────────────────

def _vram_used_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 3
    return 0.0


def _purge_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _model_input_device(model) -> torch.device:
    """Return the device of the first non-meta parameter."""
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cpu")


# ── Log-prob helper ────────────────────────────────────────────────────────────

def _seq_log_prob(model, tokenizer, text: str) -> torch.Tensor:
    """Sum of per-token log-probabilities for `text` under `model`."""
    enc = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=MAX_SEQ_LEN,
        padding=False,
    )
    ids = enc["input_ids"].to(_model_input_device(model))
    out = model(input_ids=ids, labels=ids)
    seq_len = max(ids.shape[1] - 1, 1)
    return -out.loss * seq_len   # sum log-prob (positive)


# ── Trainer class ──────────────────────────────────────────────────────────────

class BackgroundDPOTrainer:
    """
    Daemon thread that processes bucket-move preference pairs and updates
    a QLoRA Rank-4 adapter on the DeepSeek-R1-Distill-1.5B draft model.

    VRAM protocol: loads model only when VRAM < VRAM_LIMIT_GB, unloads
    and purges after every batch. Never interferes with SpecVLM grading.
    """

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=100)
        self._model    = None
        self._tokenizer = None
        self._optimizer = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="dpo-trainer"
        )
        self._thread.start()

    # ── Public API ─────────────────────────────────────────────────────────────

    def queue_event(self, image_path: str, old_grade: str, new_grade: str) -> None:
        """
        Queue a preference pair derived from a bucket move.

        Generates synthetic JSON completions (winner = higher-score response,
        loser = lower-score response) so the model learns the user's editorial
        direction without needing the original image or prompt replay.
        """
        winner_score = _GRADE_SCORE.get(new_grade, 0.5)
        loser_score  = _GRADE_SCORE.get(old_grade, 0.5)

        if winner_score == loser_score:
            return
        if winner_score < loser_score:
            # Downgrade: swap so winner is always the higher-ranked outcome
            winner_score, loser_score = loser_score, winner_score

        fname  = Path(image_path).name
        prompt = (
            "You are a professional street photo editor.\n"
            f"Grade this image: {fname}\n"
            'Respond as JSON: {"score": <0.0–1.0>, "reasoning_log": "<text>"}'
        )
        winner = (
            f'{{"score": {winner_score:.2f}, '
            f'"reasoning_log": "User-confirmed quality — strong image."}}'
        )
        loser = (
            f'{{"score": {loser_score:.2f}, '
            f'"reasoning_log": "Prior model estimate — lower quality."}}'
        )

        try:
            self._q.put_nowait({
                "prompt":    prompt,
                "winner":    winner,
                "loser":     loser,
                "path":      image_path,
                "old_grade": old_grade,
                "new_grade": new_grade,
            })
        except queue.Full:
            pass  # silently drop when saturated

    # ── Background loop ────────────────────────────────────────────────────────

    def _run(self) -> None:
        while True:
            if self._q.empty():
                time.sleep(SLEEP_IDLE)
                continue
            if _vram_used_gb() > VRAM_LIMIT_GB:
                time.sleep(10)
                continue

            # Drain up to MAX_BATCH events before loading the model
            batch = []
            while not self._q.empty() and len(batch) < MAX_BATCH:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break

            if not batch:
                continue

            try:
                if not self._load():
                    for ev in batch:
                        try: self._q.put_nowait(ev)
                        except queue.Full: pass
                    time.sleep(60)
                    continue

                for ev in batch:
                    if _vram_used_gb() > VRAM_LIMIT_GB:
                        try: self._q.put_nowait(ev)
                        except queue.Full: pass
                        break
                    try:
                        loss = self._step(ev["prompt"], ev["winner"], ev["loser"])
                        print(
                            f"[dpo] loss={loss:.4f}  {Path(ev['path']).name}"
                            f"  ({ev['old_grade']} → {ev['new_grade']})"
                        )
                    except Exception as e_step:
                        print(f"[dpo] step error: {e_step}")

                self._save()
                _update_style_log(batch)

            except Exception as e:
                print(f"[dpo] batch error: {e}")
            finally:
                self._unload()

    # ── Model management ───────────────────────────────────────────────────────

    def _load(self) -> bool:
        """Load INT4 base model + LoRA adapter for training. Returns True on success."""
        if not MODEL_CACHE_DIR.exists():
            print("[dpo] Draft model not cached — DPO skipped.")
            return False

        try:
            from peft import (
                LoraConfig, get_peft_model, PeftModel,
                prepare_model_for_kbit_training,
            )
            from transformers import (
                AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
            )
        except ImportError:
            print("[dpo] peft/transformers unavailable — DPO disabled.")
            return False

        try:
            q_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            device_map = "auto" if torch.cuda.is_available() else "cpu"
            base = AutoModelForCausalLM.from_pretrained(
                str(MODEL_CACHE_DIR),
                quantization_config=q_cfg,
                device_map=device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            base = prepare_model_for_kbit_training(base)

            adapter_ready = (
                ADAPTER_PATH.exists()
                and (ADAPTER_PATH / "adapter_config.json").exists()
            )
            if adapter_ready:
                self._model = PeftModel.from_pretrained(
                    base, str(ADAPTER_PATH), is_trainable=True
                )
                print("[dpo] Existing LoRA adapter loaded for continued training.")
            else:
                lora_cfg = LoraConfig(
                    r=LORA_RANK,
                    lora_alpha=LORA_ALPHA,
                    lora_dropout=LORA_DROPOUT,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self._model = get_peft_model(base, lora_cfg)
                print("[dpo] New LoRA adapter initialised.")

            self._model.train()
            trainable = [p for p in self._model.parameters() if p.requires_grad]
            self._optimizer = torch.optim.Adam(trainable, lr=DPO_LR)

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(MODEL_CACHE_DIR), trust_remote_code=True, use_fast=True,
            )
            return True

        except Exception as e:
            print(f"[dpo] Load failed: {e}")
            self._unload()
            return False

    def _step(self, prompt: str, winner: str, loser: str) -> float:
        """Single DPO gradient update for one (winner, loser) pair. Returns loss."""
        assert self._model is not None and self._optimizer is not None

        full_w = prompt + "\n" + winner
        full_l = prompt + "\n" + loser

        # Reference log-probs: base model only (LoRA disabled = straight-through reference)
        self._model.disable_adapter_layers()
        with torch.no_grad():
            ref_lp_w = _seq_log_prob(self._model, self._tokenizer, full_w)
            ref_lp_l = _seq_log_prob(self._model, self._tokenizer, full_l)
        self._model.enable_adapter_layers()

        # Policy log-probs: base + LoRA (with gradients for LoRA params)
        self._optimizer.zero_grad()
        pi_lp_w = _seq_log_prob(self._model, self._tokenizer, full_w)
        pi_lp_l = _seq_log_prob(self._model, self._tokenizer, full_l)

        # DPO objective: -log σ(β * ((log π/π_ref)_winner - (log π/π_ref)_loser))
        log_ratio = (pi_lp_w - ref_lp_w.detach()) - (pi_lp_l - ref_lp_l.detach())
        loss = -F.logsigmoid(DPO_BETA * log_ratio)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self._model.parameters() if p.requires_grad], 1.0
        )
        self._optimizer.step()
        return float(loss.item())

    def _save(self) -> None:
        if self._model is None:
            return
        ADAPTER_PATH.mkdir(parents=True, exist_ok=True)
        try:
            self._model.save_pretrained(str(ADAPTER_PATH))
        except Exception as e:
            print(f"[dpo] Adapter save failed: {e}")

    def _unload(self) -> None:
        self._model     = None
        self._tokenizer = None
        self._optimizer = None
        _purge_vram()


# ── Style log ──────────────────────────────────────────────────────────────────

def _update_style_log(events: list[dict]) -> None:
    """Accumulate user preference statistics for system-prompt injection."""
    ADAPTER_PATH.mkdir(parents=True, exist_ok=True)
    log: dict = {}
    if STYLE_LOG_PATH.exists():
        try:
            log = json.loads(STYLE_LOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    for ev in events:
        old_r = _GRADE_RANK.get(ev["old_grade"], 1)
        new_r = _GRADE_RANK.get(ev["new_grade"], 1)
        if new_r > old_r:
            log["upgrade_count"] = log.get("upgrade_count", 0) + 1
        elif new_r < old_r:
            log["downgrade_count"] = log.get("downgrade_count", 0) + 1
    log["total_events"] = log.get("total_events", 0) + len(events)

    up  = log.get("upgrade_count",   0)
    dn  = log.get("downgrade_count", 0)
    tot = log.get("total_events",    0)

    if tot >= 5:
        if up > dn * 1.5:
            note = "User tends to upgrade borderline shots — reward potential and be generous."
        elif dn > up * 1.5:
            note = "User is strict — only clearly excellent shots get Strong; penalise weak moments."
        else:
            note = "User applies balanced, decisive editorial judgement."
    else:
        note = ""
    log["style_note"] = note

    STYLE_LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")


def load_style_instruction() -> str:
    """Return the accumulated style note for prompt injection (empty string if none)."""
    try:
        if STYLE_LOG_PATH.exists():
            log = json.loads(STYLE_LOG_PATH.read_text(encoding="utf-8"))
            return log.get("style_note", "")
    except Exception:
        pass
    return ""


# ── Singleton ──────────────────────────────────────────────────────────────────

_trainer: Optional[BackgroundDPOTrainer] = None
_trainer_lock = threading.Lock()


def get_trainer() -> BackgroundDPOTrainer:
    """Return the global BackgroundDPOTrainer singleton (lazy-init, thread-safe)."""
    global _trainer
    if _trainer is None:
        with _trainer_lock:
            if _trainer is None:
                _trainer = BackgroundDPOTrainer()
    return _trainer
