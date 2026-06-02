"""
Online DPO (Direct Preference Optimization) trainer for the 1.5B draft model.

Preferences are persisted to SQLite (dpo_prefs.db) instead of an in-memory queue.
Training fires only when ≥ 20 untrained preferences have accumulated, mixing the
20 newest untrained rows with 5 random historical trained rows to prevent
catastrophic forgetting.

Reference policy: base model with LoRA adapters disabled (no second model copy).
Adapter footprint: ~6 MB (Rank 4, four attention projections, 28 layers).
VRAM peak during training: ~1.2 GB. Guard: pauses if VRAM > 5.2 GB used.
Adapter persists to models/dpo_adapter/ and is loaded read-only by SpecVLM.
"""
from __future__ import annotations

import gc
import json
import os
import sqlite3
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
LORA_RANK        = 4
LORA_ALPHA       = 8
LORA_DROPOUT     = 0.05
DPO_LR           = 5e-5
DPO_BETA         = 0.1
VRAM_LIMIT_GB    = 5.2
MAX_SEQ_LEN      = 256
SLEEP_IDLE       = 30      # seconds to sleep when batch threshold not met
BATCH_THRESHOLD  = 20      # minimum untrained rows before training fires
HISTORY_MIX      = 5       # historical trained rows mixed in for replay

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
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cpu")


# ── Log-prob helper ────────────────────────────────────────────────────────────

def _seq_log_prob(model, tokenizer, text: str) -> torch.Tensor:
    enc = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=MAX_SEQ_LEN,
        padding=False,
    )
    ids = enc["input_ids"].to(_model_input_device(model))
    out = model(input_ids=ids, labels=ids)
    seq_len = max(ids.shape[1] - 1, 1)
    return -out.loss * seq_len


# ── Trainer class ──────────────────────────────────────────────────────────────

class BackgroundDPOTrainer:
    """
    Daemon thread that processes bucket-move preference pairs and updates
    a QLoRA Rank-4 adapter on the DeepSeek-R1-Distill-1.5B draft model.

    Preferences are stored in SQLite; training fires once 20 untrained rows
    accumulate, mixing in 5 historical trained rows to prevent forgetting.
    """

    def __init__(self) -> None:
        self._db_path  = ADAPTER_PATH / "dpo_prefs.db"
        self._db_lock  = threading.Lock()
        self._model    = None
        self._tokenizer = None
        self._optimizer = None
        self._init_db()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="dpo-trainer"
        )
        self._thread.start()

    # ── DB setup ───────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        ADAPTER_PATH.mkdir(parents=True, exist_ok=True)
        with self._db_lock:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS dpo_prefs (
                        id         INTEGER PRIMARY KEY AUTOINCREMENT,
                        prompt     TEXT    NOT NULL,
                        winner     TEXT    NOT NULL,
                        loser      TEXT    NOT NULL,
                        path       TEXT    NOT NULL,
                        old_grade  TEXT    NOT NULL,
                        new_grade  TEXT    NOT NULL,
                        trained    INTEGER NOT NULL DEFAULT 0,
                        created_at REAL    NOT NULL DEFAULT (strftime('%s', 'now'))
                    )
                """)
                conn.commit()

    # ── Public API ─────────────────────────────────────────────────────────────

    def queue_event(self, image_path: str, old_grade: str, new_grade: str) -> None:
        """
        Persist a preference pair derived from a bucket move to SQLite.

        Training is deferred until BATCH_THRESHOLD untrained rows accumulate.
        """
        winner_score = _GRADE_SCORE.get(new_grade, 0.5)
        loser_score  = _GRADE_SCORE.get(old_grade, 0.5)

        if winner_score == loser_score:
            return
        if winner_score < loser_score:
            winner_score, loser_score = loser_score, winner_score

        fname  = Path(image_path).name
        prompt = (
            "You are a professional street photo editor.\n"
            f"Grade this image: {fname}\n"
            'Respond as JSON: {"score": <0.0-1.0>, "reasoning_log": "<text>"}'
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
            with self._db_lock:
                with sqlite3.connect(str(self._db_path)) as conn:
                    conn.execute(
                        "INSERT INTO dpo_prefs "
                        "(prompt, winner, loser, path, old_grade, new_grade) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (prompt, winner, loser, image_path, old_grade, new_grade),
                    )
                    conn.commit()
        except Exception as _e:
            print(f"[dpo] queue_event insert failed: {_e}")

    # ── Background loop ────────────────────────────────────────────────────────

    def _run(self) -> None:
        while True:
            # Check how many untrained preferences are waiting.
            try:
                with self._db_lock:
                    with sqlite3.connect(str(self._db_path)) as conn:
                        (untrained_count,) = conn.execute(
                            "SELECT COUNT(*) FROM dpo_prefs WHERE trained=0"
                        ).fetchone()
            except Exception as _e:
                print(f"[dpo] DB read error: {_e}")
                time.sleep(10)
                continue

            if untrained_count < BATCH_THRESHOLD:
                time.sleep(SLEEP_IDLE)
                continue

            if _vram_used_gb() > VRAM_LIMIT_GB:
                time.sleep(10)
                continue

            # Fetch BATCH_THRESHOLD untrained rows — hard negatives first.
            # grade_delta = |rank(new_grade) - rank(old_grade)|
            # delta=2 → Strong↔Weak flip (hardest mistake); delta=1 → adjacent move
            # 40 % of the batch is reserved for delta=2 rows; remainder fills by recency.
            _RANK_SQL = (
                "CASE {col} "
                "WHEN 'Strong ✅' THEN 2 "
                "WHEN 'Mid ⚠️'   THEN 1 "
                "ELSE 0 END"
            )
            _delta_expr = (
                f"ABS({_RANK_SQL.format(col='new_grade')} "
                f"  - {_RANK_SQL.format(col='old_grade')})"
            )
            _HARD_NEG_SLOTS = max(1, int(BATCH_THRESHOLD * 0.40))   # ≥40 % hard negatives
            _SOFT_SLOTS     = BATCH_THRESHOLD - _HARD_NEG_SLOTS
            try:
                with self._db_lock:
                    with sqlite3.connect(str(self._db_path)) as conn:
                        conn.row_factory = sqlite3.Row
                        # Hard negatives: delta=2 rows (Strong↔Weak), newest first
                        hard_rows = conn.execute(
                            f"SELECT * FROM dpo_prefs WHERE trained=0 "
                            f"AND {_delta_expr} = 2 "
                            f"ORDER BY created_at DESC LIMIT ?",
                            (_HARD_NEG_SLOTS,),
                        ).fetchall()
                        # Remaining slots: any untrained row not already selected, by delta then recency
                        hard_ids  = {r["id"] for r in hard_rows}
                        excl      = ",".join("?" * len(hard_ids)) if hard_ids else "0"
                        soft_rows = conn.execute(
                            f"SELECT * FROM dpo_prefs WHERE trained=0 "
                            f"AND id NOT IN ({excl}) "
                            f"ORDER BY {_delta_expr} DESC, created_at DESC LIMIT ?",
                            list(hard_ids) + [_SOFT_SLOTS],
                        ).fetchall()
                        new_rows = list(hard_rows) + list(soft_rows)
                        hist_rows = conn.execute(
                            "SELECT * FROM dpo_prefs WHERE trained=1 "
                            "ORDER BY RANDOM() LIMIT ?",
                            (HISTORY_MIX,),
                        ).fetchall()
                batch   = [dict(r) for r in new_rows] + [dict(r) for r in hist_rows]
                new_ids = [r["id"] for r in new_rows]
            except Exception as _e:
                print(f"[dpo] batch fetch error: {_e}")
                time.sleep(10)
                continue

            if not batch:
                time.sleep(SLEEP_IDLE)
                continue

            try:
                if not self._load():
                    time.sleep(60)
                    continue

                for ev in batch:
                    if _vram_used_gb() > VRAM_LIMIT_GB:
                        break
                    try:
                        loss = self._step(ev["prompt"], ev["winner"], ev["loser"])
                        print(
                            f"[dpo] loss={loss:.4f}  {Path(ev['path']).name}"
                            f"  ({ev['old_grade']} -> {ev['new_grade']})"
                        )
                    except Exception as e_step:
                        print(f"[dpo] step error: {e_step}")

                self._save()
                _update_style_log([dict(r) for r in new_rows])

                # Mark the newly trained rows so they become history for replay.
                if new_ids:
                    placeholders = ",".join("?" * len(new_ids))
                    try:
                        with self._db_lock:
                            with sqlite3.connect(str(self._db_path)) as conn:
                                conn.execute(
                                    f"UPDATE dpo_prefs SET trained=1 "
                                    f"WHERE id IN ({placeholders})",
                                    new_ids,
                                )
                                conn.commit()
                    except Exception as _e:
                        print(f"[dpo] mark trained error: {_e}")

            except Exception as e:
                print(f"[dpo] batch error: {e}")
            finally:
                self._unload()

    # ── Model management ───────────────────────────────────────────────────────

    def _load(self) -> bool:
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
        assert self._model is not None and self._optimizer is not None

        full_w = prompt + "\n" + winner
        full_l = prompt + "\n" + loser

        self._model.disable_adapter_layers()
        with torch.no_grad():
            ref_lp_w = _seq_log_prob(self._model, self._tokenizer, full_w)
            ref_lp_l = _seq_log_prob(self._model, self._tokenizer, full_l)
        self._model.enable_adapter_layers()

        self._optimizer.zero_grad()
        pi_lp_w = _seq_log_prob(self._model, self._tokenizer, full_w)
        pi_lp_l = _seq_log_prob(self._model, self._tokenizer, full_l)

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
        tmp_path = ADAPTER_PATH.parent / "dpo_adapter.tmp"
        try:
            self._model.save_pretrained(str(tmp_path))
            # Atomic swap: rename temp dir over the live adapter dir.
            import shutil
            if ADAPTER_PATH.exists():
                shutil.rmtree(str(ADAPTER_PATH))
            tmp_path.rename(ADAPTER_PATH)
        except Exception as e:
            print(f"[dpo] Adapter save failed: {e}")
            try:
                import shutil
                shutil.rmtree(str(tmp_path), ignore_errors=True)
            except Exception:
                pass

    def _unload(self) -> None:
        self._model     = None
        self._tokenizer = None
        self._optimizer = None
        _purge_vram()


# ── Style log ──────────────────────────────────────────────────────────────────

def _update_style_log(events: list[dict]) -> None:
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

    # Atomic write.
    tmp = STYLE_LOG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(log, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(STYLE_LOG_PATH))


def load_style_instruction() -> str:
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
    global _trainer
    if _trainer is None:
        with _trainer_lock:
            if _trainer is None:
                _trainer = BackgroundDPOTrainer()
    return _trainer
