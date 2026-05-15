"""
VRAM Manager

Manages VRAM between pipeline phases to prevent overflow on laptop GPUs.

Essential for 4-6GB VRAM laptops where models must be loaded/unloaded
strategically to avoid out-of-memory errors.
"""

from __future__ import annotations

import gc
import torch
from typing import Dict, Optional, Tuple


class VRAMManager:
    """
    Manages VRAM between pipeline phases to prevent overflow.
    
    Provides utilities for:
    - Clearing VRAM between phases
    - Monitoring VRAM usage
    - Ensuring sufficient memory for model loading
    """
    
    @staticmethod
    def purge_vram() -> None:
        """Full VRAM purge — call between every major inference step."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        gc.collect()

    @staticmethod
    def clear_between_phases() -> None:
        """
        Force cleanup between Bulk Encoder and Reasoning phases.

        Call this between heavy operations to free VRAM for the next phase.
        Essential for 4-6GB VRAM laptops.
        """
        VRAMManager.purge_vram()
    
    @staticmethod
    def get_vram_usage() -> Dict[str, float]:
        """
        Return current VRAM usage statistics.
        
        Returns:
            Dict with 'total', 'allocated', 'reserved' keys (in bytes)
        """
        if not torch.cuda.is_available():
            return {
                "total": 0.0,
                "allocated": 0.0,
                "reserved": 0.0,
            }
        
        return {
            "total": torch.cuda.get_device_properties(0).total_memory,
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
        }
    
    @staticmethod
    def get_vram_usage_percent() -> Dict[str, float]:
        """
        Return VRAM usage as percentages.
        
        Returns:
            Dict with 'allocated_pct', 'reserved_pct', 'free_pct' keys
        """
        usage = VRAMManager.get_vram_usage()
        total = usage["total"]
        
        if total == 0:
            return {
                "allocated_pct": 0.0,
                "reserved_pct": 0.0,
                "free_pct": 100.0,
            }
        
        return {
            "allocated_pct": usage["allocated"] / total * 100,
            "reserved_pct": usage["reserved"] / total * 100,
            "free_pct": (total - usage["allocated"]) / total * 100,
        }
    
    @staticmethod
    def ensure_sufficient_vram(min_required: int = 3_000_000_000) -> bool:
        """
        Check if at least `min_required` bytes are available.
        
        Args:
            min_required: Minimum required VRAM in bytes (default: 3GB)
        
        Returns:
            True if sufficient VRAM is available
        """
        if not torch.cuda.is_available():
            return True
        
        usage = VRAMManager.get_vram_usage()
        available = usage["total"] - usage["allocated"]
        return available >= min_required
    
    @staticmethod
    def get_available_vram() -> int:
        """
        Return available VRAM in bytes.
        
        Returns:
            Available VRAM (total - allocated)
        """
        if not torch.cuda.is_available():
            return 0
        
        usage = VRAMManager.get_vram_usage()
        return usage["total"] - usage["allocated"]
    
    @staticmethod
    def get_model_size_estimate(model: torch.nn.Module) -> int:
        """
        Estimate model size in VRAM.
        
        Args:
            model: PyTorch model
        
        Returns:
            Estimated size in bytes
        """
        total_bytes = 0
        for param in model.parameters():
            total_bytes += param.numel() * param.element_size()
        return total_bytes
    
    @staticmethod
    def safe_load_model(
        model_fn,
        min_vram: int = 2_000_000_000,
        max_retries: int = 3,
    ):
        """
        Safely load a model with VRAM checks and retries.
        
        Args:
            model_fn: Function that creates and returns the model
            min_vram: Minimum required VRAM in bytes
            max_retries: Maximum number of retry attempts
        
        Returns:
            Loaded model
        
        Raises:
            RuntimeError: If model cannot be loaded after retries
        """
        for attempt in range(max_retries):
            # Clear VRAM before loading
            VRAMManager.clear_between_phases()
            
            # Check available VRAM
            if not VRAMManager.ensure_sufficient_vram(min_vram):
                if attempt < max_retries - 1:
                    continue
                raise RuntimeError(
                    f"Insufficient VRAM. Required: {min_vram/1e9:.1f}GB, "
                    f"Available: {VRAMManager.get_available_vram()/1e9:.1f}GB"
                )
            
            try:
                # Try to load model
                model = model_fn()
                return model
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and attempt < max_retries - 1:
                    continue
                raise
        
        raise RuntimeError("Failed to load model after multiple attempts")


class VRAMMonitor:
    """
    Monitors VRAM usage over time for profiling and debugging.
    """
    
    def __init__(self):
        self._history: list = []
        self._enabled = torch.cuda.is_available()
    
    def record(self, tag: str = "") -> None:
        """
        Record current VRAM usage with optional tag.
        
        Args:
            tag: Label for this measurement
        """
        if not self._enabled:
            return
        
        usage = VRAMManager.get_vram_usage()
        self._history.append({
            "tag": tag,
            "timestamp": torch.cuda.Event(enable_timing=True).elapsed_time(
                torch.cuda.Event(enable_timing=True)
            ) if torch.cuda.is_available() else 0,
            "allocated_mb": usage["allocated"] / 1e6,
            "reserved_mb": usage["reserved"] / 1e6,
        })
    
    def get_peak_usage(self) -> Dict[str, float]:
        """
        Get peak VRAM usage from recorded history.
        
        Returns:
            Dict with 'allocated_peak_mb', 'reserved_peak_mb'
        """
        if not self._history:
            return {"allocated_peak_mb": 0.0, "reserved_peak_mb": 0.0}
        
        return {
            "allocated_peak_mb": max(h["allocated_mb"] for h in self._history),
            "reserved_peak_mb": max(h["reserved_mb"] for h in self._history),
        }
    
    def reset(self) -> None:
        """Clear recorded history."""
        self._history = []
    
    def print_report(self) -> None:
        """Print VRAM usage report."""
        if not self._enabled:
            print("CUDA not available - VRAM monitoring disabled")
            return
        
        print("\n=== VRAM Usage Report ===")
        for entry in self._history:
            tag = entry["tag"] or "unnamed"
            print(f"{tag}: {entry['allocated_mb']:.1f}MB allocated, "
                  f"{entry['reserved_mb']:.1f}MB reserved")
        
        peak = self.get_peak_usage()
        print(f"\nPeak usage: {peak['allocated_peak_mb']:.1f}MB allocated, "
              f"{peak['reserved_peak_mb']:.1f}MB reserved")
        print("=========================\n")


# Global VRAM manager instance
_vram_manager = VRAMManager()


def get_vram_manager() -> VRAMManager:
    """Get global VRAM manager instance."""
    return _vram_manager
