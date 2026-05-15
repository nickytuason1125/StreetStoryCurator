"""
Priority-Gate Controller

Controls whether to skip the 7B verifier based on draft confidence.

Logic:
    - If draft_confidence > 0.88: skip verification, use draft score
    - If draft_confidence <= 0.88: trigger 7B verifier for correction

This enables speculative decoding - fast path for high-confidence drafts,
with verification only when needed.
"""

from __future__ import annotations

from typing import Optional


class PriorityGate:
    """
    Priority-Gate Controller for Speculative Decoding.
    
    Manages the trade-off between speed (skip verification) and accuracy
    (run verification) based on draft model confidence.
    """
    
    def __init__(self, threshold: float = 0.88):
        """
        Initialize Priority-Gate.
        
        Args:
            threshold: Confidence threshold for skipping verification.
                      Drafts with confidence > threshold skip 7B verifier.
        """
        self.threshold = threshold
    
    def should_skip(self, confidence: float) -> bool:
        """
        Determine if verification can be skipped.
        
        Args:
            confidence: Draft model confidence score [0, 1]
        
        Returns:
            True if verification should be skipped (draft is confident enough)
        """
        return confidence > self.threshold
    
    def trigger_verification(self, confidence: float) -> bool:
        """
        Determine if verification should be triggered.
        
        Args:
            confidence: Draft model confidence score [0, 1]
        
        Returns:
            True if verification should be triggered
        """
        return confidence <= self.threshold
    
    def get_priority_score(self, confidence: float) -> float:
        """
        Calculate priority score for scheduling.
        
        Higher confidence = higher priority = can skip verification.
        
        Args:
            confidence: Draft model confidence score [0, 1]
        
        Returns:
            Priority score [0, 1] (1.0 = highest priority, skip verification)
        """
        # Linear mapping: confidence > threshold → priority 1.0
        if confidence > self.threshold:
            return 1.0
        # Below threshold: scale from 0 to 1
        return confidence / self.threshold
    
    def adjust_threshold(self, new_threshold: float) -> None:
        """
        Adjust the confidence threshold dynamically.
        
        Lower threshold = more verifications (higher accuracy, slower)
        Higher threshold = fewer verifications (faster, potentially less accurate)
        
        Args:
            new_threshold: New confidence threshold [0, 1]
        """
        self.threshold = max(0.0, min(1.0, new_threshold))


class AdaptivePriorityGate(PriorityGate):
    """
    Adaptive Priority-Gate that adjusts threshold based on context.
    
    Can dynamically adjust the confidence threshold based on:
    - Image complexity
    - Available VRAM
    - Performance requirements
    """
    
    def __init__(
        self,
        base_threshold: float = 0.88,
        min_threshold: float = 0.70,
        max_threshold: float = 0.95,
    ):
        super().__init__(base_threshold)
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self._current_threshold = base_threshold
    
    def adjust_for_vram(self, vram_available: float, vram_total: float) -> None:
        """
        Adjust threshold based on available VRAM.
        
        Low VRAM → higher threshold (skip more verifications)
        High VRAM → lower threshold (run more verifications)
        """
        if vram_total == 0:
            return
        
        vram_ratio = vram_available / vram_total
        
        # Map VRAM ratio to threshold
        # Low VRAM → high threshold (skip more)
        # High VRAM → low threshold (verify more)
        new_threshold = self.max_threshold - (self.max_threshold - self.min_threshold) * vram_ratio
        self._current_threshold = new_threshold
    
    def adjust_for_complexity(self, complexity_score: float) -> None:
        """
        Adjust threshold based on image complexity.
        
        Complex images → lower threshold (need verification)
        Simple images → higher threshold (skip verification)
        """
        # Complex images need more verification
        # Map complexity [0, 1] to threshold adjustment
        adjustment = -0.1 * complexity_score  # Complex → lower threshold
        self._current_threshold = self.threshold + adjustment
        self._current_threshold = max(
            self.min_threshold, min(self.max_threshold, self._current_threshold)
        )
    
    def should_skip(self, confidence: float) -> bool:
        """Use current (possibly adjusted) threshold."""
        return confidence > self._current_threshold
    
    def trigger_verification(self, confidence: float) -> bool:
        """Use current (possibly adjusted) threshold."""
        return confidence <= self._current_threshold


def create_priority_gate(adaptive: bool = False, **kwargs) -> PriorityGate:
    """
    Factory function to create PriorityGate instances.
    
    Args:
        adaptive: If True, return AdaptivePriorityGate
        **kwargs: Arguments passed to constructor
    
    Returns:
        PriorityGate or AdaptivePriorityGate instance
    """
    if adaptive:
        return AdaptivePriorityGate(**kwargs)
    return PriorityGate(**kwargs)
