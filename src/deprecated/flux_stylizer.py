"""
flux_stylizer — DEPRECATED

Flux 2 / ControlNet / IP-Adapter stylization has been removed.
The Creative Direction pipeline is now Purist: no pixel modification.

This stub exists to satisfy any stale imports without crashing the server.
"""
import warnings

warnings.warn(
    "flux_stylizer is deprecated. The Creative Direction pipeline no longer "
    "performs stylization. Import creative_director instead.",
    DeprecationWarning,
    stacklevel=2,
)


class FluxStylizer:
    """Deprecated stub — all methods are no-ops."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "FluxStylizer is deprecated and does nothing.",
            DeprecationWarning,
            stacklevel=2,
        )

    def process_batch(self, strong_paths, *args, **kwargs):
        return [
            {"source_path": p, "output_path": p, "success": False,
             "error": "FluxStylizer is deprecated — use purist pipeline"}
            for p in strong_paths
        ]
