"""Keep grading processes at full CPU speed (2026-10-04).

Windows 11 may infer a low quality-of-service for windowless background
processes and throttle them (EcoQoS: lower clocks / efficiency cores) — the
grade runner and its workers have no window. Explicitly opting OUT of
execution-speed throttling (ControlMask=EXECUTION_SPEED, StateMask=0) marks
them HighQoS. Measured no effect plugged in on "Ultimate Performance"; it is
insurance for battery and balanced power modes. Never raises.
"""
from __future__ import annotations

import sys


def opt_out_power_throttling() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class _PPTS(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]

        k32 = ctypes.windll.kernel32
        state = _PPTS(1, 0x1, 0x0)       # PROCESS_POWER_THROTTLING_EXECUTION_SPEED, off
        return bool(k32.SetProcessInformation(k32.GetCurrentProcess(), 4,   # ProcessPowerThrottling
                                              ctypes.byref(state), ctypes.sizeof(state)))
    except Exception:
        return False
