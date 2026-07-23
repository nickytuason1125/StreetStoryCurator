"""
Windows Job Object helper — ties a spawned subprocess's lifetime to this
process's own lifetime, so a killed/crashed parent always takes its
grading-pipeline children down with it.

The bug this fixes: subprocess.run()/Popen() on Windows do NOT propagate
termination to children. TerminateProcess() (what Task Manager's "End task",
Stop-Process -Force, and a crash all effectively do) only kills the exact PID
targeted — everything else keeps running as an orphan. Confirmed in this repo
2026-07-21: killing a grading run mid-IQA left iqa_worker.py running
indefinitely, silently holding ~9 GB RAM until the machine was nearly out of
memory. The same unprotected spawn pattern is used for encode_worker.py and
server.py's own top-level grade_runner.py launch, so the exposure isn't one
subprocess — a server crash can orphan the whole grading pipeline as a tree.

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE means: when the last handle to the job
object closes (which happens automatically when this process exits, cleanly
or not), every process still assigned to that job is force-killed by the OS
itself. No polling, no watchdog thread, no code in the child needs to
cooperate — it's enforced by the kernel.

Best-effort by design: any failure here (missing pywin32, already inside a
job that doesn't support nesting on very old Windows, permission issues)
degrades to "no job protection" rather than breaking the grading pipeline —
losing the safety net is far preferable to a hard failure on every grade.
"""
from __future__ import annotations

import subprocess
from typing import Optional

try:
    import win32job
    import win32api
    import win32con
    import win32process
    import pywintypes
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_job_handle = None  # type: ignore
_job_init_attempted = False


def _get_or_create_job():
    """One job object per parent process, created lazily on first use."""
    global _job_handle, _job_init_attempted
    if _job_init_attempted:
        return _job_handle
    _job_init_attempted = True
    if not _AVAILABLE:
        return None
    try:
        job = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
        _job_handle = job
    except Exception as e:
        print(f"[win_job] Job object unavailable, subprocesses won't be tied to parent lifetime: {e}")
        _job_handle = None
    return _job_handle


def _assign(pid: int) -> None:
    job = _get_or_create_job()
    if job is None:
        return
    try:
        h = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, pid)
        try:
            win32job.AssignProcessToJobObject(job, h)
        finally:
            win32api.CloseHandle(h)
    except pywintypes.error as e:
        # Common benign case: process already exited before we could assign it
        # (very fast children), or it's already in a job that predates Windows 8
        # nested-job support. Either way, that one child just runs unprotected.
        print(f"[win_job] Could not assign pid {pid} to job (non-fatal): {e}")
    except Exception as e:
        print(f"[win_job] Could not assign pid {pid} to job (non-fatal): {e}")


def run(args, **kwargs) -> subprocess.CompletedProcess:
    """Drop-in replacement for subprocess.run() that additionally ties the
    child's lifetime to this process's, on Windows. Identical behavior and
    return value everywhere else, including non-Windows (falls through to
    plain subprocess.run)."""
    if not _AVAILABLE:
        return subprocess.run(args, **kwargs)
    # input/timeout are subprocess.run()-only conveniences that communicate()
    # accepts but the Popen() constructor does not — must not forward them.
    input_arg = kwargs.pop("input", None)
    timeout = kwargs.pop("timeout", None)
    check = kwargs.pop("check", False)
    proc = subprocess.Popen(args, **kwargs)
    _assign(proc.pid)
    try:
        stdout, stderr = proc.communicate(input=input_arg, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise
    result = subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, args, stdout, stderr)
    return result


def popen(args, **kwargs) -> subprocess.Popen:
    """Drop-in replacement for subprocess.Popen() with the same job-object
    protection, for call sites that need the Popen object itself (e.g. to
    poll or stream) rather than a blocking run()."""
    proc = subprocess.Popen(args, **kwargs)
    if _AVAILABLE:
        _assign(proc.pid)
    return proc
