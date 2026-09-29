"""Bounded child-process execution with process-tree cleanup."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

if os.name == "nt":
    from benchlib.windows_job import CREATE_SUSPENDED, WindowsJob, WindowsJobError
else:
    CREATE_SUSPENDED = 0

    class WindowsJobError(RuntimeError):
        """Placeholder for the platform-specific startup exception."""

TIMEOUT_EXIT_CODE = 124


class ProcessDeadlineError(RuntimeError):
    """A child could not start or finish inside its absolute deadline."""


class ProcessTreeError(RuntimeError):
    """A child process tree could not be proven terminated."""


@dataclass(frozen=True)
class Deadline:
    """An absolute monotonic deadline."""

    ends_at: float

    @classmethod
    def after(cls, seconds: float) -> "Deadline":
        return cls(time.monotonic() + seconds)

    def remaining(self) -> float:
        return self.ends_at - time.monotonic()


@dataclass(frozen=True)
class ProcessResult:
    """Completed child outcome."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


def operation_deadline(suite: Deadline, ceiling_seconds: float) -> Deadline:
    """Bound one operation by both its ceiling and the suite remainder."""
    now = time.monotonic()
    if suite.ends_at <= now:
        raise ProcessDeadlineError("suite wall-clock budget is exhausted")
    return Deadline(min(suite.ends_at, now + ceiling_seconds))


def _remaining(deadline: Deadline, action: str) -> float:
    remaining = deadline.remaining()
    if remaining <= 0:
        raise ProcessDeadlineError(f"deadline expired during {action}")
    return remaining


def _process_options() -> dict[str, object]:
    if os.name == "nt":
        return {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED
        }
    return {"start_new_session": True}


def _terminate_process_tree(
    process: subprocess.Popen[str], deadline: Deadline, windows_job: object | None
) -> None:
    if os.name == "nt":
        try:
            if windows_job is None:
                raise ProcessTreeError(f"PID {process.pid} has no Windows Job Object")
            windows_job.terminate(
                _remaining(deadline, "Windows process-tree termination")
            )
        except WindowsJobError as error:
            raise ProcessTreeError(f"PID {process.pid} tree cleanup failed: {error}") from error
        finally:
            if windows_job is not None:
                windows_job.close()
    else:
        try:
            # The group may outlive its Cargo parent. Always address the group;
            # checking `process.poll()` here would leak surviving descendants.
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as error:
            raise ProcessTreeError(
                f"process group {process.pid} could not be terminated: {error}"
            ) from error
    try:
        process.wait(timeout=_remaining(deadline, "process-tree reap"))
    except (ProcessDeadlineError, subprocess.TimeoutExpired) as error:
        raise ProcessTreeError(f"PID {process.pid} survived process-tree termination") from error


def run_process(
    command: Sequence[str],
    *,
    working_directory: Path,
    deadline: Deadline,
    termination_grace_seconds: float,
    capture: bool,
) -> ProcessResult:
    """Run a child while reserving part of its deadline for tree cleanup."""
    available = _remaining(deadline, "process start")
    cleanup_reserve = min(termination_grace_seconds, available / 2.0)
    execution_seconds = available - cleanup_reserve
    if execution_seconds <= 0:
        raise ProcessDeadlineError("no execution time remains after cleanup reservation")
    windows_job = WindowsJob() if os.name == "nt" else None
    try:
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            encoding="utf-8",
            errors="replace",
            **_process_options(),
        )
        if os.name == "nt":
            assert windows_job is not None
            windows_job.assign(process.pid)
            windows_job.resume(process.pid)
    except (OSError, WindowsJobError) as error:
        cleanup_error: BaseException | None = None
        try:
            if "process" in locals():
                process.kill()
                process.wait(timeout=_remaining(deadline, "failed process-start cleanup"))
        except (ProcessDeadlineError, subprocess.TimeoutExpired) as failure:
            cleanup_error = failure
        finally:
            if windows_job is not None:
                windows_job.close()
        if cleanup_error is not None:
            raise ProcessTreeError(
                f"failed child start also exceeded its cleanup deadline: {cleanup_error}"
            ) from error
        raise ProcessTreeError(f"child process could not start safely: {error}") from error
    try:
        stdout, stderr = process.communicate(timeout=execution_seconds)
    except subprocess.TimeoutExpired:
        _terminate_process_tree(process, deadline, windows_job)
        try:
            stdout, stderr = process.communicate(
                timeout=_remaining(deadline, "captured-output cleanup")
            )
        except subprocess.TimeoutExpired as error:
            raise ProcessTreeError(
                f"PID {process.pid} pipes remained open after tree termination"
            ) from error
        return ProcessResult(TIMEOUT_EXIT_CODE, stdout or "", stderr or "", True)
    except KeyboardInterrupt:
        _terminate_process_tree(process, deadline, windows_job)
        raise
    except BaseException:
        _terminate_process_tree(process, deadline, windows_job)
        raise
    returncode = process.returncode
    _terminate_process_tree(process, deadline, windows_job)
    return ProcessResult(returncode, stdout or "", stderr or "", False)
