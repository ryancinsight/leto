"""Bounded child-process execution with process-tree cleanup."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import current_thread, main_thread
from types import FrameType
from typing import Iterator, Sequence

if os.name == "nt":
    from benchlib.windows_job import CREATE_SUSPENDED, WindowsJob, WindowsJobError
else:
    CREATE_SUSPENDED = 0

    class WindowsJobError(RuntimeError):
        """Placeholder for the platform-specific startup exception."""

TIMEOUT_EXIT_CODE = 124
# 100 Hz caps process-group exit detection latency at 10 ms; the absolute
# cleanup deadline remains authoritative.
PROCESS_GROUP_POLL_SECONDS = 0.01


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


def _execution_deadline(
    deadline: Deadline, termination_grace_seconds: float, started_at: float
) -> Deadline:
    """Reserve cleanup time from the budget left after process setup."""
    available = deadline.ends_at - started_at
    if available <= 0:
        raise ProcessDeadlineError("deadline expired during process setup")
    cleanup_reserve = min(termination_grace_seconds, available / 2.0)
    return Deadline(deadline.ends_at - cleanup_reserve)


@contextmanager
def _defer_keyboard_interrupts() -> Iterator[None]:
    """Defer SIGINT until ownership-changing work reaches a safe boundary."""
    if current_thread() is not main_thread():
        yield
        return

    previous_handler = signal.getsignal(signal.SIGINT)
    pending: list[tuple[int, FrameType | None]] = []

    def record_interrupt(signum: int, frame: FrameType | None) -> None:
        pending.append((signum, frame))

    signal.signal(signal.SIGINT, record_interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous_handler)

    if previous_handler is signal.SIG_IGN:
        return
    for signum, frame in pending:
        if callable(previous_handler):
            previous_handler(signum, frame)
        else:
            raise KeyboardInterrupt


def _process_options() -> dict[str, object]:
    if os.name == "nt":
        return {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED
        }
    return {"start_new_session": True}


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        raise ProcessTreeError(
            f"process group {process_group} could not be inspected: {error}"
        ) from error
    return True


def _wait_for_process_group_exit(process_group: int, deadline: Deadline) -> None:
    while _process_group_exists(process_group):
        remaining = deadline.remaining()
        if remaining <= 0:
            raise ProcessTreeError(
                f"process group {process_group} survived process-tree termination"
            )
        time.sleep(min(PROCESS_GROUP_POLL_SECONDS, remaining))


def _close_process_pipes(process: subprocess.Popen[str]) -> list[BaseException]:
    errors: list[BaseException] = []
    with _defer_keyboard_interrupts():
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except BaseException as error:
                    errors.append(error)
    return errors


def _terminate_process_tree(
    process: subprocess.Popen[str],
    deadline: Deadline,
    windows_job: object | None,
    *,
    close_pipes: bool = False,
) -> None:
    try:
        with _defer_keyboard_interrupts():
            _terminate_process_tree_uninterrupted(
                process, deadline, windows_job, close_pipes=close_pipes
            )
    except BaseException as error:
        if not close_pipes:
            try:
                with _defer_keyboard_interrupts():
                    pipe_errors = _close_process_pipes(process)
            except BaseException as cleanup_error:
                pipe_errors = [cleanup_error]
            if pipe_errors:
                details = "; ".join(map(str, pipe_errors))
                raise ProcessTreeError(
                    f"PID {process.pid} pipes could not be closed after "
                    f"process-tree cleanup failed: {details}"
                ) from error
        raise


def _terminate_process_tree_uninterrupted(
    process: subprocess.Popen[str],
    deadline: Deadline,
    windows_job: object | None,
    *,
    close_pipes: bool,
) -> None:
    cleanup_errors: list[Exception] = []
    control_errors: list[BaseException] = []

    def record(error: BaseException) -> None:
        if isinstance(error, Exception):
            cleanup_errors.append(error)
        else:
            control_errors.append(error)

    if os.name == "nt":
        try:
            if windows_job is None:
                raise ProcessTreeError(f"PID {process.pid} has no Windows Job Object")
            windows_job.terminate(
                _remaining(deadline, "Windows process-tree termination")
            )
        except BaseException as error:
            record(error)
        if windows_job is not None:
            try:
                windows_job.close()
            except BaseException as error:
                record(error)
                if not isinstance(error, Exception):
                    try:
                        windows_job.close()
                    except BaseException as retry_error:
                        record(retry_error)
    else:
        try:
            # The group may outlive its Cargo parent. Always address the group;
            # checking `process.poll()` here would leak surviving descendants.
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except BaseException as error:
            record(error)
    try:
        if process.poll() is None:
            process.kill()
    except BaseException as error:
        record(error)
    if close_pipes:
        cleanup_errors.extend(_close_process_pipes(process))
    try:
        process.wait(timeout=_remaining(deadline, "process-tree reap"))
    except BaseException as error:
        record(error)
    if os.name != "nt":
        try:
            _wait_for_process_group_exit(process.pid, deadline)
        except BaseException as error:
            record(error)
    if cleanup_errors:
        details = "; ".join(map(str, cleanup_errors))
        if control_errors:
            details += "; interrupted during cleanup: " + "; ".join(
                map(str, control_errors)
            )
        raise ProcessTreeError(
            f"PID {process.pid} tree cleanup failed: {details}"
        ) from cleanup_errors[0]
    if control_errors:
        raise control_errors[0]


def run_process(
    command: Sequence[str],
    *,
    working_directory: Path,
    deadline: Deadline,
    termination_grace_seconds: float,
    capture: bool,
) -> ProcessResult:
    """Run a child while reserving part of its deadline for tree cleanup."""
    _remaining(deadline, "process start")
    windows_job = None
    process: subprocess.Popen[str] | None = None
    try:
        if os.name == "nt":
            windows_job = WindowsJob.__new__(WindowsJob)
            with _defer_keyboard_interrupts():
                windows_job.__init__()
        process = subprocess.Popen.__new__(subprocess.Popen)
        with _defer_keyboard_interrupts():
            process.__init__(
                command,
                cwd=working_directory,
                stdout=subprocess.PIPE if capture else None,
                stderr=subprocess.PIPE if capture else None,
                encoding="utf-8",
                errors="replace",
                **_process_options(),
            )
        if os.name == "nt":
            with _defer_keyboard_interrupts():
                assert windows_job is not None
                windows_job.assign(process.pid)
                windows_job.resume(process.pid)
    except BaseException as error:
        cleanup_errors: list[BaseException] = []
        try:
            if process is not None and getattr(process, "pid", None) is not None:
                _terminate_process_tree(
                    process, deadline, windows_job, close_pipes=True
                )
        except BaseException as failure:
            cleanup_errors.append(failure)
        if (
            process is None or getattr(process, "pid", None) is None
        ) and windows_job is not None:
            try:
                with _defer_keyboard_interrupts():
                    windows_job.close()
            except BaseException as failure:
                cleanup_errors.append(failure)
        if cleanup_errors:
            details = "; ".join(str(failure) for failure in cleanup_errors)
            raise ProcessTreeError(
                f"failed child start also failed cleanup after {error}: {details}"
            ) from error
        if isinstance(error, (OSError, WindowsJobError)):
            raise ProcessTreeError(
                f"child process could not start safely: {error}"
            ) from error
        raise
    try:
        execution_deadline = _execution_deadline(
            deadline, termination_grace_seconds, time.monotonic()
        )
        stdout, stderr = process.communicate(
            timeout=max(0.0, execution_deadline.remaining())
        )
    except (subprocess.TimeoutExpired, ProcessDeadlineError):
        _terminate_process_tree(process, deadline, windows_job)
        try:
            stdout, stderr = process.communicate(
                timeout=_remaining(deadline, "captured-output cleanup")
            )
        except subprocess.TimeoutExpired as error:
            pipe_errors = _close_process_pipes(process)
            if pipe_errors:
                details = "; ".join(map(str, pipe_errors))
                raise ProcessTreeError(
                    f"PID {process.pid} pipes could not be closed after "
                    f"captured-output cleanup failed: {details}"
                ) from error
            raise ProcessTreeError(
                f"PID {process.pid} pipes remained open after tree termination"
            ) from error
        except BaseException as error:
            pipe_errors = _close_process_pipes(process)
            if pipe_errors:
                details = "; ".join(map(str, pipe_errors))
                raise ProcessTreeError(
                    f"PID {process.pid} pipes could not be closed after "
                    f"captured-output cleanup failed: {details}"
                ) from error
            raise
        return ProcessResult(TIMEOUT_EXIT_CODE, stdout or "", stderr or "", True)
    except KeyboardInterrupt:
        _terminate_process_tree(process, deadline, windows_job, close_pipes=True)
        raise
    except BaseException:
        _terminate_process_tree(process, deadline, windows_job, close_pipes=True)
        raise
    returncode = process.returncode
    _terminate_process_tree(process, deadline, windows_job, close_pipes=True)
    return ProcessResult(returncode, stdout or "", stderr or "", False)
