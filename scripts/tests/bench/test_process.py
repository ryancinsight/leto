"""Real-process coverage for deadlines, output, and tree cleanup."""

from __future__ import annotations

import _thread
import socket
import tempfile
import threading
import time
from pathlib import Path

import pytest

from .support import (
    REPOSITORY,
    blocking_tree,
    cleanup_tree,
    process_running,
    python_command,
    run_python,
)
from benchlib.process import (
    Deadline,
    ProcessDeadlineError,
    operation_deadline,
    run_process,
)


def test_success_nonzero_and_visible_output(capfd) -> None:
    success = run_python("print('captured')")
    assert success.returncode == 0
    assert success.stdout.strip() == "captured"
    assert not success.timed_out

    failure = run_python(
        "import sys; print('rejected', file=sys.stderr); raise SystemExit(7)"
    )
    assert failure.returncode == 7
    assert failure.stderr.strip() == "rejected"

    visible = run_python("print('visible')", capture=False)
    output, error = capfd.readouterr()
    assert visible.returncode == 0
    assert "visible" in output
    assert not error


def test_timeout_terminates_live_parent_tree() -> None:
    with tempfile.TemporaryDirectory(prefix="leto-bench-tree-") as temporary:
        pid_file = Path(temporary) / "tree.pid"
        try:
            result = run_python(blocking_tree(False, pid_file))
            child_id = int(result.stdout.strip())
            assert result.timed_out
            assert result.returncode == 124
            assert not process_running(child_id)
        finally:
            if pid_file.is_file():
                parent_id, child_id = map(
                    int, pid_file.read_text(encoding="utf-8").split()
                )
                cleanup_tree(parent_id, child_id)


def test_captured_timeout_kills_tree_after_parent_exit() -> None:
    result = run_python(blocking_tree(True))
    child_id = int(result.stdout.strip())
    assert result.timed_out
    assert not process_running(child_id)


def test_visible_parent_success_cleans_descendants(capfd) -> None:
    with tempfile.TemporaryDirectory(prefix="leto-bench-pid-") as temporary:
        pid_file = Path(temporary) / "tree.pid"
        result = run_python(blocking_tree(True, pid_file), capture=False)
        output, error = capfd.readouterr()
        _, child_id = map(int, pid_file.read_text(encoding="utf-8").split())
        assert result.returncode == 0
        assert not result.timed_out
        assert str(child_id) in output
        assert error == ""
        assert not process_running(child_id)


def test_keyboard_interrupt_cleans_tree_with_operation_deadline() -> None:
    with tempfile.TemporaryDirectory(prefix="leto-bench-interrupt-") as temporary:
        pid_file = Path(temporary) / "tree.pid"
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]

            def interrupt_when_started() -> None:
                connection, _ = listener.accept()
                connection.close()
                _thread.interrupt_main()

            interrupter = threading.Thread(target=interrupt_when_started)
            interrupter.start()
            child_source = "import threading; threading.Event().wait()"
            source = "; ".join(
                [
                    "import os, pathlib, socket, subprocess, sys, threading",
                    "child = subprocess.Popen("
                    f"{[__import__('sys').executable, '-c', child_source]!r})",
                    f"pathlib.Path({str(pid_file)!r}).write_text("
                    "f'{os.getpid()} {child.pid}', encoding='utf-8')",
                    f"connection = socket.create_connection(('127.0.0.1', {port}))",
                    "connection.close()",
                    "threading.Event().wait()",
                ]
            )
            with pytest.raises(KeyboardInterrupt):
                run_process(
                    python_command(source),
                    working_directory=REPOSITORY,
                    deadline=Deadline.after(3),
                    termination_grace_seconds=2,
                    capture=True,
                )
            interrupter.join(timeout=1)
            assert not interrupter.is_alive()
            _, child_id = map(int, pid_file.read_text(encoding="utf-8").split())
            assert not process_running(child_id)


def test_exhausted_process_budget_never_starts_child() -> None:
    with tempfile.TemporaryDirectory(prefix="leto-bench-budget-") as temporary:
        marker = Path(temporary) / "started"
        with pytest.raises(ProcessDeadlineError, match="process start"):
            run_process(
                python_command(
                    f"from pathlib import Path; Path({str(marker)!r}).touch()"
                ),
                working_directory=REPOSITORY,
                deadline=Deadline(0),
                termination_grace_seconds=1,
                capture=True,
            )
        assert not marker.exists()

    expired = Deadline(time.monotonic() - 1)
    with pytest.raises(ProcessDeadlineError, match="suite wall-clock budget"):
        operation_deadline(expired, 25)
