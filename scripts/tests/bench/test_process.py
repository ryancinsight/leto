"""Real-process coverage for deadlines, output, and tree cleanup."""

from __future__ import annotations

import _thread
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

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
    ProcessTreeError,
    _execution_deadline,
    _process_group_exists,
    _terminate_process_tree,
    operation_deadline,
    run_process,
)


class ProcessTests(unittest.TestCase):
    def test_success_nonzero_and_captured_output(self) -> None:
        success = run_python("print('captured')")
        self.assertEqual(success.returncode, 0)
        self.assertEqual(success.stdout.strip(), "captured")
        self.assertFalse(success.timed_out)

        failure = run_python(
            "import sys; print('rejected', file=sys.stderr); raise SystemExit(7)"
        )
        self.assertEqual(failure.returncode, 7)
        self.assertEqual(failure.stderr.strip(), "rejected")

        visible = run_python("pass", capture=False)
        self.assertEqual(visible.returncode, 0)
        self.assertFalse(visible.timed_out)

    def test_failed_executable_start_reports_the_os_error(self) -> None:
        with self.assertRaisesRegex(ProcessTreeError, "could not start safely"):
            run_process(
                [str(REPOSITORY / "missing-benchmark-executable")],
                working_directory=REPOSITORY,
                deadline=Deadline.after(3),
                termination_grace_seconds=1,
                capture=True,
            )

    @unittest.skipIf(os.name == "nt", "POSIX process-group signal behavior")
    def test_process_group_kill_failure_is_reported_after_reaping_root(self) -> None:
        process = subprocess.Popen(
            python_command("pass"),
            cwd=REPOSITORY,
            start_new_session=True,
        )
        process.wait(timeout=3)
        real_killpg = os.killpg

        def fail_group_kill(process_group: int, sig: int) -> None:
            if sig == signal.SIGKILL:
                raise PermissionError("signal denied by test")
            real_killpg(process_group, sig)

        with patch("benchlib.process.os.killpg", new=fail_group_kill):
            with self.assertRaisesRegex(ProcessTreeError, "signal denied by test"):
                _terminate_process_tree(process, Deadline.after(3), None)

    def test_timeout_terminates_live_parent_tree(self) -> None:
        with tempfile.TemporaryDirectory(prefix="leto-bench-tree-") as temporary:
            pid_file = Path(temporary) / "tree.pid"
            try:
                result = run_python(blocking_tree(False, pid_file))
                _, child_id = map(
                    int, pid_file.read_text(encoding="utf-8").split()
                )
                self.assertTrue(result.timed_out)
                self.assertEqual(result.returncode, 124)
                self.assertFalse(process_running(child_id))
            finally:
                if pid_file.is_file():
                    parent_id, child_id = map(
                        int, pid_file.read_text(encoding="utf-8").split()
                    )
                    cleanup_tree(parent_id, child_id)

    def test_interrupted_timeout_output_drain_closes_captured_pipes(self) -> None:
        processes: list[subprocess.Popen[str]] = []
        real_communicate = subprocess.Popen.communicate

        def interrupt_output_drain(
            process: subprocess.Popen[str], *args: object, **kwargs: object
        ) -> tuple[str | None, str | None]:
            processes.append(process)
            if len(processes) == 2:
                raise KeyboardInterrupt
            return real_communicate(process, *args, **kwargs)

        try:
            with patch.object(
                subprocess.Popen, "communicate", new=interrupt_output_drain
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_process(
                        python_command("import time; time.sleep(60)"),
                        working_directory=REPOSITORY,
                        deadline=Deadline.after(1.5),
                        termination_grace_seconds=0.5,
                        capture=True,
                    )
            self.assertEqual(len(processes), 2)
            process = processes[0]
            self.assertIsNotNone(process.poll())
            self.assertIsNotNone(process.stdout)
            self.assertIsNotNone(process.stderr)
            self.assertTrue(process.stdout.closed)
            self.assertTrue(process.stderr.closed)
        finally:
            if processes and processes[0].poll() is None:
                cleanup_tree(processes[0].pid, processes[0].pid)
                processes[0].wait(timeout=3)

    def test_captured_timeout_kills_tree_after_parent_exit(self) -> None:
        result = run_python(blocking_tree(True))
        child_id = int(result.stdout.strip())
        self.assertTrue(result.timed_out)
        self.assertFalse(process_running(child_id))

    def test_visible_parent_success_cleans_descendants(self) -> None:
        with tempfile.TemporaryDirectory(prefix="leto-bench-pid-") as temporary:
            pid_file = Path(temporary) / "tree.pid"
            result = run_python(blocking_tree(True, pid_file), capture=False)
            parent_id, child_id = map(
                int, pid_file.read_text(encoding="utf-8").split()
            )
            self.assertEqual(result.returncode, 0)
            self.assertFalse(result.timed_out)
            self.assertFalse(process_running(child_id))
            if os.name != "nt":
                self.assertFalse(_process_group_exists(parent_id))

    def test_keyboard_interrupt_cleans_tree_with_operation_deadline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="leto-bench-interrupt-") as temporary:
            pid_file = Path(temporary) / "tree.pid"
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                listener.listen(1)
                listener.settimeout(3)
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
                        f"{[sys.executable, '-c', child_source]!r})",
                        f"pathlib.Path({str(pid_file)!r}).write_text("
                        "f'{os.getpid()} {child.pid}', encoding='utf-8')",
                        f"connection = socket.create_connection(('127.0.0.1', {port}))",
                        "connection.close()",
                        "threading.Event().wait()",
                    ]
                )
                with self.assertRaises(KeyboardInterrupt):
                    run_process(
                        python_command(source),
                        working_directory=REPOSITORY,
                        deadline=Deadline.after(5),
                        termination_grace_seconds=2,
                        capture=True,
                    )
                interrupter.join(timeout=3)
                self.assertFalse(interrupter.is_alive())
                _, child_id = map(int, pid_file.read_text(encoding="utf-8").split())
                self.assertFalse(process_running(child_id))

    def test_keyboard_interrupt_during_startup_cleans_tree(self) -> None:
        real_init = subprocess.Popen.__init__
        spawned: list[subprocess.Popen[str]] = []

        def interrupt_after_spawn(
            process: subprocess.Popen[str], *args: object, **kwargs: object
        ) -> None:
            real_init(process, *args, **kwargs)
            spawned.append(process)
            raise KeyboardInterrupt

        try:
            with patch.object(subprocess.Popen, "__init__", new=interrupt_after_spawn):
                with self.assertRaises(KeyboardInterrupt):
                    run_process(
                        python_command(
                            "import threading; threading.Event().wait()"
                        ),
                        working_directory=REPOSITORY,
                        deadline=Deadline.after(5),
                        termination_grace_seconds=2,
                        capture=True,
                    )
                self.assertEqual(len(spawned), 1)
                self.assertIsNotNone(spawned[0].poll())
                self.assertFalse(process_running(spawned[0].pid))
                if os.name != "nt":
                    self.assertFalse(_process_group_exists(spawned[0].pid))
        finally:
            if spawned and spawned[0].poll() is None:
                cleanup_tree(spawned[0].pid, spawned[0].pid)
                spawned[0].wait(timeout=3)

    @unittest.skipUnless(os.name == "nt", "Windows Job Object cleanup behavior")
    def test_keyboard_interrupt_during_job_termination_kills_descendants(self) -> None:
        from benchlib.windows_job import WindowsJob

        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(3)
            port = listener.getsockname()[1]
            with tempfile.TemporaryDirectory(prefix="leto-bench-job-interrupt-") as temporary:
                pid_file = Path(temporary) / "child.pid"
                child_source = "; ".join(
                    [
                        "import os, socket, threading",
                        f"connection = socket.create_connection(('127.0.0.1', {port}))",
                        "print(os.getpid(), flush=True)",
                        "threading.Event().wait()",
                    ]
                )
                parent_source = "; ".join(
                    [
                        "import pathlib, subprocess, sys",
                        f"child = subprocess.Popen({[sys.executable, '-c', child_source]!r}, stdout=subprocess.PIPE, text=True)",
                        "child_id = int(child.stdout.readline())",
                        "child.stdout.close()",
                        f"pathlib.Path({str(pid_file)!r}).write_text(str(child_id), encoding='utf-8')",
                    ]
                )
                accept_errors: list[BaseException] = []

                def accept_child() -> None:
                    try:
                        connection, _ = listener.accept()
                        connection.close()
                    except BaseException as error:
                        accept_errors.append(error)

                accepter = threading.Thread(target=accept_child)
                accepter.start()
                try:
                    with patch.object(
                        WindowsJob,
                        "terminate",
                        side_effect=KeyboardInterrupt,
                    ):
                        with self.assertRaises(KeyboardInterrupt):
                            run_process(
                                python_command(parent_source),
                                working_directory=REPOSITORY,
                                deadline=Deadline.after(5),
                                termination_grace_seconds=2,
                                capture=False,
                            )
                    accepter.join(timeout=3)
                    self.assertFalse(accepter.is_alive())
                    self.assertEqual(accept_errors, [])
                    child_id = int(pid_file.read_text(encoding="utf-8"))
                    self.assertFalse(process_running(child_id))
                finally:
                    if accepter.is_alive():
                        accepter.join(timeout=3)
                    if pid_file.is_file():
                        child_id = int(pid_file.read_text(encoding="utf-8"))
                        if process_running(child_id):
                            subprocess.run(
                                ["taskkill", "/PID", str(child_id), "/F"],
                                capture_output=True,
                                check=False,
                                timeout=3,
                            )

    @unittest.skipUnless(os.name == "nt", "Windows Job Object cleanup behavior")
    def test_interrupted_timeout_cleanup_closes_captured_pipes(self) -> None:
        from benchlib.windows_job import WindowsJob

        spawned: list[subprocess.Popen[str]] = []
        real_init = subprocess.Popen.__init__

        def record_process(
            process: subprocess.Popen[str], *args: object, **kwargs: object
        ) -> None:
            real_init(process, *args, **kwargs)
            spawned.append(process)

        try:
            with patch.object(
                subprocess.Popen, "__init__", new=record_process
            ), patch.object(WindowsJob, "terminate", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run_process(
                        python_command("import time; time.sleep(60)"),
                        working_directory=REPOSITORY,
                        deadline=Deadline.after(1.5),
                        termination_grace_seconds=0.5,
                        capture=True,
                    )
            self.assertEqual(len(spawned), 1)
            process = spawned[0]
            self.assertIsNotNone(process.poll())
            self.assertIsNotNone(process.stdout)
            self.assertIsNotNone(process.stderr)
            self.assertTrue(process.stdout.closed)
            self.assertTrue(process.stderr.closed)
        finally:
            for process in spawned:
                if process.poll() is None:
                    cleanup_tree(process.pid, process.pid)
                    process.wait(timeout=3)

    def test_execution_deadline_preserves_remaining_cleanup_budget(self) -> None:
        deadline = Deadline(100)
        execution = _execution_deadline(deadline, 4, started_at=95)
        self.assertEqual(execution.ends_at, 97.5)
        self.assertGreater(execution.ends_at, 95)
        self.assertLess(execution.ends_at, deadline.ends_at)
        with self.assertRaisesRegex(ProcessDeadlineError, "process setup"):
            _execution_deadline(deadline, 4, started_at=100)

    def test_exhausted_process_budget_never_starts_child(self) -> None:
        with tempfile.TemporaryDirectory(prefix="leto-bench-budget-") as temporary:
            marker = Path(temporary) / "started"
            with self.assertRaisesRegex(ProcessDeadlineError, "process start"):
                run_process(
                    python_command(
                        f"from pathlib import Path; Path({str(marker)!r}).touch()"
                    ),
                    working_directory=REPOSITORY,
                    deadline=Deadline(0),
                    termination_grace_seconds=1,
                    capture=True,
                )
            self.assertFalse(marker.exists())

        with self.assertRaisesRegex(ProcessDeadlineError, "suite wall-clock budget"):
            operation_deadline(Deadline(-1), 25)


if __name__ == "__main__":
    unittest.main()
