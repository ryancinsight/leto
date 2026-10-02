"""Windows Job Object process-count and failure behavior."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from .support import REPOSITORY, process_running, python_command

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

@unittest.skipUnless(os.name == "nt", "Windows Job Object behavior")
class WindowsJobTests(unittest.TestCase):
    def test_empty_job_finishes_without_waiting_on_the_job_handle(self) -> None:
        from benchlib.windows_job import WindowsJob

        job = WindowsJob()
        try:
            job.terminate(1)
            self.assertEqual(job.active_processes(), 0)
        finally:
            job.close()

    def test_closed_job_fails_closed(self) -> None:
        from benchlib.windows_job import WindowsJob, WindowsJobError

        job = WindowsJob()
        job.close()
        with self.assertRaisesRegex(WindowsJobError, "already closed"):
            job.terminate(1)

    def test_owned_handle_close_failure_is_reported(self) -> None:
        from benchlib.windows_job import (
            PROCESS_ACCESS,
            WindowsJobError,
            _close_handle,
            _kernel32,
            _owned_handle,
        )

        handle = _kernel32.OpenProcess(PROCESS_ACCESS, False, os.getpid())
        self.assertTrue(handle)
        _close_handle(handle, "closing initial process handle failed")
        with self.assertRaisesRegex(WindowsJobError, "closing test handle failed"):
            with _owned_handle(handle, "closing test handle failed"):
                pass

    def test_process_cleanup_surfaces_a_closed_job_failure(self) -> None:
        from benchlib.process import Deadline, ProcessTreeError, _terminate_process_tree
        from benchlib.windows_job import CREATE_SUSPENDED, WindowsJob

        job = WindowsJob()
        job.close()
        process = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            creationflags=CREATE_SUSPENDED,
        )
        try:
            with self.assertRaisesRegex(ProcessTreeError, "tree cleanup failed"):
                _terminate_process_tree(process, Deadline.after(3), job)
        finally:
            process.kill()
            process.wait(timeout=3)

    def test_assignment_and_resume_failures_reap_the_suspended_root(self) -> None:
        from benchlib.process import Deadline, ProcessTreeError, run_process
        from benchlib.windows_job import WindowsJob, WindowsJobError

        real_init = subprocess.Popen.__init__
        for method in ("assign", "resume"):
            with self.subTest(method=method):
                spawned: list[subprocess.Popen[str]] = []

                def record_process(
                    process: subprocess.Popen[str],
                    *args: object,
                    **kwargs: object,
                ) -> None:
                    real_init(process, *args, **kwargs)
                    spawned.append(process)

                try:
                    with patch.object(
                        subprocess.Popen, "__init__", new=record_process
                    ), patch.object(
                        WindowsJob,
                        method,
                        side_effect=WindowsJobError(f"{method} failure"),
                    ):
                        with self.assertRaisesRegex(
                            ProcessTreeError, "could not start safely"
                        ):
                            run_process(
                                python_command("pass"),
                                working_directory=REPOSITORY,
                                deadline=Deadline.after(3),
                                termination_grace_seconds=1,
                                capture=True,
                            )
                    self.assertEqual(len(spawned), 1)
                    self.assertIsNotNone(spawned[0].poll())
                    self.assertFalse(process_running(spawned[0].pid))
                finally:
                    for process in spawned:
                        if process.poll() is None:
                            process.kill()
                            process.wait(timeout=3)

    def test_sigint_after_native_handle_acquisition_closes_the_handle(self) -> None:
        from benchlib import windows_job
        from benchlib.process import Deadline, run_process
        from benchlib.windows_job import WindowsJob

        acquisitions = (
            "OpenProcess",
            "CreateToolhelp32Snapshot",
            "OpenThread",
        )
        for operation in acquisitions:
            with self.subTest(operation=operation):
                acquired_handles: list[object] = []
                closed_handles: list[object] = []
                original_acquire = getattr(windows_job._kernel32, operation)
                original_close = windows_job._close_handle
                owned_close_action = {
                    "OpenProcess": "closing process handle for PID",
                    "CreateToolhelp32Snapshot": "closing the thread snapshot",
                    "OpenThread": "closing thread handle for PID",
                }[operation]

                def interrupt_after_acquire(
                    *arguments: object,
                ) -> object:
                    handle = original_acquire(*arguments)
                    if handle:
                        acquired_handles.append(handle)
                        signal.raise_signal(signal.SIGINT)
                    return handle

                def record_close(handle: object, action: str) -> None:
                    if handle in acquired_handles and action.startswith(
                        owned_close_action
                    ):
                        closed_handles.append(handle)
                    original_close(handle, action)

                with patch.object(
                    windows_job._kernel32,
                    operation,
                    side_effect=interrupt_after_acquire,
                ), patch.object(
                    windows_job, "_close_handle", side_effect=record_close
                ):
                    with self.assertRaises(KeyboardInterrupt):
                        run_process(
                            python_command("pass"),
                            working_directory=REPOSITORY,
                            deadline=Deadline.after(3),
                            termination_grace_seconds=1,
                            capture=False,
                        )
                self.assertEqual(len(acquired_handles), 1)
                self.assertEqual(closed_handles, acquired_handles)


if __name__ == "__main__":
    unittest.main()
