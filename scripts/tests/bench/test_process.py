"""The pinned-checkout consumer's contract: resolution, pin verification, and
the benchmark-facing surface over the Atlas runner (ADR 0067)."""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from benchlib import process  # noqa: E402


class FakeRunner:
    """A stand-in for the imported Atlas module, for the mapping tests."""

    class ProcessTreeTimeout(TimeoutError):
        def __init__(self, message: str) -> None:
            super().__init__(message)

    def __init__(self, *, timeout: bool = False, returncode: int = 0) -> None:
        self._timeout = timeout
        self._returncode = returncode
        self.commands: list[list[str]] = []
        self.timeouts: list[float] = []

    def run(self, command, *, cwd=None, env=None, input=None, timeout):
        self.commands.append(list(command))
        self.timeouts.append(timeout)
        import subprocess

        if self._timeout:
            raise self.ProcessTreeTimeout("timed out")
        return subprocess.CompletedProcess(command, self._returncode, b"out", b"err")


class PinResolutionTests(unittest.TestCase):
    def test_the_stack_tree_is_the_default_checkout(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ATLAS_TOOLING_CHECKOUT", None)
            resolved = process.atlas_checkout()
            self.assertEqual(resolved, Path(__file__).resolve().parents[5])

    def test_the_override_names_the_checkout(self) -> None:
        with mock.patch.dict(os.environ, {"ATLAS_TOOLING_CHECKOUT": "/tmp/atlas"}):
            self.assertEqual(process.atlas_checkout(), Path("/tmp/atlas"))

    def test_a_checkout_at_the_wrong_commit_is_refused(self) -> None:
        wrong = FakeRunner()
        with mock.patch.object(process, "atlas_checkout", return_value=Path(".")), \
                mock.patch.object(process, "_verify_pin", side_effect=process.ProcessTreeError(
                    "Atlas checkout is deadbeef, not the pinned " + process.ATLAS_PIN)):
            with self.assertRaises(process.ProcessTreeError):
                process.run_process(
                    ["true"],
                    working_directory=Path("."),
                    deadline=process.Deadline.after(30.0),
                    termination_grace_seconds=1.0,
                    capture=True,
                )


class PinnedConsumerTests(unittest.TestCase):
    def test_the_pinned_runner_receives_the_remaining_deadline(self) -> None:
        fake = FakeRunner(returncode=7)
        deadline = process.Deadline.after(30.0)
        with mock.patch.object(process, "_load_runner", return_value=fake):
            result = process.run_process(
                ["cargo", "bench", "--bench", "kernels"],
                working_directory=Path("/repo"),
                deadline=deadline,
                termination_grace_seconds=1.5,
                capture=True,
            )
        self.assertEqual(fake.commands, [["cargo", "bench", "--bench", "kernels"]])
        self.assertLessEqual(fake.timeouts[0], 30.0)
        self.assertGreater(fake.timeouts[0], 29.0)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout, "out")
        self.assertEqual(result.stderr, "err")
        self.assertFalse(result.timed_out)

    def test_the_runner_timeout_maps_to_the_ceiling_report(self) -> None:
        fake = FakeRunner(timeout=True)
        with mock.patch.object(process, "_load_runner", return_value=fake):
            result = process.run_process(
                ["cargo", "bench"],
                working_directory=Path("/repo"),
                deadline=process.Deadline.after(30.0),
                termination_grace_seconds=1.5,
                capture=True,
            )
        self.assertTrue(result.timed_out)
        self.assertEqual(result.returncode, process.TIMEOUT_EXIT_CODE)

    def test_a_past_deadline_never_launches(self) -> None:
        expired = process.Deadline(time.monotonic() - 1.0)
        with mock.patch.object(process, "_load_runner") as loader:
            with self.assertRaises(process.ProcessDeadlineError):
                process.run_process(
                    ["cargo", "bench"],
                    working_directory=Path("."),
                    deadline=expired,
                    termination_grace_seconds=1.0,
                    capture=True,
                )
            loader.assert_not_called()

    def test_uncaptured_output_reaches_the_callers_streams(self) -> None:
        import io

        fake = FakeRunner(returncode=0)
        captured_out, captured_err = io.StringIO(), io.StringIO()
        with mock.patch.object(process, "_load_runner", return_value=fake), \
                mock.patch.object(sys, "stdout", captured_out), \
                mock.patch.object(sys, "stderr", captured_err):
            process.run_process(
                ["cargo", "bench"],
                working_directory=Path("."),
                deadline=process.Deadline.after(30.0),
                termination_grace_seconds=1.0,
                capture=False,
            )
        self.assertEqual(captured_out.getvalue(), "out")
        self.assertEqual(captured_err.getvalue(), "err")


class SuiteDeadlineTests(unittest.TestCase):
    def test_an_exhausted_suite_refuses_new_work(self) -> None:
        suite = process.Deadline.after(0.0)
        with self.assertRaises(process.ProcessDeadlineError):
            process.operation_deadline(suite, 10.0)

    def test_an_operation_never_outlives_its_suite(self) -> None:
        suite = process.Deadline.after(5.0)
        operation = process.operation_deadline(suite, 60.0)
        self.assertLessEqual(operation.ends_at, suite.ends_at)


if __name__ == "__main__":
    unittest.main()
