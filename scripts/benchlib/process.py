"""Bounded benchmark execution delegating to the Atlas process-tree runner.

Process-tree ownership is an Atlas-owned capability (ADR 0067): the kill-on-close
Job Object, the process-group supervisor, the interrupt deferral, and the
deadline machinery live once in Atlas's `scripts/process_tree.py`, and this
module is the pinned-checkout consumer that ADR names — it imports the runner
from the Atlas tree at `ATLAS_PIN`, verified at that exact commit before
import, never a moving branch.

The resolution order matches the two checkouts the runner serves:

- inside the Atlas stack, the Atlas checkout is the tree containing this
  repository (`<atlas>/repos/leto/scripts/benchlib/process.py`);
- standalone CI materializes a sparse Atlas checkout and names it through
  `ATLAS_TOOLING_CHECKOUT`.

The benchmark-facing surface stays unchanged: absolute deadlines, the suite
ceiling, and the typed failures the runner reports.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

TIMEOUT_EXIT_CODE = 124

#: The Atlas revision this consumer pins. Advancing it is a reviewed change:
#: the runner's behavior at the new revision is the behavior every bounded
#: benchmark run inherits.
ATLAS_PIN = "b94ee304b83fbdc1a1542af66ba4668ebda2ae15"

_PROCESS_TREE_RELATIVE = "scripts/process_tree.py"


class ProcessDeadlineError(RuntimeError):
    """A child could not start or finish inside its absolute deadline."""


class ProcessTreeError(RuntimeError):
    """The pinned Atlas runner could not be resolved or its run failed."""


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
    """The earlier of the suite ceiling and this operation's own budget."""
    now = time.monotonic()
    if suite.remaining() <= 0:
        raise ProcessDeadlineError("suite wall-clock budget is exhausted")
    return Deadline(min(suite.ends_at, now + ceiling_seconds))


def atlas_checkout() -> Path:
    """Resolve the pinned Atlas checkout: the override, else the stack tree."""
    override = os.environ.get("ATLAS_TOOLING_CHECKOUT")
    if override:
        return Path(override)
    # <atlas>/repos/leto/scripts/benchlib/process.py -> the Atlas root.
    return Path(__file__).resolve().parents[4]


def _verify_pin(root: Path) -> None:
    """Refuse an Atlas checkout that is not exactly the pinned commit."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if revision.returncode != 0:
        raise ProcessTreeError(f"no Atlas checkout at {root}: {revision.stderr.strip()}")
    head = revision.stdout.strip()
    if head != ATLAS_PIN:
        raise ProcessTreeError(
            f"Atlas checkout at {root} is {head}, not the pinned {ATLAS_PIN}"
        )


def _load_runner(root: Path):
    """Import the Atlas runner from the verified checkout."""
    _verify_pin(root)
    module_path = root / _PROCESS_TREE_RELATIVE
    if not module_path.is_file():
        raise ProcessTreeError(f"the Atlas runner is missing at {module_path}")
    spec = importlib.util.spec_from_file_location("atlas_process_tree", module_path)
    if spec is None or spec.loader is None:
        raise ProcessTreeError(f"the Atlas runner at {module_path} is not importable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # The runner imports its platform sibling (`windows_process`) by module
    # name, so its own directory must lead the import path for the load.
    sys.path.insert(0, str(module_path.parent))
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise ProcessTreeError(f"the Atlas runner failed to load: {error}") from error
    finally:
        sys.path.remove(str(module_path.parent))
    return module


def run_process(
    command: Sequence[str],
    *,
    working_directory: Path,
    deadline: Deadline,
    termination_grace_seconds: float,
    capture: bool,
) -> ProcessResult:
    """Run a benchmark command through the pinned Atlas tree-owning runner.

    The Atlas runner owns cleanup budgets and interrupt handling; the
    termination grace this repository's configuration names is folded into the
    runner's own cleanup time, which reserves a finite slice for exactly that.
    Output is always captured by the runner and replayed here when
    ``capture=False`` requests the caller's streams.
    """
    del termination_grace_seconds
    remaining = deadline.remaining()
    if remaining <= 0:
        raise ProcessDeadlineError("process start is past its deadline")
    runner = _load_runner(atlas_checkout())
    try:
        completed = runner.run(list(command), cwd=working_directory, timeout=remaining)
    except runner.ProcessTreeTimeout as timeout:
        # The runner cleaned the tree up; the ceiling report is a value the
        # caller prints, not an exception.
        return ProcessResult(
            returncode=TIMEOUT_EXIT_CODE,
            stdout="",
            stderr=str(timeout),
            timed_out=True,
        )
    except ValueError as error:
        raise ProcessTreeError(f"the Atlas runner rejected the request: {error}") from error

    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    if not capture:
        _replay(sys.stdout, stdout)
        _replay(sys.stderr, stderr)
    return ProcessResult(
        returncode=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
    )


def _replay(stream, text: str) -> None:
    """Write captured output through, replacing what the stream cannot encode.

    Criterion's report text is UTF-8 and includes typographic signs (the minus
    U+2212 among them) that a Windows console code page cannot encode; the
    benchmark's own output is data, never a reason for the bounded run to
    crash.
    """
    encoding = getattr(stream, "encoding", None) or "utf-8"
    stream.write(text.encode(encoding, errors="replace").decode(encoding))
    stream.flush()
