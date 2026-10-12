from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2]
REPOSITORY = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from benchlib.process import Deadline, run_process  # noqa: E402


def python_command(source: str) -> list[str]:
    return [sys.executable, "-c", source]


def run_python(source: str, *, capture: bool = True):
    return run_process(
        python_command(source),
        working_directory=REPOSITORY,
        deadline=Deadline.after(3),
        termination_grace_seconds=2,
        capture=capture,
    )
