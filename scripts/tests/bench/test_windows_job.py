"""Windows Job Object failure behavior."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object behavior")
def test_closed_job_fails_closed() -> None:
    from benchlib.windows_job import WindowsJob, WindowsJobError

    job = WindowsJob()
    job.close()
    with pytest.raises(WindowsJobError, match="already closed"):
        job.terminate(1)
