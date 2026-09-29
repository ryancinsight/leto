from __future__ import annotations

import os
import subprocess
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


def process_running(process_id: int) -> bool:
    if os.name == "nt":
        import ctypes

        query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            query_limited_information, False, process_id
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    status = Path(f"/proc/{process_id}/stat")
    if status.is_file():
        return status.read_text(encoding="utf-8").split()[2] != "Z"
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    return True


def blocking_tree(parent_exits: bool, pid_file: Path | None = None) -> str:
    child_source = "import threading; threading.Event().wait()"
    statements = [
        "import subprocess, sys, threading",
        f"child = subprocess.Popen({[sys.executable, '-c', child_source]!r})",
        "print(child.pid, flush=True)",
    ]
    if pid_file is not None:
        statements.append(
            f"__import__('pathlib').Path({str(pid_file)!r}).write_text("
            "f'{__import__(\"os\").getpid()} {child.pid}', encoding='utf-8')"
        )
    if not parent_exits:
        statements.append("threading.Event().wait()")
    return "; ".join(statements)


def cleanup_tree(parent_id: int, child_id: int) -> None:
    if os.name == "nt" and process_running(parent_id):
        subprocess.run(
            ["taskkill", "/PID", str(parent_id), "/T", "/F"],
            capture_output=True,
            check=False,
            timeout=3,
        )
    elif os.name != "nt" and process_running(child_id):
        os.killpg(parent_id, 9)
