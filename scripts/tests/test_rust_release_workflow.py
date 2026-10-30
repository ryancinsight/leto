"""Executable contract for Leto's unarmed Rust release workflow."""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "rust-release.yml"
ATLAS = "https://github.com/ryancinsight/atlas.git"
COMMAND_TIMEOUT_SECONDS = 60
PENDING_REVISION = "0227830912a12dbae0ff62b3f16575899fc9087e"


def yaml_block(text: str, heading: str, indent: int) -> str:
    """Return one indentation-delimited YAML mapping entry."""
    lines = text.splitlines()
    marker = f"{' ' * indent}{heading}"
    start = lines.index(marker)
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        stripped = line.lstrip()
        indentation = len(line) - len(stripped)
        if stripped and not stripped.startswith("#") and indentation <= indent:
            end = index
            break
    return "\n".join(lines[start:end])


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one bounded command and retain complete diagnostics."""
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        check=False,
    )


class RustReleaseWorkflowTests(unittest.TestCase):
    def test_automatic_publisher_receives_only_the_empty_cohort(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        pending = yaml_block(text, "publish-pending:", 2)
        self.assertIn(
            f"crates-publish-pending.yml@{PENDING_REVISION}",
            pending,
        )
        self.assertIn(
            "if: github.event_name == 'push' || github.event_name == 'schedule'",
            pending,
        )
        self.assertRegex(pending, r"(?m)^      cohort-json: '\{\}'$")
        self.assertEqual(text.count("cohort-json: '{}'"), 1)

        manual = yaml_block(text, "publish:", 2)
        self.assertIn("github.event_name == 'workflow_dispatch'", manual)
        self.assertIn("startsWith(github.event.release.tag_name, 'crate-')", manual)
        self.assertNotIn("cohort-json", manual)

    def test_empty_cohort_stops_before_registry_authentication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = root / "provider.git"
            initialized = run(["git", "init", "--bare", str(provider)], cwd=root)
            self.assertEqual(initialized.returncode, 0, initialized.stderr)

            fetched = run(
                [
                    "git",
                    "-C",
                    str(provider),
                    "fetch",
                    "--depth=1",
                    "--no-tags",
                    ATLAS,
                    PENDING_REVISION,
                ],
                cwd=root,
            )
            self.assertEqual(fetched.returncode, 0, fetched.stderr)
            revision = run(
                ["git", "-C", str(provider), "rev-parse", "FETCH_HEAD"],
                cwd=root,
            )
            self.assertEqual(revision.returncode, 0, revision.stderr)
            self.assertEqual(revision.stdout.strip(), PENDING_REVISION)

            shared = run(
                [
                    "git",
                    "-C",
                    str(provider),
                    "show",
                    "FETCH_HEAD:.github/workflows/crates-publish-pending.yml",
                ],
                cwd=root,
            )
            self.assertEqual(shared.returncode, 0, shared.stderr)
            action_pin = re.search(
                r"actions/crates-pending@([0-9a-f]{40})", shared.stdout
            )
            if action_pin is None:
                self.fail("shared workflow must pin the crates-pending action")

            plan = yaml_block(shared.stdout, "plan:", 2)
            semver = yaml_block(shared.stdout, "semver:", 2)
            publish = yaml_block(shared.stdout, "publish:", 2)
            self.assertNotIn("crates-io-auth-action", plan)
            self.assertNotIn("crates-io-auth-action", semver)
            self.assertIn("cohort-json: ${{ inputs.cohort-json }}", plan)
            self.assertIn("if: needs.plan.outputs.count != '0'", semver)
            self.assertIn("if: needs.plan.outputs.count != '0'", publish)
            self.assertIn("crates-io-auth-action@", publish)
            self.assertIn("command: publish", publish)
            self.assertIn("command: tag", publish)
            self.assertEqual(publish.count("cohort-json: ${{ inputs.cohort-json }}"), 2)

            action_fetch = run(
                [
                    "git",
                    "-C",
                    str(provider),
                    "fetch",
                    "--depth=1",
                    "--no-tags",
                    ATLAS,
                    action_pin.group(1),
                ],
                cwd=root,
            )
            self.assertEqual(action_fetch.returncode, 0, action_fetch.stderr)
            archived = subprocess.run(
                [
                    "git",
                    "-C",
                    str(provider),
                    "archive",
                    "FETCH_HEAD",
                    ".github/actions/crates-pending",
                ],
                cwd=root,
                capture_output=True,
                timeout=COMMAND_TIMEOUT_SECONDS,
                check=False,
            )
            self.assertEqual(
                archived.returncode,
                0,
                archived.stderr.decode(errors="replace"),
            )
            with tarfile.open(fileobj=io.BytesIO(archived.stdout)) as archive:
                member = archive.getmember(
                    ".github/actions/crates-pending/crates_pending.py"
                )
                source = archive.extractfile(member)
                if source is None:
                    self.fail("pinned action archive lacks its executable helper")
                helper = root / "crates_pending.py"
                helper.write_bytes(source.read())

            workspace = root / "empty-workspace"
            workspace.mkdir()
            output = root / "github-output"
            environment = os.environ.copy()
            environment["GITHUB_OUTPUT"] = str(output)
            planned = run(
                [
                    sys.executable,
                    str(helper),
                    "plan",
                    "--owner",
                    "ryancinsight",
                    "--cohort-json",
                    "{}",
                ],
                cwd=workspace,
                env=environment,
            )
            self.assertEqual(planned.returncode, 0, planned.stderr)
            self.assertEqual(planned.stdout, "")
            self.assertEqual(output.read_text(encoding="utf-8"), "count=0\n")
            self.assertEqual(list(workspace.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
