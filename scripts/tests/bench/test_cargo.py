"""Cargo inventory and configuration value tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .support import REPOSITORY
from benchlib.budget import ConfigurationError
from benchlib.cargo import (
    bench_targets_from_metadata,
    benchmark_command,
    check_target_inventory,
    read_declared_targets,
)
from benchlib.environment import CargoContext, OVERLAY_BEGIN, cargo_context
from benchlib.process import Deadline, ProcessResult


class CargoTests(unittest.TestCase):
    def test_target_inventory_must_match_budget_map(self) -> None:
        metadata = {
            "packages": [
                {
                    "name": "leto-ops",
                    "targets": [
                        {"name": "timing", "kind": ["bench"]},
                        {"name": "library", "kind": ["lib"]},
                    ],
                },
                {"name": "other", "targets": [{"name": "foreign", "kind": ["bench"]}]},
            ]
        }
        self.assertEqual(bench_targets_from_metadata(metadata), {"timing"})
        with self.assertRaisesRegex(
            ConfigurationError, "unbudgeted Cargo targets: declared"
        ):
            check_target_inventory({"declared"}, {"configured"})

    def test_metadata_shape_is_validated(self) -> None:
        for metadata in ({}, {"packages": "invalid"}, {"packages": []}):
            with self.subTest(metadata=metadata):
                with self.assertRaises(ConfigurationError):
                    bench_targets_from_metadata(metadata)

    def test_metadata_failures_preserve_process_outcomes(self) -> None:
        cargo_context_value = CargoContext(REPOSITORY, ("cargo",))
        cases = (
            (
                ProcessResult(0, "{", "", False),
                "invalid JSON",
            ),
            (
                ProcessResult(17, "", "metadata rejected", False),
                "exit code 17: metadata rejected",
            ),
            (
                ProcessResult(124, "", "", True),
                "exceeded its committed deadline",
            ),
        )
        for result, message in cases:
            with self.subTest(message=message):
                with patch("benchlib.cargo.run_process", return_value=result):
                    with self.assertRaisesRegex(ConfigurationError, message):
                        read_declared_targets(
                            cargo_context_value,
                            REPOSITORY,
                            Deadline.after(3),
                            1,
                        )

    def test_cargo_context_preserves_toolchain_stack_and_member_env(self) -> None:
        with tempfile.TemporaryDirectory(prefix="leto-bench-context-") as temporary:
            stack = Path(temporary)
            repository = stack / "repos" / "leto"
            (stack / ".cargo").mkdir()
            (repository / ".cargo").mkdir(parents=True)
            (stack / ".cargo" / "config.toml").write_text(
                '[build]\ntarget-dir = "target"\n[profile.release]\nstrip = "symbols"\n'
                f'{OVERLAY_BEGIN}\n[patch."https://example.invalid"]\n'
                'provider = { path = "provider" }\n',
                encoding="utf-8",
            )
            member_configuration = repository / ".cargo" / "config.toml"
            member_configuration.write_text(
                '[env]\nPYO3_USE_ABI3_FORWARD_COMPATIBILITY = "1"\n',
                encoding="utf-8",
            )
            (repository / "rust-toolchain.toml").write_text(
                '[toolchain]\nchannel = "1.97.0"\n', encoding="utf-8"
            )
            with cargo_context(repository) as context:
                self.assertEqual(
                    context.command_prefix,
                    (
                        "rustup",
                        "run",
                        "1.97.0",
                        "cargo",
                        "--config",
                        str(member_configuration),
                    ),
                )
                rendered = (
                    context.working_directory / ".cargo" / "config.toml"
                ).read_text(encoding="utf-8")
                expected_target = (stack / "target").resolve().as_posix()
                self.assertIn(f'target-dir = "{expected_target}"', rendered)
                self.assertIn('[profile.release]\nstrip = "symbols"', rendered)
                self.assertNotIn("[patch.", rendered)
                command = benchmark_command(context, REPOSITORY, "col_piv_qr")
                self.assertNotIn("--target-dir", command)
                self.assertNotIn("--profile", command)
                self.assertTrue(
                    member_configuration.read_text(encoding="utf-8").startswith("[env]")
                )


if __name__ == "__main__":
    unittest.main()
