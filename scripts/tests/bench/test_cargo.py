"""Cargo inventory and configuration value tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from .support import REPOSITORY
from benchlib.budget import ConfigurationError
from benchlib.cargo import (
    bench_targets_from_metadata,
    benchmark_command,
    check_target_inventory,
)
from benchlib.environment import OVERLAY_BEGIN, cargo_context


def test_target_inventory_must_match_budget_map() -> None:
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
    assert bench_targets_from_metadata(metadata) == {"timing"}
    with pytest.raises(ConfigurationError, match="unbudgeted Cargo targets: declared"):
        check_target_inventory({"declared"}, {"configured"})


def test_cargo_context_preserves_toolchain_stack_and_member_env() -> None:
    with tempfile.TemporaryDirectory(prefix="leto-bench-context-") as temporary:
        stack = Path(temporary)
        repository = stack / "repos" / "leto"
        (stack / ".cargo").mkdir()
        (repository / ".cargo").mkdir(parents=True)
        (stack / ".cargo" / "config.toml").write_text(
            '[build]\ntarget-dir = "target"\n[profile.release]\nstrip = "symbols"\n'
            f"{OVERLAY_BEGIN}\n[patch.\"https://example.invalid\"]\n"
            'provider = { path = "provider" }\n',
            encoding="utf-8",
        )
        member_configuration = repository / ".cargo" / "config.toml"
        member_configuration.write_text(
            '[env]\nPYO3_USE_ABI3_FORWARD_COMPATIBILITY = "1"\n', encoding="utf-8"
        )
        (repository / "rust-toolchain.toml").write_text(
            '[toolchain]\nchannel = "1.97.0"\n', encoding="utf-8"
        )
        with cargo_context(repository) as context:
            assert context.command_prefix == (
                "rustup",
                "run",
                "1.97.0",
                "cargo",
                "--config",
                str(member_configuration),
            )
            rendered = (context.working_directory / ".cargo" / "config.toml").read_text(
                encoding="utf-8"
            )
            expected_target = (stack / "target").resolve().as_posix()
            assert f'target-dir = "{expected_target}"' in rendered
            assert '[profile.release]\nstrip = "symbols"' in rendered
            assert "[patch." not in rendered
            command = benchmark_command(context, REPOSITORY, "col_piv_qr")
            assert "--target-dir" not in command
            assert "--profile" not in command
            assert member_configuration.read_text(encoding="utf-8").startswith("[env]")
