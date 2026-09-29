"""Cargo benchmark target discovery and command construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from benchlib.budget import ConfigurationError
from benchlib.environment import CargoContext
from benchlib.process import Deadline, ProcessTreeError, run_process

PACKAGE = "leto-ops"


def metadata_command(context: CargoContext, repository: Path) -> list[str]:
    return [
        *context.command_prefix,
        "metadata",
        "--locked",
        "--no-deps",
        "--format-version",
        "1",
        "--manifest-path",
        str(repository / "Cargo.toml"),
    ]


def benchmark_command(
    context: CargoContext, repository: Path, target: str
) -> list[str]:
    return [
        *context.command_prefix,
        "bench",
        "--locked",
        "-p",
        PACKAGE,
        "--bench",
        target,
        "--manifest-path",
        str(repository / "Cargo.toml"),
        "--",
        "--noplot",
    ]


def bench_targets_from_metadata(metadata: Mapping[str, Any]) -> set[str]:
    packages = metadata.get("packages")
    if not isinstance(packages, list):
        raise ConfigurationError("Cargo metadata has no packages array")
    package = next(
        (
            candidate
            for candidate in packages
            if isinstance(candidate, Mapping) and candidate.get("name") == PACKAGE
        ),
        None,
    )
    if package is None:
        raise ConfigurationError(f"Cargo metadata has no {PACKAGE} package")
    targets = package.get("targets")
    if not isinstance(targets, list):
        raise ConfigurationError(f"Cargo metadata has no targets for {PACKAGE}")
    return {
        str(target["name"])
        for target in targets
        if isinstance(target, Mapping)
        and isinstance(target.get("kind"), list)
        and "bench" in target["kind"]
        and isinstance(target.get("name"), str)
    }


def check_target_inventory(declared: set[str], configured: set[str]) -> None:
    missing = sorted(declared - configured)
    obsolete = sorted(configured - declared)
    if not missing and not obsolete:
        return
    details = []
    if missing:
        details.append(f"unbudgeted Cargo targets: {', '.join(missing)}")
    if obsolete:
        details.append(f"budgets without Cargo targets: {', '.join(obsolete)}")
    raise ConfigurationError("benchmark target inventory differs: " + "; ".join(details))


def read_declared_targets(
    context: CargoContext,
    repository: Path,
    deadline: Deadline,
    termination_grace_seconds: float,
) -> set[str]:
    try:
        result = run_process(
            metadata_command(context, repository),
            working_directory=context.working_directory,
            deadline=deadline,
            termination_grace_seconds=termination_grace_seconds,
            capture=True,
        )
    except ProcessTreeError as error:
        raise ConfigurationError(f"Cargo metadata cleanup failed: {error}") from error
    if result.timed_out:
        raise ConfigurationError("Cargo metadata exceeded its committed deadline")
    if result.returncode != 0:
        diagnostic = result.stderr.strip() or result.stdout.strip()
        raise ConfigurationError(
            f"Cargo metadata failed with exit code {result.returncode}: {diagnostic}"
        )
    try:
        metadata = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ConfigurationError(f"Cargo metadata emitted invalid JSON: {error}") from error
    if not isinstance(metadata, Mapping):
        raise ConfigurationError("Cargo metadata root is not an object")
    return bench_targets_from_metadata(metadata)
