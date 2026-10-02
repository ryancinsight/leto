#!/usr/bin/env python3
"""Run every leto-ops Criterion binary under committed wall-clock budgets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from benchlib.budget import ConfigurationError, load_configuration, select_targets
from benchlib.cargo import (
    benchmark_command,
    check_target_inventory,
    read_declared_targets,
)
from benchlib.environment import cargo_context
from benchlib.process import (
    Deadline,
    ProcessDeadlineError,
    ProcessTreeError,
    TIMEOUT_EXIT_CODE,
    operation_deadline,
    run_process,
)

REPOSITORY = Path(__file__).resolve().parent.parent
CONFIG = REPOSITORY / ".config" / "bench.toml"


def execute(requested: Sequence[str]) -> int:
    """Run selected binaries within their ceilings and one suite deadline."""
    configuration = load_configuration(CONFIG)
    suite = Deadline.after(configuration.suite_seconds)
    with cargo_context(REPOSITORY) as context:
        declared = read_declared_targets(
            context,
            REPOSITORY,
            operation_deadline(suite, configuration.metadata_seconds),
            configuration.termination_grace_seconds,
        )
        check_target_inventory(declared, set(configuration.targets))
        targets = select_targets(requested, configuration)
        for target in targets:
            budget = configuration.targets[target]
            deadline = operation_deadline(suite, budget.ceiling_seconds)
            print(
                f"benchmark {target}: ceiling {budget.ceiling_seconds:g}s "
                f"({budget.criterion_seconds:g}s Criterion + "
                f"{budget.overhead_seconds:g}s overhead)",
                flush=True,
            )
            try:
                result = run_process(
                    benchmark_command(context, REPOSITORY, target),
                    working_directory=context.working_directory,
                    deadline=deadline,
                    termination_grace_seconds=configuration.termination_grace_seconds,
                    capture=False,
                )
            except (ProcessDeadlineError, ProcessTreeError) as error:
                print(f"error: benchmark {target} cleanup failed: {error}", file=sys.stderr)
                return TIMEOUT_EXIT_CODE
            if result.timed_out:
                print(
                    f"error: benchmark {target} exceeded its wall-clock ceiling",
                    file=sys.stderr,
                )
                return TIMEOUT_EXIT_CODE
            if result.returncode != 0:
                print(
                    f"error: benchmark {target} failed with exit code "
                    f"{result.returncode}",
                    file=sys.stderr,
                )
                return result.returncode
    if suite.remaining() <= 0:
        print("error: benchmark suite exceeded 300s including cleanup", file=sys.stderr)
        return TIMEOUT_EXIT_CODE
    return 0


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets",
        nargs="*",
        help="bench targets to run; omission runs the full committed suite",
    )
    parsed = parser.parse_args(arguments)
    try:
        return execute(parsed.targets)
    except (ConfigurationError, ProcessDeadlineError, ProcessTreeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("error: benchmark run interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
