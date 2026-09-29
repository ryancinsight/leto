"""Committed benchmark budget parsing and target selection."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

MAXIMUM_SUITE_SECONDS = 300


class ConfigurationError(ValueError):
    """The committed benchmark configuration is inconsistent."""


@dataclass(frozen=True)
class TargetBudget:
    """Nominal Criterion work plus bounded process overhead for one binary."""

    criterion_seconds: float
    overhead_seconds: float
    ceiling_seconds: float


@dataclass(frozen=True)
class BudgetConfiguration:
    """Validated suite and per-target wall-clock limits."""

    suite_seconds: float
    metadata_seconds: float
    termination_grace_seconds: float
    targets: dict[str, TargetBudget]


def _positive_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{field} must be a number")
    number = float(value)
    if not 0 < number < float("inf"):
        raise ConfigurationError(f"{field} must be positive and finite")
    return number


def _nonnegative_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{field} must be a number")
    number = float(value)
    if not 0 <= number < float("inf"):
        raise ConfigurationError(f"{field} must be nonnegative and finite")
    return number


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{field} must be a positive integer")
    return value


def parse_configuration(document: Mapping[str, Any]) -> BudgetConfiguration:
    """Validate a parsed benchmark budget document."""
    suite_seconds = _positive_number(document.get("suite_seconds"), "suite_seconds")
    metadata_seconds = _positive_number(document.get("metadata_seconds"), "metadata_seconds")
    termination_grace_seconds = _positive_number(
        document.get("termination_grace_seconds"), "termination_grace_seconds"
    )
    if suite_seconds > MAXIMUM_SUITE_SECONDS:
        raise ConfigurationError(
            f"suite_seconds is {suite_seconds:g}; maximum is {MAXIMUM_SUITE_SECONDS}"
        )
    raw_targets = document.get("targets")
    if not isinstance(raw_targets, Mapping) or not raw_targets:
        raise ConfigurationError("targets must be a non-empty table")

    targets: dict[str, TargetBudget] = {}
    for name, raw_budget in raw_targets.items():
        if not isinstance(name, str) or not name:
            raise ConfigurationError("target names must be non-empty strings")
        if not isinstance(raw_budget, Mapping):
            raise ConfigurationError(f"targets.{name} must be a table")
        case_count = _positive_integer(
            raw_budget.get("case_count"), f"targets.{name}.case_count"
        )
        _positive_integer(raw_budget.get("sample_size"), f"targets.{name}.sample_size")
        warm_up_seconds = _positive_number(
            raw_budget.get("warm_up_seconds"), f"targets.{name}.warm_up_seconds"
        )
        measurement_seconds = _positive_number(
            raw_budget.get("measurement_seconds"),
            f"targets.{name}.measurement_seconds",
        )
        additional_seconds = _nonnegative_number(
            raw_budget.get("additional_criterion_seconds"),
            f"targets.{name}.additional_criterion_seconds",
        )
        criterion_seconds = _positive_number(
            raw_budget.get("criterion_seconds"), f"targets.{name}.criterion_seconds"
        )
        overhead_seconds = _positive_number(
            raw_budget.get("overhead_seconds"), f"targets.{name}.overhead_seconds"
        )
        ceiling_seconds = _positive_number(
            raw_budget.get("ceiling_seconds"), f"targets.{name}.ceiling_seconds"
        )
        expected_criterion = (
            case_count * (warm_up_seconds + measurement_seconds) + additional_seconds
        )
        if abs(expected_criterion - criterion_seconds) > 1e-9:
            raise ConfigurationError(
                f"targets.{name}.criterion_seconds must equal case_count * "
                f"(warm_up_seconds + measurement_seconds) + "
                f"additional_criterion_seconds ({expected_criterion:g})"
            )
        expected_ceiling = criterion_seconds + overhead_seconds
        if abs(expected_ceiling - ceiling_seconds) > 1e-9:
            raise ConfigurationError(
                f"targets.{name}.ceiling_seconds must equal criterion_seconds + "
                f"overhead_seconds ({expected_ceiling:g})"
            )
        targets[name] = TargetBudget(
            criterion_seconds=criterion_seconds,
            overhead_seconds=overhead_seconds,
            ceiling_seconds=ceiling_seconds,
        )

    allocated = sum(budget.ceiling_seconds for budget in targets.values())
    if allocated > suite_seconds:
        raise ConfigurationError(
            f"target ceilings total {allocated:g}s, exceeding the "
            f"{suite_seconds:g}s suite budget"
        )
    return BudgetConfiguration(
        suite_seconds=suite_seconds,
        metadata_seconds=metadata_seconds,
        termination_grace_seconds=termination_grace_seconds,
        targets=targets,
    )


def load_configuration(path: Path) -> BudgetConfiguration:
    """Read and validate the committed TOML budget."""
    try:
        with path.open("rb") as stream:
            return parse_configuration(tomllib.load(stream))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError(f"cannot read {path}: {error}") from error


def select_targets(
    requested: Sequence[str], configuration: BudgetConfiguration
) -> list[str]:
    """Select requested targets or preserve committed order for the full suite."""
    if not requested:
        return list(configuration.targets)
    unknown = sorted(set(requested) - configuration.targets.keys())
    if unknown:
        raise ConfigurationError(f"unknown benchmark targets: {', '.join(unknown)}")
    if len(set(requested)) != len(requested):
        raise ConfigurationError("a benchmark target may be selected only once")
    return list(requested)
