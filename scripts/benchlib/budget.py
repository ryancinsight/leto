"""Committed benchmark budget parsing and target selection."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from decimal import Decimal
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
        raw_groups = raw_budget.get("groups", {})
        if not isinstance(raw_groups, Mapping):
            raise ConfigurationError(f"targets.{name}.groups must be a table")
        group_cases = 0
        expected_criterion = Decimal(0)
        for group_name, raw_group in raw_groups.items():
            if not isinstance(group_name, str) or not group_name:
                raise ConfigurationError(
                    f"targets.{name}.groups names must be non-empty strings"
                )
            if not isinstance(raw_group, Mapping):
                raise ConfigurationError(
                    f"targets.{name}.groups.{group_name} must be a table"
                )
            cases = _positive_integer(
                raw_group.get("case_count"),
                f"targets.{name}.groups.{group_name}.case_count",
            )
            _positive_integer(
                raw_group.get("sample_size"),
                f"targets.{name}.groups.{group_name}.sample_size",
            )
            group_warm_up = _positive_number(
                raw_group.get("warm_up_seconds"),
                f"targets.{name}.groups.{group_name}.warm_up_seconds",
            )
            group_measurement = _positive_number(
                raw_group.get("measurement_seconds"),
                f"targets.{name}.groups.{group_name}.measurement_seconds",
            )
            group_cases += cases
            expected_criterion += Decimal(cases) * (
                Decimal(str(group_warm_up)) + Decimal(str(group_measurement))
            )
        if group_cases > case_count:
            raise ConfigurationError(
                f"targets.{name}.groups account for {group_cases} cases, "
                f"exceeding case_count {case_count}"
            )
        expected_criterion += Decimal(case_count - group_cases) * (
            Decimal(str(warm_up_seconds)) + Decimal(str(measurement_seconds))
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
        if Decimal(str(criterion_seconds)) != expected_criterion:
            raise ConfigurationError(
                f"targets.{name}.criterion_seconds must equal the sum of each "
                f"case group's warm_up_seconds + measurement_seconds "
                f"({expected_criterion:g})"
            )
        expected_ceiling = Decimal(str(criterion_seconds)) + Decimal(
            str(overhead_seconds)
        )
        if Decimal(str(ceiling_seconds)) != expected_ceiling:
            raise ConfigurationError(
                f"targets.{name}.ceiling_seconds must equal criterion_seconds + "
                f"overhead_seconds ({expected_ceiling:g})"
            )
        targets[name] = TargetBudget(
            criterion_seconds=criterion_seconds,
            overhead_seconds=overhead_seconds,
            ceiling_seconds=ceiling_seconds,
        )

    target_seconds = sum(
        (Decimal(str(budget.ceiling_seconds)) for budget in targets.values()),
        start=Decimal(0),
    )
    allocated_seconds = target_seconds + Decimal(str(metadata_seconds))
    if allocated_seconds > Decimal(str(suite_seconds)):
        raise ConfigurationError(
            f"target ceilings plus metadata total {allocated_seconds:g}s, "
            f"exceeding the {suite_seconds:g}s suite budget"
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
