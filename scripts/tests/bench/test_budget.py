"""Committed benchmark budget value tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
REPOSITORY = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from benchlib.budget import (
    ConfigurationError,
    TargetBudget,
    load_configuration,
    select_targets,
)


def test_budget_and_selected_target() -> None:
    configuration = load_configuration(REPOSITORY / ".config" / "bench.toml")
    assert sum(
        budget.ceiling_seconds for budget in configuration.targets.values()
    ) == 297
    assert configuration.suite_seconds == 300
    assert configuration.targets["col_piv_qr"] == TargetBudget(18, 7, 25)
    selected = select_targets(["col_piv_qr"], configuration)
    assert selected == ["col_piv_qr"]
    assert configuration.targets[selected[0]].ceiling_seconds == 25


def test_unknown_or_duplicate_target_is_rejected() -> None:
    configuration = load_configuration(REPOSITORY / ".config" / "bench.toml")
    with pytest.raises(ConfigurationError, match="unknown benchmark targets"):
        select_targets(["missing"], configuration)
    with pytest.raises(ConfigurationError, match="selected only once"):
        select_targets(["kernels", "kernels"], configuration)
