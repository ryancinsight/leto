"""Committed benchmark budget value tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2]
REPOSITORY = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from benchlib.budget import (  # noqa: E402
    BudgetConfiguration,
    ConfigurationError,
    TargetBudget,
    load_configuration,
    parse_configuration,
    select_targets,
)


def valid_document() -> dict[str, object]:
    return {
        "suite_seconds": 20,
        "metadata_seconds": 3,
        "termination_grace_seconds": 2,
        "targets": {
            "kernel": {
                "case_count": 2,
                "sample_size": 10,
                "warm_up_seconds": 1,
                "measurement_seconds": 3,
                "criterion_seconds": 8,
                "overhead_seconds": 2,
                "ceiling_seconds": 10,
            }
        },
    }


class BudgetTests(unittest.TestCase):
    def test_committed_budget_and_selected_target(self) -> None:
        configuration = load_configuration(REPOSITORY / ".config" / "bench.toml")
        self.assertEqual(sum(
            budget.ceiling_seconds for budget in configuration.targets.values()
        ), 297)
        self.assertEqual(configuration.suite_seconds, 300)
        self.assertEqual(configuration.targets["col_piv_qr"], TargetBudget(18, 7, 25))
        self.assertEqual(configuration.targets["kernels"].criterion_seconds, 111)
        self.assertEqual(select_targets(["col_piv_qr"], configuration), ["col_piv_qr"])

    def test_valid_budget_parses_to_typed_values(self) -> None:
        configuration = parse_configuration(valid_document())
        self.assertIsInstance(configuration, BudgetConfiguration)
        self.assertEqual(configuration.targets["kernel"], TargetBudget(8, 2, 10))

    def test_unknown_or_duplicate_target_is_rejected(self) -> None:
        configuration = load_configuration(REPOSITORY / ".config" / "bench.toml")
        with self.assertRaisesRegex(ConfigurationError, "unknown benchmark targets"):
            select_targets(["missing"], configuration)
        with self.assertRaisesRegex(ConfigurationError, "selected only once"):
            select_targets(["kernels", "kernels"], configuration)

    def test_nonpositive_nonfinite_and_boolean_limits_are_rejected(self) -> None:
        cases = [
            ("suite_seconds", 0),
            ("suite_seconds", float("nan")),
            ("suite_seconds", float("inf")),
            ("metadata_seconds", True),
            ("termination_grace_seconds", -1),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                document = valid_document()
                document[field] = value
                with self.assertRaises(ConfigurationError):
                    parse_configuration(document)

    def test_target_counts_and_durations_are_validated(self) -> None:
        cases = [
            ("case_count", 0),
            ("sample_size", -1),
            ("warm_up_seconds", float("nan")),
            ("measurement_seconds", 0),
            ("overhead_seconds", 0),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                document = valid_document()
                document["targets"]["kernel"][field] = value
                with self.assertRaises(ConfigurationError):
                    parse_configuration(document)

    def test_group_timing_overrides_account_for_case_counts(self) -> None:
        document = valid_document()
        document["suite_seconds"] = 50
        document["targets"]["kernel"].update(
            {
                "case_count": 8,
                "groups": {
                    "operator_chain": {
                        "case_count": 2,
                        "sample_size": 50,
                        "warm_up_seconds": 1,
                        "measurement_seconds": 5,
                    }
                },
                "criterion_seconds": 36,
                "ceiling_seconds": 38,
            }
        )
        configuration = parse_configuration(document)
        self.assertEqual(configuration.targets["kernel"], TargetBudget(36, 2, 38))

        document["targets"]["kernel"]["groups"]["operator_chain"][
            "case_count"
        ] = 9
        with self.assertRaisesRegex(ConfigurationError, "exceeding case_count"):
            parse_configuration(document)

    def test_formula_mismatch_and_suite_overallocation_are_rejected(self) -> None:
        document = valid_document()
        document["targets"]["kernel"]["criterion_seconds"] = 9
        with self.assertRaisesRegex(ConfigurationError, "sum of each case group's"):
            parse_configuration(document)

        document = valid_document()
        document["targets"]["kernel"]["ceiling_seconds"] = 11
        with self.assertRaisesRegex(ConfigurationError, "ceiling_seconds must equal"):
            parse_configuration(document)

        document = valid_document()
        document["suite_seconds"] = 10
        with self.assertRaisesRegex(ConfigurationError, "ceilings plus metadata"):
            parse_configuration(document)

    def test_suite_limit_and_empty_target_map_are_rejected(self) -> None:
        document = valid_document()
        document["suite_seconds"] = 301
        with self.assertRaisesRegex(ConfigurationError, "maximum is"):
            parse_configuration(document)

        document = valid_document()
        document["targets"] = {}
        with self.assertRaisesRegex(ConfigurationError, "non-empty table"):
            parse_configuration(document)

    def test_malformed_toml_is_reported_as_configuration_error(self) -> None:
        with tempfile.TemporaryDirectory(prefix="leto-bench-invalid-") as temporary:
            path = Path(temporary) / "bench.toml"
            path.write_text("[targets\\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "cannot read"):
                load_configuration(path)


if __name__ == "__main__":
    unittest.main()
