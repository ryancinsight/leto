"""Bounded installed-wheel contracts for every supported release platform.

All values have exact binary representations and closed-form results. This
suite needs NumPy only; the full NumPy/SciPy differential suite runs separately.
Imports are mandatory so a missing or unloadable extension fails collection.
"""

import leto_python as leto
import numpy as np
import pytest


@pytest.mark.parametrize(
    "operation, expected",
    [
        ("add", [3.0, 6.0]),
        ("sub", [1.0, 2.0]),
        ("mul", [2.0, 8.0]),
        ("div", [2.0, 2.0]),
    ],
)
def test_float32_elementwise(operation, expected):
    left = np.array([[2.0, 4.0]], dtype=np.float32)
    right = np.array([[1.0, 2.0]], dtype=np.float32)
    result = np.asarray(getattr(leto, operation)(left, right))
    assert result.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(result, np.array([expected], dtype=np.float32))


def test_float32_matmul():
    left = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    right = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    result = np.asarray(leto.matmul(left, right))
    assert result.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(result, [[19.0, 22.0], [43.0, 50.0]])


def test_float64_solve():
    matrix = np.array([[2.0, 0.0], [0.0, 4.0]], dtype=np.float64)
    rhs = np.array([6.0, 20.0], dtype=np.float64)
    result = np.asarray(leto.solve(matrix, rhs))
    assert result.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(result, [3.0, 5.0])


def test_float64_inverse():
    matrix = np.array([[2.0, 0.0], [0.0, 4.0]], dtype=np.float64)
    result = np.asarray(leto.inv(matrix))
    assert result.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(result, [[0.5, 0.0], [0.0, 0.25]])


def test_matmul_rejects_incompatible_shapes():
    with pytest.raises(ValueError):
        leto.matmul(np.ones((2, 3), dtype=np.float32), np.ones((2, 2), dtype=np.float32))


def test_elementwise_rejects_wrong_dtype():
    wrong = np.array([[1.0, 2.0]], dtype=np.float64)
    with pytest.raises(TypeError):
        leto.add(wrong, wrong)
