"""NumPy / SciPy output-parity tests for the ``leto_python`` bindings.

Each test verifies that ``leto_python`` produces numerically equivalent results
to the reference NumPy / SciPy operation on identical inputs. The whole module
skips automatically when ``leto_python`` is not installed.

``leto_python`` dtype contract (mirrors the Rust binding signatures):
- elementwise (``add``/``sub``/``mul``/``div``) and ``matmul`` take ``float32``;
- linear-algebra ops (``det``/``inv``/``solve``/``norm``/``trace``/``kron``) and
  the decompositions (``cholesky``/``qr``/``svd``/``symmetric_eigen``/``matexp``)
  take ``float64``.

Sign- and order-ambiguous decompositions (QR, Cholesky, eigh, SVD) are verified
by their mathematical invariants — reconstruction, orthonormality, and sorted
spectra — never by raw factor equality, since the factorization is only unique
up to signs/permutations.

Run via::

    pytest crates/leto-python/tests/test_numpy_parity.py -v
"""

import numpy as np
import pytest

leto = pytest.importorskip("leto_python")
scipy_linalg = pytest.importorskip("scipy.linalg")

# ---------------------------------------------------------------------------
# Fixtures: deterministic, well-conditioned reference matrices.
# ---------------------------------------------------------------------------

# Symmetric positive-definite (for Cholesky / symmetric_eigen).
_SPD = np.array([[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]], dtype=np.float64)
# General nonsingular (for det / inv / solve / qr / svd / matexp / schur).
_GEN = np.array([[3.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 3.0]], dtype=np.float64)
_RHS = np.array([1.0, 2.0, 3.0], dtype=np.float64)
# Rectangular f32 operands (for elementwise / matmul).
_F32_A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
_F32_B = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
_F32_VEC = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)  # 1D (for dot)
_F32_3D = np.arange(24, dtype=np.float32).reshape(2, 3, 4)  # N-d (for sum_dyn)
# Symmetric indefinite (for Bunch-Kaufman LDL^T).
_SYM_INDEF = np.array(
    [[4.0, 1.0, 2.0], [1.0, -3.0, 1.0], [2.0, 1.0, 5.0]], dtype=np.float64
)


def _close(label, got, expected, atol):
    got_a = np.asarray(got)
    exp_a = np.asarray(expected)
    assert got_a.shape == exp_a.shape, (
        f"{label}: shape {got_a.shape} != {exp_a.shape}"
    )
    diff = float(np.max(np.abs(got_a - exp_a))) if got_a.size else 0.0
    assert diff <= atol, f"{label}: max|diff|={diff:.3e} > atol={atol:.3e}"


# ---------------------------------------------------------------------------
# Elementwise + matmul (float32)
# ---------------------------------------------------------------------------


def test_elementwise_matches_numpy() -> None:
    a, b = _F32_A, _F32_A
    _close("add", leto.add(a, b), a + b, atol=1e-5)
    _close("sub", leto.sub(a, b), a - b, atol=1e-5)
    _close("mul", leto.mul(a, b), a * b, atol=1e-5)
    _close("div", leto.div(a, b), a / b, atol=1e-5)


def test_matmul_matches_numpy() -> None:
    _close("matmul", leto.matmul(_F32_A, _F32_B), _F32_A @ _F32_B, atol=1e-5)


# ---------------------------------------------------------------------------
# Reductions / vector dot (float32)
# ---------------------------------------------------------------------------


def test_sum_matches_numpy() -> None:
    _close("sum", [leto.sum(_F32_A)], [float(np.sum(_F32_A))], atol=1e-4)


def test_sum_dyn_matches_numpy() -> None:
    # Dynamic-rank sum over an N-d array.
    _close("sum_dyn", [leto.sum_dyn(_F32_3D)], [float(np.sum(_F32_3D))], atol=1e-2)


def test_dot_matches_numpy() -> None:
    _close("dot", [leto.dot(_F32_VEC, _F32_VEC)], [float(np.dot(_F32_VEC, _F32_VEC))], atol=1e-4)


# ---------------------------------------------------------------------------
# Linear algebra (float64): det / inv / solve / trace / kron / norm
# ---------------------------------------------------------------------------


def test_det_matches_numpy() -> None:
    _close("det", [leto.det(_GEN)], [np.linalg.det(_GEN)], atol=1e-9)


def test_inv_matches_numpy() -> None:
    _close("inv", leto.inv(_GEN), np.linalg.inv(_GEN), atol=1e-9)
    # A @ inv(A) == I (round-trip invariant).
    _close("inv_roundtrip", _GEN @ np.asarray(leto.inv(_GEN)), np.eye(3), atol=1e-9)


def test_solve_matches_numpy() -> None:
    x = np.asarray(leto.solve(_GEN, _RHS))
    _close("solve", x, np.linalg.solve(_GEN, _RHS), atol=1e-8)
    _close("solve_residual", _GEN @ x, _RHS, atol=1e-8)


def test_trace_matches_numpy() -> None:
    _close("trace", [leto.trace(_GEN)], [np.trace(_GEN)], atol=1e-9)


def test_kron_matches_numpy() -> None:
    _close("kron", leto.kron(_GEN, _GEN), np.kron(_GEN, _GEN), atol=1e-9)


def test_norm_matches_numpy() -> None:
    _close("norm_fro", [leto.norm(_SPD, "fro")], [np.linalg.norm(_SPD, "fro")], atol=1e-9)


# ---------------------------------------------------------------------------
# Decompositions (float64): verified by invariants, not raw factors.
# ---------------------------------------------------------------------------


def test_cholesky_matches_numpy() -> None:
    # leto returns the lower factor L with A == L @ L.T.
    lower = np.asarray(leto.cholesky(_SPD))
    _close("cholesky_lower_is_lower", np.triu(lower, 1), np.zeros((3, 3)), atol=1e-12)
    _close("cholesky_reconstruct", lower @ lower.T, _SPD, atol=1e-8)


def test_qr_matches_numpy() -> None:
    q, r = leto.qr(_GEN)
    q = np.asarray(q)
    r = np.asarray(r)
    _close("qr_reconstruct", q @ r, _GEN, atol=1e-8)
    _close("qr_orthonormal", q.T @ q, np.eye(q.shape[1]), atol=1e-8)
    _close("qr_upper", np.tril(r, -1), np.zeros_like(r), atol=1e-8)


def test_singular_values_match_numpy() -> None:
    got = sorted(np.asarray(leto.singular_values(_GEN)).flatten(), reverse=True)
    exp = sorted(np.linalg.svd(_GEN, compute_uv=False), reverse=True)
    _close("singular_values", got, exp, atol=1e-8)


def test_symmetric_eigen_matches_numpy() -> None:
    vals, vecs = leto.symmetric_eigen(_SPD)
    vals = np.asarray(vals).flatten()
    vecs = np.asarray(vecs)
    _close("eigh_values", sorted(vals), sorted(np.linalg.eigvalsh(_SPD)), atol=1e-7)
    # A == V diag(lambda) V^T (eigendecomposition invariant).
    _close("eigh_reconstruct", vecs @ np.diag(vals) @ vecs.T, _SPD, atol=1e-7)


def test_svd_matches_numpy() -> None:
    # Full SVD: leto returns (U, S, Vt) with A == U @ diag(S) @ Vt.
    u, s, vt = leto.svd(_GEN)
    u = np.asarray(u)
    s = np.asarray(s).flatten()
    vt = np.asarray(vt)
    _close("svd_reconstruct", u @ np.diag(s) @ vt, _GEN, atol=1e-8)
    _close("svd_u_orthonormal", u.T @ u, np.eye(u.shape[1]), atol=1e-8)
    _close("svd_vt_orthonormal", vt @ vt.T, np.eye(vt.shape[0]), atol=1e-8)
    exp_s = sorted(np.linalg.svd(_GEN, compute_uv=False), reverse=True)
    _close("svd_singular_values", sorted(s, reverse=True), exp_s, atol=1e-8)


def test_col_piv_qr_matches_numpy() -> None:
    # Column-pivoted QR: leto returns (Q, R, P) with A[:, P] == Q @ R.
    q, r, p = leto.col_piv_qr(_GEN)
    q = np.asarray(q)
    r = np.asarray(r)
    p = np.asarray(p).astype(int)
    _close("col_piv_qr_reconstruct", q @ r, _GEN[:, p], atol=1e-7)
    _close("col_piv_qr_orthonormal", q.T @ q, np.eye(q.shape[1]), atol=1e-8)
    _close("col_piv_qr_upper", np.tril(r, -1), np.zeros_like(r), atol=1e-7)


def test_schur_matches_scipy() -> None:
    # leto returns (Z, T) with A == Z @ T @ Z.T and Z orthogonal; the diagonal of
    # the (quasi-)triangular T holds the real eigenvalues for this symmetric input.
    z, t = leto.schur(_GEN)
    z = np.asarray(z)
    t = np.asarray(t)
    _close("schur_reconstruct", z @ t @ z.T, _GEN, atol=1e-6)
    _close("schur_z_orthonormal", z.T @ z, np.eye(3), atol=1e-8)
    _close(
        "schur_eigenvalues",
        sorted(np.diag(t)),
        sorted(np.linalg.eigvals(_GEN).real),
        atol=1e-6,
    )


def test_bunch_kaufman_matches_scipy() -> None:
    # Bunch-Kaufman LDL^T of a symmetric indefinite matrix: leto returns (L, D, P)
    # with L @ D @ L.T == A permuted by P (A[P][:, P]).
    lmat, d, p = leto.bunch_kaufman(_SYM_INDEF)
    lmat = np.asarray(lmat)
    d = np.asarray(d)
    p = np.asarray(p).astype(int)
    _close(
        "bunch_kaufman_reconstruct",
        lmat @ d @ lmat.T,
        _SYM_INDEF[np.ix_(p, p)],
        atol=1e-6,
    )


def test_matexp_matches_scipy() -> None:
    _close("matexp", leto.matexp(_GEN), scipy_linalg.expm(_GEN), atol=1e-6)
