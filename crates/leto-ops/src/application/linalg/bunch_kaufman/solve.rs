//! Solve and inverse for a Bunch–Kaufman `P A Pᵀ = L D Lᵀ` factorization.

use super::decompose::Factored;
use crate::domain::real::RealScalar;
use leto::{Array1, Array2, ArrayView1, LetoError, Result};

#[inline]
fn idx(i: usize, j: usize, n: usize) -> usize {
    i * n + j
}

/// Solve `A x = rhs` via `P A Pᵀ = L D Lᵀ`:
/// permute `rhs`, forward-solve `L z = Pb`, block-solve `D w = z`, back-solve
/// `Lᵀ y = w`, then inverse-permute `x[perm[i]] = y[i]`.
pub(super) fn solve<T: RealScalar>(
    factor: &Factored<T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>> {
    let n = factor.n;
    if rhs.shape() != [n] {
        return Err(LetoError::ShapeMismatch {
            lhs: rhs.shape().to_vec(),
            rhs: vec![n],
        });
    }

    // Pb: b_perm[i] = rhs[perm[i]].
    let mut b = vec![T::ZERO; n];
    for (slot, &p) in b.iter_mut().zip(factor.perm.iter()) {
        *slot = *rhs.get([p])?;
    }

    // Forward: L z = b (unit lower).
    let mut z = vec![T::ZERO; n];
    for i in 0..n {
        let mut acc = b[i];
        for (j, &zj) in z.iter().enumerate().take(i) {
            acc = acc.sub(factor.l[idx(i, j, n)].mul(zj));
        }
        z[i] = acc;
    }

    // Block-diagonal solve D w = z.
    let mut w = vec![T::ZERO; n];
    let mut k = 0usize;
    while k < n {
        if factor.two[k] {
            let d00 = factor.d[idx(k, k, n)];
            let d01 = factor.d[idx(k, k + 1, n)];
            let d11 = factor.d[idx(k + 1, k + 1, n)];
            let det = d00.mul(d11).sub(d01.mul(d01));
            if det == T::ZERO {
                return Err(LetoError::StorageError {
                    reason: "Bunch-Kaufman solve: singular 2x2 block".to_string(),
                });
            }
            // [w_k; w_{k+1}] = E⁻¹ [z_k; z_{k+1}].
            w[k] = d11.mul(z[k]).sub(d01.mul(z[k + 1])).div(det);
            w[k + 1] = d00.mul(z[k + 1]).sub(d01.mul(z[k])).div(det);
            k += 2;
        } else {
            let d = factor.d[idx(k, k, n)];
            if d == T::ZERO {
                return Err(LetoError::StorageError {
                    reason: "Bunch-Kaufman solve: singular pivot".to_string(),
                });
            }
            w[k] = z[k].div(d);
            k += 1;
        }
    }

    // Back: Lᵀ y = w (unit upper).
    let mut y = vec![T::ZERO; n];
    for i in (0..n).rev() {
        let mut acc = w[i];
        for (j, &yj) in y.iter().enumerate().skip(i + 1) {
            acc = acc.sub(factor.l[idx(j, i, n)].mul(yj));
        }
        y[i] = acc;
    }

    // Inverse permutation: x[perm[i]] = y[i].
    let mut x = vec![T::ZERO; n];
    for i in 0..n {
        x[factor.perm[i]] = y[i];
    }
    Array1::from_shape_vec([n], x)
}

/// Inverse `A⁻¹` by solving against each canonical basis vector.
pub(super) fn inverse<T: RealScalar>(factor: &Factored<T>) -> Result<Array2<T>> {
    let n = factor.n;
    let mut values = vec![T::ZERO; n * n];
    for col in 0..n {
        let mut rhs = vec![T::ZERO; n];
        rhs[col] = T::ONE;
        let rhs = Array1::from_shape_vec([n], rhs)?;
        let x = solve(factor, &rhs.view())?;
        for row in 0..n {
            values[idx(row, col, n)] = *x.get([row])?;
        }
    }
    Array2::from_shape_vec([n, n], values)
}

/// Determinant `det(A) = det(D) = ∏ blocks` (`det(L) = 1`, `det(P)² = 1`).
pub(super) fn determinant<T: RealScalar>(factor: &Factored<T>) -> T {
    let n = factor.n;
    let mut det = T::ONE;
    let mut k = 0usize;
    while k < n {
        if factor.two[k] {
            let d00 = factor.d[idx(k, k, n)];
            let d01 = factor.d[idx(k, k + 1, n)];
            let d11 = factor.d[idx(k + 1, k + 1, n)];
            det = det.mul(d00.mul(d11).sub(d01.mul(d01)));
            k += 2;
        } else {
            det = det.mul(factor.d[idx(k, k, n)]);
            k += 1;
        }
    }
    det
}
