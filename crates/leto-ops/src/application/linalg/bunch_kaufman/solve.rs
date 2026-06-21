//! Solve and inverse for a Bunch–Kaufman `P A Pᵀ = L D Lᵀ` factorization.

use super::decompose::Factored;
use crate::domain::real::RealScalar;
use leto::{Array1, Array2, ArrayView1, LetoError, Result};

#[inline]
fn idx(i: usize, j: usize, n: usize) -> usize {
    i * n + j
}

fn solve_impl<T: RealScalar>(
    factor: &Factored<T>,
    buf: &mut [T],
    x: &mut [T],
) -> Result<()> {
    let n = factor.n;

    // Step 1: Forward solve L z = b in-place in buf
    for i in 0..n {
        let mut acc = buf[i];
        for (j, &buf_j) in buf[..i].iter().enumerate() {
            acc = acc.sub(factor.l[idx(i, j, n)].mul(buf_j));
        }
        buf[i] = acc;
    }

    // Step 2: Block-diagonal solve D w = z in-place in buf
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
            let zk = buf[k];
            let zk1 = buf[k + 1];
            buf[k] = d11.mul(zk).sub(d01.mul(zk1)).div(det);
            buf[k + 1] = d00.mul(zk1).sub(d01.mul(zk)).div(det);
            k += 2;
        } else {
            let d = factor.d[idx(k, k, n)];
            if d == T::ZERO {
                return Err(LetoError::StorageError {
                    reason: "Bunch-Kaufman solve: singular pivot".to_string(),
                });
            }
            buf[k] = buf[k].div(d);
            k += 1;
        }
    }

    // Step 3: Back-solve Lᵀ y = w in-place in buf
    for i in (0..n).rev() {
        let mut acc = buf[i];
        for (j, &buf_j) in buf.iter().enumerate().take(n).skip(i + 1) {
            acc = acc.sub(factor.l[idx(j, i, n)].mul(buf_j));
        }
        buf[i] = acc;
    }

    // Step 4: Inverse permutation: x[perm[i]] = buf[i]
    for i in 0..n {
        x[factor.perm[i]] = buf[i];
    }

    Ok(())
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

    let mut buf_stack = [T::ZERO; 128];
    let mut buf_vec = Vec::new();
    let buf = if n <= 128 {
        &mut buf_stack[..n]
    } else {
        buf_vec.resize(n, T::ZERO);
        &mut buf_vec[..]
    };

    if let Some(rhs_slice) = rhs.as_slice() {
        for (i, buf_i) in buf.iter_mut().enumerate().take(n) {
            *buf_i = rhs_slice[factor.perm[i]];
        }
    } else {
        for (i, buf_i) in buf.iter_mut().enumerate().take(n) {
            *buf_i = *rhs.get([factor.perm[i]])?;
        }
    }

    let mut x = vec![T::ZERO; n];
    solve_impl(factor, buf, &mut x)?;
    Array1::from_shape_vec([n], x)
}

/// Inverse `A⁻¹` by solving against each canonical basis vector.
pub(super) fn inverse<T: RealScalar>(factor: &Factored<T>) -> Result<Array2<T>> {
    let n = factor.n;
    let mut values = vec![T::ZERO; n * n];

    let mut buf_stack = [T::ZERO; 128];
    let mut buf_vec = Vec::new();
    let buf = if n <= 128 {
        &mut buf_stack[..n]
    } else {
        buf_vec.resize(n, T::ZERO);
        &mut buf_vec[..]
    };

    let mut x_stack = [T::ZERO; 128];
    let mut x_vec = Vec::new();
    let x = if n <= 128 {
        &mut x_stack[..n]
    } else {
        x_vec.resize(n, T::ZERO);
        &mut x_vec[..]
    };

    for col in 0..n {
        // e_col: rhs[i] = 1 if i == col else 0.
        // buf[i] = rhs[perm[i]] = 1 if perm[i] == col else 0.
        for (i, buf_i) in buf.iter_mut().enumerate().take(n) {
            *buf_i = if factor.perm[i] == col { T::ONE } else { T::ZERO };
        }
        solve_impl(factor, buf, x)?;
        for row in 0..n {
            values[idx(row, col, n)] = x[row];
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
