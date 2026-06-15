//! Solve and inverse routines for an unpivoted `U D Uᵀ` factorization.

use super::decompose::Factored;
use crate::domain::real::RealScalar;
use leto::{Array1, Array2, ArrayView1, LetoError, Result};

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

    // A x = U D Uᵀ x. First solve U y = b.
    let mut y = vec![T::ZERO; n];
    for i in (0..n).rev() {
        let mut acc = *rhs.get([i])?;
        for (j, &yj) in y.iter().enumerate().skip(i + 1) {
            acc = acc.sub(factor.u[i * n + j].mul(yj));
        }
        y[i] = acc;
    }

    // Then z = D⁻¹ y.
    let mut z = vec![T::ZERO; n];
    for (i, slot) in z.iter_mut().enumerate() {
        *slot = y[i].div(factor.d[i]);
    }

    // Finally solve Uᵀ x = z.
    let mut x = vec![T::ZERO; n];
    for i in 0..n {
        let mut acc = z[i];
        for (j, &xj) in x.iter().enumerate().take(i) {
            acc = acc.sub(factor.u[j * n + i].mul(xj));
        }
        x[i] = acc;
    }

    Array1::from_shape_vec([n], x)
}

pub(super) fn inverse<T: RealScalar>(factor: &Factored<T>) -> Result<Array2<T>> {
    let n = factor.n;
    let mut values = vec![T::ZERO; n * n];
    for col in 0..n {
        let mut rhs = vec![T::ZERO; n];
        rhs[col] = T::ONE;
        let rhs = Array1::from_shape_vec([n], rhs)?;
        let x = solve(factor, &rhs.view())?;
        for row in 0..n {
            values[row * n + col] = *x.get([row])?;
        }
    }
    Array2::from_shape_vec([n, n], values)
}
