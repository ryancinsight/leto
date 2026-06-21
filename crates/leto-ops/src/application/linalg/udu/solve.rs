//! Solve and inverse routines for an unpivoted `U D Uᵀ` factorization.

use super::decompose::Factored;
use crate::domain::real::RealScalar;
use leto::{Array1, Array2, ArrayView1, LetoError, Result};

fn solve_impl<T: RealScalar>(
    factor: &Factored<T>,
    buf: &mut [T],
    rhs: &ArrayView1<'_, T>,
) -> Result<()> {
    let n = factor.n;
    if let Some(rhs_slice) = rhs.as_slice() {
        buf[..n].copy_from_slice(&rhs_slice[..n]);
    } else {
        for (i, buf_i) in buf.iter_mut().enumerate().take(n) {
            *buf_i = *rhs.get([i])?;
        }
    }

    // A x = U D Uᵀ x. First solve U y = b.
    for i in (0..n).rev() {
        let mut acc = buf[i];
        for (j, &buf_j) in buf.iter().enumerate().take(n).skip(i + 1) {
            acc = acc.sub(factor.u[i * n + j].mul(buf_j));
        }
        buf[i] = acc;
    }

    // Then z = D⁻¹ y.
    for (i, buf_i) in buf.iter_mut().enumerate().take(n) {
        *buf_i = buf_i.div(factor.d[i]);
    }

    // Finally solve Uᵀ x = z.
    for i in 0..n {
        let mut acc = buf[i];
        for (j, &buf_j) in buf[..i].iter().enumerate() {
            acc = acc.sub(factor.u[j * n + i].mul(buf_j));
        }
        buf[i] = acc;
    }

    Ok(())
}

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

    let mut x = vec![T::ZERO; n];
    solve_impl(factor, &mut x, rhs)?;
    Array1::from_shape_vec([n], x)
}

pub(super) fn inverse<T: RealScalar>(factor: &Factored<T>) -> Result<Array2<T>> {
    let n = factor.n;
    let mut values = vec![T::ZERO; n * n];

    let mut col_buf_stack = [T::ZERO; 128];
    let mut col_buf_vec = Vec::new();
    let col_buf = if n <= 128 {
        &mut col_buf_stack[..n]
    } else {
        col_buf_vec.resize(n, T::ZERO);
        &mut col_buf_vec[..]
    };

    for col in 0..n {
        // Initialize col_buf for the current standard basis vector e_col
        for (i, col_buf_i) in col_buf.iter_mut().enumerate().take(n) {
            *col_buf_i = if i == col { T::ONE } else { T::ZERO };
        }

        // Solve U y = e_col
        for i in (0..n).rev() {
            let mut acc = col_buf[i];
            for (j, &col_buf_j) in col_buf.iter().enumerate().take(n).skip(i + 1) {
                acc = acc.sub(factor.u[i * n + j].mul(col_buf_j));
            }
            col_buf[i] = acc;
        }

        // z = D^-1 y
        for (i, col_buf_i) in col_buf.iter_mut().enumerate().take(n) {
            *col_buf_i = col_buf_i.div(factor.d[i]);
        }

        // U^T x = z
        for i in 0..n {
            let mut acc = col_buf[i];
            for (j, &col_buf_j) in col_buf[..i].iter().enumerate() {
                acc = acc.sub(factor.u[j * n + i].mul(col_buf_j));
            }
            col_buf[i] = acc;
        }

        for row in 0..n {
            values[row * n + col] = col_buf[row];
        }
    }
    Array2::from_shape_vec([n, n], values)
}
