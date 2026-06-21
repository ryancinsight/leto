//! Dense square-matrix helpers shared by the matrix-function kernels (SSOT).
//!
//! All helpers consume and return C-contiguous [`Array2`] built by
//! [`Array2::from_shape_vec`]; the public entry points materialize a possibly
//! strided input once (`to_contiguous`) before calling in, so the storage slice
//! is always row-major here.

use crate::application::matrix::matmul;
use crate::domain::real::RealScalar;
use crate::domain::scalar::Scalar;
use leto::{Array2, ArrayView2, Result};

/// The `n × n` identity matrix.
pub(super) fn identity<T: Scalar>(n: usize) -> Array2<T> {
    let mut values = vec![T::ZERO; n * n];
    for i in 0..n {
        values[i * n + i] = T::ONE;
    }
    Array2::from_shape_vec([n, n], values).expect("identity square storage matches shape")
}

/// Matrix product `a · b`, delegating to the caller-owned [`matmul`] kernel
/// (SSOT — no second contraction path).
pub(super) fn mul<T: Scalar>(a: &ArrayView2<'_, T>, b: &ArrayView2<'_, T>) -> Result<Array2<T>> {
    let [rows, _] = a.shape();
    let [_, cols] = b.shape();
    let mut out = Array2::from_shape_vec([rows, cols], vec![T::ZERO; rows * cols])
        .expect("product storage matches shape");
    {
        let mut out_view = out.view_mut();
        matmul(a, b, &mut out_view)?;
    }
    Ok(out)
}

/// Elementwise scalar multiple `c · a`.
pub(super) fn scale<T: Scalar>(a: &ArrayView2<'_, T>, c: T) -> Array2<T> {
    let [rows, cols] = a.shape();
    let values: Vec<T> = if let Some(slice) = a.as_slice() {
        slice.iter().map(|&x| x.mul(c)).collect()
    } else {
        a.iter().map(|&x| x.mul(c)).collect()
    };
    Array2::from_shape_vec([rows, cols], values).expect("scaled storage matches shape")
}

/// Elementwise sum `a + b` (same shape by construction).
pub(super) fn add<T: Scalar>(a: &ArrayView2<'_, T>, b: &ArrayView2<'_, T>) -> Array2<T> {
    let [rows, cols] = a.shape();
    let values: Vec<T> = if let (Some(sa), Some(sb)) = (a.as_slice(), b.as_slice()) {
        sa.iter().zip(sb).map(|(&x, &y)| x.add(y)).collect()
    } else {
        a.iter().zip(b.iter()).map(|(&x, &y)| x.add(y)).collect()
    };
    Array2::from_shape_vec([rows, cols], values).expect("sum storage matches shape")
}

/// Elementwise difference `a − b` (same shape by construction).
pub(super) fn sub<T: Scalar>(a: &ArrayView2<'_, T>, b: &ArrayView2<'_, T>) -> Array2<T> {
    let [rows, cols] = a.shape();
    let values: Vec<T> = if let (Some(sa), Some(sb)) = (a.as_slice(), b.as_slice()) {
        sa.iter().zip(sb).map(|(&x, &y)| x.sub(y)).collect()
    } else {
        a.iter().zip(b.iter()).map(|(&x, &y)| x.sub(y)).collect()
    };
    Array2::from_shape_vec([rows, cols], values).expect("difference storage matches shape")
}

/// Induced ∞-norm `‖A‖_∞ = maxᵢ Σⱼ |aᵢⱼ|` (the maximum absolute row sum).
pub(super) fn inf_norm<T: RealScalar>(a: &ArrayView2<'_, T>) -> T {
    let [rows, cols] = a.shape();
    let mut max = T::ZERO;
    if let Some(slice) = a.as_slice() {
        for i in 0..rows {
            let mut row_sum = T::ZERO;
            for j in 0..cols {
                row_sum = row_sum.add(slice[i * cols + j].abs());
            }
            if row_sum > max {
                max = row_sum;
            }
        }
    } else {
        for i in 0..rows {
            let mut row_sum = T::ZERO;
            for j in 0..cols {
                row_sum = row_sum.add(a.get([i, j]).expect("in bounds").abs());
            }
            if row_sum > max {
                max = row_sum;
            }
        }
    }
    max
}
