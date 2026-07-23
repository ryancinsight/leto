//! Internal vector-arithmetic helpers for iterative solvers.
//!
//! These are zero-allocation scalar/vector operations on `Array1<T>` used as
//! building-blocks by CG, BiCGSTAB, GMRES and LSQR. All functions are `#[inline]`.

use eunomia::{NumericElement, RealField};
use leto::Array1;

/// Return the length (shape\[0\]) of a 1-D array.
#[inline]
pub(super) fn vector_len<T>(v: &Array1<T>) -> usize {
    v.shape()[0]
}

/// Dot product ⟨lhs, rhs⟩ for real arrays.
#[inline]
pub(super) fn dot<T: RealField + Copy + NumericElement>(lhs: &Array1<T>, rhs: &Array1<T>) -> T {
    let mut s = <T as NumericElement>::ZERO;
    for i in 0..vector_len(lhs) {
        s += lhs[i] * rhs[i];
    }
    s
}

/// Euclidean norm ‖v‖₂.
#[inline]
pub(super) fn norm<T: RealField + Copy + NumericElement>(v: &Array1<T>) -> T {
    NumericElement::sqrt(dot(v, v))
}

/// Copy `src` into `dst` element-wise.
#[inline]
pub(super) fn copy_vec<T: Copy>(src: &Array1<T>, dst: &mut Array1<T>) {
    for i in 0..vector_len(src) {
        dst[i] = src[i];
    }
}

/// `r ← b − Ax`  (assign residual).
#[inline]
pub(super) fn assign_residual<T: RealField + Copy + NumericElement>(
    r: &mut Array1<T>,
    b: &Array1<T>,
    ax: &Array1<T>,
) {
    for i in 0..vector_len(b) {
        r[i] = b[i] - ax[i];
    }
}

/// AXPY: `x ← x + α·y`.
#[inline]
pub(super) fn axpy<T: RealField + Copy>(x: &mut Array1<T>, alpha: T, y: &Array1<T>) {
    for i in 0..vector_len(x) {
        x[i] += alpha * y[i];
    }
}

/// Scale-add: `x ← scale·x + y`.
#[inline]
pub(super) fn scale_add<T: RealField + Copy>(x: &mut Array1<T>, scale: T, y: &Array1<T>) {
    for i in 0..vector_len(x) {
        x[i] = x[i] * scale + y[i];
    }
}

/// Validate that `v` has length `expected`; return `Err(InvalidInput)` otherwise.
#[inline]
pub(super) fn validate_len<T>(name: &str, v: &Array1<T>, expected: usize) -> leto::Result<()> {
    let n = vector_len(v);
    if n != expected {
        return Err(leto::LetoError::InvalidInput(format!(
            "{name} length mismatch: expected {expected}, got {n}"
        )));
    }
    Ok(())
}

/// Test whether `x` is finite (not NaN and not ±∞).
#[allow(dead_code)]
#[inline]
pub(super) fn is_finite<T: NumericElement>(x: T) -> bool {
    NumericElement::to_f64(x).is_finite()
}
