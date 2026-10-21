//! Row-major transpose helper for the compact-WY block forms.

use crate::domain::real::RealScalar;

/// Transpose a row-major `rows × cols` matrix into a mutable slice `out` (`cols × rows`).
pub(in crate::application::linalg) fn transpose_into<T: RealScalar>(
    src: &[T],
    out: &mut [T],
    rows: usize,
    cols: usize,
) {
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
}
