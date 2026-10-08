//! Solve a square CSR system by bridging through dense LU.
//!
//! For small systems the dense path wins on constant factors: the sparse
//! machinery spends more on dispatch and indirect indexing than the
//! `O(n³)` dense factorisation costs once `n` is small enough for both to
//! sit in cache. This bridge is how a caller reaches that path from CSR
//! storage without assembling a dense matrix by hand.
//!
//! The expansion writes each stored nonzero into its dense position and
//! leaves every unstored position at zero — the unstored entries *are*
//! the zeros of the dense form, so the buffer is written once by
//! construction and no separate zeroing pass exists.
//!
//! # Memory
//!
//! The bridge allocates the `n × n` dense buffer — `O(n²)` storage — and
//! the dense LU factorisation on top of it. That is the price of the
//! small-`n` regime it exists for; a caller holding a large sparse system
//! should stay on the sparse kernels
//! ([`SparseLuSolver`](super::SparseLuSolver)) instead.

use crate::domain::real::RealScalar;
use leto::{Array1, Array2, LetoError, Result};

use super::CsrMatrix;
use crate::application::linalg::solve;

/// Solve `A·x = b` for a square CSR matrix by dense LU.
///
/// The matrix is expanded to dense row-major storage and factorised by the
/// SSOT partial-pivoting LU
/// ([`solve`](crate::application::linalg::solve)).
///
/// # Errors
///
/// Returns [`LetoError::InvalidInput`] for a non-square matrix or a
/// right-hand side whose length differs from the matrix order, and the
/// dense LU failure (zero pivot) unchanged otherwise.
pub fn solve_csr_via_dense_lu<T: RealScalar>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
) -> Result<Array1<T>> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    if nrows != ncols {
        return Err(LetoError::InvalidInput(format!(
            "dense LU bridge requires a square matrix, got {nrows}x{ncols}"
        )));
    }
    if rhs.shape()[0] != nrows {
        return Err(LetoError::InvalidInput(format!(
            "dense LU bridge requires a right-hand side of length {nrows}, got {}",
            rhs.shape()[0]
        )));
    }

    let mut dense_values = vec![T::ZERO; nrows * ncols];
    let row_offsets = matrix.row_ptr();
    let col_indices = matrix.col_indices();
    let values = matrix.values();
    for row in 0..nrows {
        for index in row_offsets[row]..row_offsets[row + 1] {
            dense_values[row * ncols + col_indices[index]] = values[index];
        }
    }

    let dense = Array2::from_shape_vec([nrows, ncols], dense_values)
        .expect("invariant: row-major shape matches the buffer just filled");
    solve(&dense.view(), &rhs.view())
}

#[cfg(test)]
mod tests {
    use super::solve_csr_via_dense_lu;
    use super::CsrMatrix;
    use eunomia::{FloatElement, RealField};
    use leto::{Array1, LetoError};

    /// The recorded two-by-two fixture CFDrs pinned this bridge on:
    /// `[4 1; 2 3]·x = [1; 7]` solves to `x = [-0.4; 2.6]`.
    ///
    /// Dense LU solves the system exactly in real arithmetic; the observed
    /// error is the factorisation's rounding, bounded by `κ·ε_T·‖x‖₂` with
    /// `κ = 100` covering the O(n³) elimination's error accumulation for
    /// the fixture's condition number (~10) at both shipped precisions.
    macro_rules! recorded_fixture {
        ($t:ty) => {{
            let matrix = CsrMatrix::from_parts(
                vec![
                    <$t as FloatElement>::from_f64(4.0),
                    <$t as FloatElement>::from_f64(1.0),
                    <$t as FloatElement>::from_f64(2.0),
                    <$t as FloatElement>::from_f64(3.0),
                ],
                vec![0, 1, 0, 1],
                vec![0, 2, 4],
                2,
                2,
            )
            .expect("invariant: fixture CSR structure is valid");
            let rhs = Array1::from_shape_vec(
                [2],
                vec![
                    <$t as FloatElement>::from_f64(1.0),
                    <$t as FloatElement>::from_f64(7.0),
                ],
            )
            .expect("invariant: fixture RHS shape is valid");
            let solution = solve_csr_via_dense_lu(&matrix, &rhs)
                .expect("invariant: the fixture system is nonsingular");
            let norm = <$t as FloatElement>::from_f64((0.4_f64 * 0.4 + 2.6 * 2.6).sqrt());
            let bound = <$t as FloatElement>::from_f64(100.0) * <$t as RealField>::EPSILON * norm;
            assert!(
                (<$t as FloatElement>::from_f64(-0.4) - solution[0]).abs() <= bound,
                "x[0] {:?} deviates beyond {bound:?}",
                solution[0]
            );
            assert!(
                (<$t as FloatElement>::from_f64(2.6) - solution[1]).abs() <= bound,
                "x[1] {:?} deviates beyond {bound:?}",
                solution[1]
            );
        }};
    }

    #[test]
    fn solves_the_recorded_fixture_at_every_shipped_scalar() {
        recorded_fixture!(f32);
        recorded_fixture!(f64);
    }

    #[test]
    fn rejects_a_non_square_matrix() {
        let matrix = CsrMatrix::<f64>::from_parts(vec![1.0], vec![0], vec![0, 1], 1, 2)
            .expect("invariant: 1x2 CSR structure is valid");
        let rhs = Array1::from_elem([1], 1.0_f64);
        match solve_csr_via_dense_lu(&matrix, &rhs) {
            Err(LetoError::InvalidInput(reason)) => assert!(
                reason.contains("square"),
                "the rejection must name the square-matrix requirement, got: {reason}"
            ),
            Err(other) => panic!("a 1x2 system must reject as invalid input, got: {other}"),
            Ok(solution) => panic!("a 1x2 system must be rejected, solved {solution:?}"),
        }
    }

    #[test]
    fn rejects_a_mismatched_right_hand_side() {
        let matrix = CsrMatrix::<f64>::from_parts(vec![2.0], vec![0], vec![0, 1], 1, 1)
            .expect("invariant: 1x1 CSR structure is valid");
        let rhs = Array1::from_elem([2], 1.0_f64);
        match solve_csr_via_dense_lu(&matrix, &rhs) {
            Err(LetoError::InvalidInput(reason)) => assert!(
                reason.contains("length"),
                "the rejection must name the length requirement, got: {reason}"
            ),
            Err(other) => panic!("a mismatched RHS must reject as invalid input, got: {other}"),
            Ok(solution) => panic!("a mismatched RHS must be rejected, solved {solution:?}"),
        }
    }
}
