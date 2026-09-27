//! [`sparse_lu_solve`] one-call convenience wrapper and the
//! [`csr_to_dense`] CSR-to-dense bridge the dense fallback path uses.

use super::types::SparseLuSolver;
use crate::application::sparse::csr::CsrMatrix;
use crate::domain::real::RealScalar;
use leto::{Array2, Result};

/// Convenience: solve `A · x = b` in one call without constructing a solver.
///
/// Uses `DENSE_LIMIT_DEFAULT` as the maximum system order and the
/// measured dispatch thresholds `SMALL_SWITCH_DEFAULT` and
/// `DENSITY_THRESHOLD_DEFAULT`.
///
/// # Errors
/// Forwards all errors from [`SparseLuSolver::solve`].
pub fn sparse_lu_solve<T: RealScalar>(matrix: &CsrMatrix<T>, rhs: &[T]) -> Result<Vec<T>> {
    SparseLuSolver::default().solve(matrix, rhs)
}

/// Expand a `CsrMatrix<T>` to a dense row-major `Array2<T>`.
///
/// This is the bridge between the sparse atlas storage format and the dense LU path.
/// Cost: `O(n·m + nnz)` — one pass over the zero-filled buffer and one pass over
/// the nonzeros.
#[must_use]
pub fn csr_to_dense<T: RealScalar>(matrix: &CsrMatrix<T>) -> Array2<T> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    let mut data = vec![T::ZERO; nrows * ncols];
    for row in 0..nrows {
        for (col, &value) in matrix
            .row(row)
            .col_indices()
            .iter()
            .zip(matrix.row(row).values())
        {
            data[row * ncols + col] = value;
        }
    }
    Array2::from_shape_vec([nrows, ncols], data).expect("shape matches nrows * ncols")
}
