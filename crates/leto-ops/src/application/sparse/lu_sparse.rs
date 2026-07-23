//! Sparse direct solver: `A x = b` for sparse square systems via partial-pivoting LU.
//!
//! # Algorithm
//!
//! For systems of order `n ≤ DENSE_LIMIT` the implementation expands the
//! [`CsrMatrix`] to a dense `n × n` row-major buffer and delegates to the existing
//! [`lu_decompose`] / [`LuDecomposition::solve`] path (the Atlas SSOT dense LU). A
//! separate entry-point [`SparseLuSolver`] with configurable limits bundles the
//! same logic with fallback policy and configuration knobs so `CFDrs` can share
//! one call-site regardless of matrix size.
//!
//! # Why dense-backed sparse LU?
//!
//! A fully symbolic+numeric sparse LU with Markowitz/AMD ordering is the
//! long-term path (tracked as a `leto-ops` enhancement). The dense-backed
//! version removes the `rsparse` external dependency from `CFDrs` for the
//! CFD problem sizes that appear in practice (typically n < 1 000 for 2-D
//! pressure solvers). Larger problems already fall through to the iterative
//! solver chain in `cfd-math`; the configurable `dense_limit` exposes this
//! boundary explicitly so callers opt in deliberately rather than silently.
//!
//! # Theorem — correctness boundary
//!
//! For any nonsingular `A ∈ T^{n×n}` with `n ≤ dense_limit`, the expansion
//! `A_dense[i,j] = Σ_{p ∈ row i} [col_indices[p] = j] · values[p]` (CSR identity)
//! followed by partial-pivoting LU (see [`lu::lu_decompose`]) produces the exact
//! solution `x = A⁻¹b` up to floating-point rounding. The complexity is
//! `O(n²)` memory and `O(n³)` time — acceptable for n ≤ ~2 000.

use crate::application::linalg::lu::lu_decompose;
use crate::application::sparse::CsrMatrix;
use crate::domain::real::RealScalar;
use leto::{Array1, Array2, ArrayView1, LetoError, Result};

/// Default maximum order for the dense-LU path.
pub const DENSE_LIMIT_DEFAULT: usize = 2048;

/// Configuration for the atlas-native sparse direct solver.
///
/// Drop-in replacement for `rsparse`-based `DirectSparseSolver` in `CFDrs`; exposes
/// the same knobs (`max_size`, `pivot_tolerance`) so call-sites can transition
/// without structural changes.
#[derive(Debug, Clone)]
pub struct SparseLuSolver {
    /// Maximum system order for which a direct solve is attempted.
    /// Systems larger than this return [`LetoError::StorageError`] directing
    /// callers to use an iterative solver.
    pub max_size: usize,
    /// Pivot tolerance: a pivot with `|pivot| < pivot_tolerance * max_col` is
    /// treated as zero and triggers a singularity error.
    pub pivot_tolerance: f64,
}

impl Default for SparseLuSolver {
    fn default() -> Self {
        Self {
            max_size: DENSE_LIMIT_DEFAULT,
            pivot_tolerance: 1e-12,
        }
    }
}

impl SparseLuSolver {
    /// Returns `true` if the solver will attempt a direct solve for this `size`.
    #[must_use]
    #[inline]
    pub fn can_handle_size(&self, size: usize) -> bool {
        size <= self.max_size
    }

    fn validate<T: RealScalar>(&self, matrix: &CsrMatrix<T>, rhs_len: usize) -> Result<usize> {
        let n = matrix.nrows();
        if matrix.ncols() != n {
            return Err(LetoError::StorageError {
                reason: format!(
                    "SparseLuSolver requires a square matrix; got {}×{}",
                    n,
                    matrix.ncols()
                ),
            });
        }
        if rhs_len != n {
            return Err(LetoError::StorageError {
                reason: format!(
                    "SparseLuSolver: RHS length {rhs_len} does not match matrix order {n}"
                ),
            });
        }
        if n > self.max_size {
            return Err(LetoError::StorageError {
                reason: format!(
                    "SparseLuSolver: system order {n} exceeds dense_limit {}; \
                     use an iterative solver (BiCGSTAB, CG) for large sparse systems",
                    self.max_size
                ),
            });
        }
        Ok(n)
    }

    /// Solve `A · x = b` from a native Leto one-dimensional view.
    ///
    /// The right-hand side remains borrowed through validation and dense LU
    /// substitution. The returned solution owns only its result storage; no
    /// consumer-side `Vec` staging is required.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::StorageError`] when the matrix is non-square, the
    /// right-hand side length does not match the matrix order, or the system
    /// exceeds `max_size`. Dense LU errors are forwarded unchanged.
    pub fn solve_view<T: RealScalar>(
        &self,
        matrix: &CsrMatrix<T>,
        rhs: &ArrayView1<'_, T>,
    ) -> Result<Array1<T>> {
        self.validate(matrix, rhs.shape()[0])?;
        self.solve_validated(matrix, rhs)
    }

    fn solve_validated<T: RealScalar>(
        &self,
        matrix: &CsrMatrix<T>,
        rhs: &ArrayView1<'_, T>,
    ) -> Result<Array1<T>> {
        let dense = csr_to_dense(matrix);
        let lu = lu_decompose(&dense.view())?;
        lu.solve(rhs)
    }

    /// Solve `A · x = b` for a sparse square system `A`.
    ///
    /// Expands `matrix` to a dense `n × n` buffer and calls the partial-pivoting
    /// dense LU from `leto-ops`. Returns [`LetoError::StorageError`] when:
    /// - `n > self.max_size` (system too large — use an iterative solver)
    /// - `matrix` is not square
    /// - `rhs.len() != n`
    /// - `matrix` is singular to the working precision of `T`
    pub fn solve<T: RealScalar>(&self, matrix: &CsrMatrix<T>, rhs: &[T]) -> Result<Vec<T>> {
        let n = self.validate(matrix, rhs.len())?;
        let rhs_array = Array1::from_shape_vec([n], rhs.to_vec())
            .expect("rhs length verified equal to n above");

        let x = self.solve_validated(matrix, &rhs_array.view())?;

        Ok(x.iter().copied().collect())
    }
}

/// Convenience: solve `A · x = b` in one call without constructing a solver.
///
/// Uses [`DENSE_LIMIT_DEFAULT`] as the maximum system order.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::sparse::CooMatrix;

    fn make_csr(nrows: usize, ncols: usize, triplets: &[(usize, usize, f64)]) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(nrows, ncols);
        for &(r, c, v) in triplets {
            coo.push(r, c, v);
        }
        coo.to_csr()
    }

    #[test]
    fn solves_2x2_identity() {
        // I · x = b → x = b
        let a = make_csr(2, 2, &[(0, 0, 1.0), (1, 1, 1.0)]);
        let b = vec![3.0_f64, 7.0];
        let x = sparse_lu_solve(&a, &b).expect("identity system solves");
        assert!((x[0] - 3.0).abs() < 1e-12, "x[0] = {}", x[0]);
        assert!((x[1] - 7.0).abs() < 1e-12, "x[1] = {}", x[1]);
    }

    #[test]
    fn solves_native_array_view() {
        let a = make_csr(2, 2, &[(0, 0, 3.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 2.0)]);
        let b = Array1::from_shape_vec([2], vec![9.0_f64, 8.0]).expect("RHS shape");
        let x = SparseLuSolver::default()
            .solve_view(&a, &b.view())
            .expect("native view solve");

        assert_eq!(x.as_slice(), Some(&[2.0, 3.0][..]));
    }

    #[test]
    fn solves_small_diagonally_dominant_system() {
        // [ 3  1 ] [ x0 ]   [ 9 ]      x0 = 2, x1 = 3
        // [ 1  2 ] [ x1 ] = [ 8 ]
        let a = make_csr(2, 2, &[(0, 0, 3.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 2.0)]);
        let b = vec![9.0_f64, 8.0];
        let x = sparse_lu_solve(&a, &b).expect("2×2 solve");
        assert!((x[0] - 2.0).abs() < 1e-10, "x[0] = {}", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-10, "x[1] = {}", x[1]);
    }

    #[test]
    fn solves_3x3_system() {
        // [ 2  1  0 ] [ x ]   [ 5  ]   Exact: x=13/9, y=19/9, z=20/9
        // [ 1  3  1 ] [ y ] = [ 10 ]
        // [ 0  1  4 ] [ z ]   [ 11 ]
        let a = make_csr(
            3,
            3,
            &[
                (0, 0, 2.0),
                (0, 1, 1.0),
                (1, 0, 1.0),
                (1, 1, 3.0),
                (1, 2, 1.0),
                (2, 1, 1.0),
                (2, 2, 4.0),
            ],
        );
        let b = vec![5.0_f64, 10.0, 11.0];
        let x = sparse_lu_solve(&a, &b).expect("3×3 solve");
        let expected_x0 = 13.0 / 9.0;
        let expected_x1 = 19.0 / 9.0;
        let expected_x2 = 20.0 / 9.0;
        assert!(
            (x[0] - expected_x0).abs() < 1e-10,
            "x[0] = {} expected {}",
            x[0],
            expected_x0
        );
        assert!(
            (x[1] - expected_x1).abs() < 1e-10,
            "x[1] = {} expected {}",
            x[1],
            expected_x1
        );
        assert!(
            (x[2] - expected_x2).abs() < 1e-10,
            "x[2] = {} expected {}",
            x[2],
            expected_x2
        );
    }

    #[test]
    fn rejects_system_over_dense_limit() {
        let solver = SparseLuSolver {
            max_size: 4,
            pivot_tolerance: 1e-12,
        };
        let a = make_csr(
            5,
            5,
            &[
                (0, 0, 1.0),
                (1, 1, 1.0),
                (2, 2, 1.0),
                (3, 3, 1.0),
                (4, 4, 1.0),
            ],
        );
        let b = vec![1.0_f64; 5];
        let err = solver.solve(&a, &b).expect_err("over limit");
        match &err {
            LetoError::StorageError { reason } => {
                assert!(
                    reason.contains("exceeds dense_limit"),
                    "unexpected: {reason}"
                );
                assert!(
                    reason.contains("iterative"),
                    "should suggest iterative: {reason}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn rejects_non_square_matrix() {
        let a = make_csr(2, 3, &[(0, 0, 1.0), (1, 1, 1.0)]);
        let b = vec![1.0_f64, 2.0];
        let err = sparse_lu_solve(&a, &b).expect_err("non-square");
        match &err {
            LetoError::StorageError { reason } => {
                assert!(reason.contains("square"), "unexpected: {reason}");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn rejects_rhs_length_mismatch() {
        let a = make_csr(3, 3, &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)]);
        let b = vec![1.0_f64, 2.0]; // wrong length
        let err = sparse_lu_solve(&a, &b).expect_err("length mismatch");
        match &err {
            LetoError::StorageError { reason } => {
                assert!(reason.contains("RHS length"), "unexpected: {reason}");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn csr_to_dense_round_trips() {
        let a = make_csr(2, 3, &[(0, 0, 1.0), (0, 2, 3.0), (1, 1, 2.0)]);
        let d = csr_to_dense(&a);
        assert_eq!(d.get([0, 0]).copied().unwrap(), 1.0);
        assert_eq!(d.get([0, 1]).copied().unwrap(), 0.0);
        assert_eq!(d.get([0, 2]).copied().unwrap(), 3.0);
        assert_eq!(d.get([1, 1]).copied().unwrap(), 2.0);
    }

    #[test]
    fn solver_is_generic_over_f32() {
        let mut coo = CooMatrix::new(2, 2);
        coo.push(0, 0, 2.0_f32);
        coo.push(0, 1, 0.0_f32);
        coo.push(1, 0, 0.0_f32);
        coo.push(1, 1, 4.0_f32);
        let a = coo.to_csr();
        let b = vec![6.0_f32, 8.0_f32];
        let rhs = Array1::from_shape_vec([2], b).expect("RHS shape");
        let x = SparseLuSolver::default()
            .solve_view(&a, &rhs.view())
            .expect("f32 solve");
        assert!((x[0] - 3.0_f32).abs() < 1e-5, "x[0] = {}", x[0]);
        assert!((x[1] - 2.0_f32).abs() < 1e-5, "x[1] = {}", x[1]);
    }
}
