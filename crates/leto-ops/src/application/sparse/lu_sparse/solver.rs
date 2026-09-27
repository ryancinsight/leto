//! [`SparseLuSolver`]'s dense/sparse dispatch and solve/factor methods.

use super::owned::{OwnedLuRepr, OwnedNumericLu};
use super::types::SparseLuSolver;
use crate::application::linalg::lu::lu_decompose;
use crate::application::sparse::csc::CscMatrix;
use crate::application::sparse::csr::CsrMatrix;
use crate::application::sparse::lu_numeric::factor_numeric;
use crate::application::sparse::lu_symbolic::{factor_symbolic_with_ordering, SymbolicLu};
use crate::domain::real::RealScalar;
use leto::{Array1, ArrayView1, LetoError, Result};

impl SparseLuSolver {
    /// Returns `true` if the solver will attempt a direct solve for this `size`.
    #[must_use]
    #[inline]
    pub fn can_handle_size(&self, size: usize) -> bool {
        size <= self.max_size
    }

    fn validate_matrix<T: RealScalar>(&self, matrix: &CsrMatrix<T>) -> Result<usize> {
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
        if n > self.max_size {
            return Err(LetoError::StorageError {
                reason: format!(
                    "SparseLuSolver: system order {n} exceeds max_size {}; \
                     use an Athena iterative solver for large sparse systems",
                    self.max_size
                ),
            });
        }
        Ok(n)
    }

    fn validate<T: RealScalar>(&self, matrix: &CsrMatrix<T>, rhs_len: usize) -> Result<usize> {
        let n = self.validate_matrix(matrix)?;
        if rhs_len != n {
            return Err(LetoError::StorageError {
                reason: format!(
                    "SparseLuSolver: RHS length {rhs_len} does not match matrix order {n}"
                ),
            });
        }
        Ok(n)
    }

    /// Dispatch predicate: returns `true` if the dense path should be used
    /// for a matrix of order `n` with `nnz` stored nonzeros.
    pub(super) fn use_dense_path(&self, n: usize, nnz: usize) -> bool {
        if n <= self.small_switch {
            return true;
        }
        let density = (nnz as f64) / ((n as f64) * (n as f64));
        density >= self.density_threshold
    }

    /// Dense fallback (also the small-`n` primary path): CSR → dense →
    /// existing partial-pivoting LU.
    fn solve_dense_fallback<T: RealScalar>(
        &self,
        matrix: &CsrMatrix<T>,
        rhs: &ArrayView1<'_, T>,
    ) -> Result<Array1<T>> {
        let dense = super::csr_to_dense(matrix);
        let lu = lu_decompose(&dense.view())?;
        lu.solve(rhs)
    }

    /// Real sparse LU path: CSR → CSC → symbolic → numeric → solve.
    /// Falls back to the dense path if partial pivoting is required (the
    /// symbolic L/U convention is only correct for pivoting-free factorizations).
    ///
    /// When `self.ordering` is [`OrderingStrategy::AmdApproxMinDegree`](super::types::OrderingStrategy::AmdApproxMinDegree),
    /// the column-ordering permutation is applied symmetrically to the
    /// CSC before the symbolic phase runs; the numeric solve
    /// inverse-permutes the result back to original row/column order.
    fn solve_sparse_path<T: RealScalar>(
        &self,
        matrix: &CsrMatrix<T>,
        rhs: &ArrayView1<'_, T>,
    ) -> Result<Array1<T>> {
        let csc = CscMatrix::from_csr(matrix);
        let symbolic: SymbolicLu = factor_symbolic_with_ordering(&csc, self.ordering);
        // The numeric phase must operate on the same matrix the symbolic
        // was built for. For AMD that matrix is `A_perm = A[perm, perm]`,
        // so we reapply the permutation here when applicable.
        let factor_input = match &symbolic.amd_col_perm {
            Some(perm) => {
                let n = symbolic.n();
                debug_assert_eq!(perm.len(), n);
                let mut inv = vec![0usize; n];
                for (slot, &orig) in perm.iter().enumerate() {
                    inv[orig] = slot;
                }
                // Scatter A's nonzeros into permuted CSC.
                let src_col_ptr = csc.col_ptr();
                let src_row_indices = csc.row_indices();
                let src_values = csc.values();
                // Build a COO intermediate and dedupe via to_csc.
                let mut coo = crate::application::sparse::CooMatrix::new(n, n);
                for j in 0..n {
                    for p in src_col_ptr[j]..src_col_ptr[j + 1] {
                        let i = src_row_indices[p];
                        coo.push(inv[i], inv[j], src_values[p]);
                    }
                }
                coo.to_csc()
            }
            None => csc,
        };
        match factor_numeric(&factor_input, &symbolic, self.pivot_tolerance) {
            Ok(lu) => lu.solve(rhs),
            Err(LetoError::NumericalBreakdown(_)) => {
                // Partial pivoting needed; fall back to the dense LU path.
                self.solve_dense_fallback(matrix, rhs)
            }
            Err(e) => Err(e),
        }
    }

    /// Shared dense-vs-sparse solve dispatch for a validated matrix and RHS.
    #[inline]
    fn solve_with_dispatch<T: RealScalar>(
        &self,
        n: usize,
        matrix: &CsrMatrix<T>,
        rhs: &ArrayView1<'_, T>,
    ) -> Result<Array1<T>> {
        if self.use_dense_path(n, matrix.nnz()) {
            self.solve_dense_fallback(matrix, rhs)
        } else {
            self.solve_sparse_path(matrix, rhs)
        }
    }

    /// Solve `A · x = b` from a native Leto one-dimensional view.
    ///
    /// The right-hand side remains borrowed through validation and the
    /// dispatch decision (dense or sparse path). The returned solution
    /// owns only its result storage; no consumer-side `Vec` staging is
    /// required.
    ///
    /// # Dispatch
    ///
    ///- Matrices with `n ≤ self.small_switch` or `nnz/n² ≥
    ///  self.density_threshold` are routed through the dense partial-
    ///  pivoting LU path (the Atlas SSOT dense LU)._All other inputs are
    ///  routed through the real sparse LU path (CSC-based symbolic + numeric
    ///  factorization with partial pivoting; see ADR 0031)._
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::StorageError`] when the matrix is non-square, the
    /// right-hand side length does not match the matrix order, the system
    /// exceeds `max_size`, or the matrix is singular to working
    /// precision. Both dispatch paths surface the same typed errors.
    pub fn solve_view<T: RealScalar>(
        &self,
        matrix: &CsrMatrix<T>,
        rhs: &ArrayView1<'_, T>,
    ) -> Result<Array1<T>> {
        let n = self.validate(matrix, rhs.shape()[0])?;
        self.solve_with_dispatch(n, matrix, rhs)
    }

    /// Solve `A · x = b` for a sparse square system `A`.
    ///
    /// Dispatches to the dense or sparse path per [`Self::solve_view`].
    /// Returns [`LetoError::StorageError`] when:
    /// - `n > self.max_size` (system too large — use an iterative solver)
    /// - `matrix` is not square
    /// - `rhs.len() != n`
    /// - `matrix` is singular to the working precision of `T`
    pub fn solve<T: RealScalar>(&self, matrix: &CsrMatrix<T>, rhs: &[T]) -> Result<Vec<T>> {
        let n = self.validate(matrix, rhs.len())?;
        let rhs_array = Array1::from_shape_vec([n], rhs.to_vec())
            .expect("rhs length verified equal to n above");
        let x = self.solve_with_dispatch(n, matrix, &rhs_array.view())?;
        Ok(x.iter().copied().collect())
    }

    /// Factor `matrix` once for repeated solves, reusing a precomputed
    /// symbolic analysis of its sparsity pattern.
    ///
    /// The factor-phase analogue of [`Self::solve_view`]: the same dispatch
    /// criteria route small or near-dense matrices through the dense
    /// partial-pivoting LU, and a sparse factorization that reports
    /// pivoting-required falls back to the dense path — so the returned
    /// factor is defined for any nonsingular input within `max_size`.
    /// Callers with an unchanged pattern amortize
    /// [`factor_symbolic`](crate::application::sparse::factor_symbolic)
    /// across refactorizations (the CFDrs block-preconditioner cache is
    /// the driving consumer); `symbolic` is consulted only on the sparse
    /// arm and must describe `matrix`'s pattern.
    ///
    /// # Errors
    ///
    /// - [`LetoError::ShapeMismatch`] when `symbolic.n()` differs from the
    ///   matrix order.
    /// - [`LetoError::StorageError`] when the matrix is non-square, exceeds
    ///   `max_size`, or is singular to working precision.
    ///
    /// # Examples
    ///
    /// ```
    /// use leto_ops::application::sparse::{CooMatrix, factor_symbolic, CscMatrix, SparseLuSolver};
    /// use leto::Array1;
    ///
    /// let mut coo = CooMatrix::new(2, 2);
    /// coo.push(0, 0, 4.0_f64);
    /// coo.push(0, 1, 1.0_f64);
    /// coo.push(1, 0, 1.0_f64);
    /// coo.push(1, 1, 3.0_f64);
    /// let csr = coo.to_csr();
    /// let symbolic = factor_symbolic(&CscMatrix::from_csr(&csr));
    /// let factor = SparseLuSolver::default()
    ///     .factor_sparse_with_symbolic(&csr, &symbolic)
    ///     .expect("2x2 SPD factors");
    ///
    /// let b = Array1::from_shape_vec([2], vec![11.0_f64, 11.0]).expect("b shape");
    /// let mut x = Array1::from_shape_vec([2], vec![0.0_f64; 2]).expect("x shape");
    /// factor.solve_into(&b.view(), &mut x.view_mut()).expect("solve");
    /// assert!((x[0] - 2.0_f64).abs() < 1e-10);
    /// assert!((x[1] - 3.0_f64).abs() < 1e-10);
    /// ```
    pub fn factor_sparse_with_symbolic<T: RealScalar>(
        &self,
        matrix: &CsrMatrix<T>,
        symbolic: &SymbolicLu,
    ) -> Result<OwnedNumericLu<T>> {
        let n = self.validate_matrix(matrix)?;
        if symbolic.n() != n {
            return Err(LetoError::ShapeMismatch {
                lhs: vec![symbolic.n()],
                rhs: vec![n],
            });
        }
        if self.use_dense_path(n, matrix.nnz()) {
            return Self::factor_dense(matrix);
        }
        // If the caller's symbolic was produced under a column-ordering
        // strategy, the numeric phase must operate on the same
        // permuted matrix. Otherwise we fall back to the natural input.
        let csc = CscMatrix::from_csr(matrix);
        let (factor_input, col_perm_owned): (CscMatrix<T>, Vec<usize>) =
            match &symbolic.amd_col_perm {
                Some(perm) => {
                    let permuted =
                        crate::application::sparse::lu_symbolic::apply_symmetric_perm(&csc, perm);
                    (permuted, perm.clone())
                }
                None => {
                    let identity: Vec<usize> = (0..n).collect();
                    (csc, identity)
                }
            };
        match factor_numeric(&factor_input, symbolic, self.pivot_tolerance) {
            Ok(lu) => {
                let (l_values, u_values, row_perm) = lu.into_parts();
                Ok(OwnedNumericLu {
                    repr: OwnedLuRepr::Sparse {
                        symbolic: symbolic.clone(),
                        l_values,
                        u_values,
                        row_perm,
                        col_perm: col_perm_owned,
                    },
                })
            }
            // Partial pivoting needed; the sparse value-storage convention
            // cannot represent it (see factor_numeric) — factor dense.
            Err(LetoError::NumericalBreakdown(_)) => Self::factor_dense(matrix),
            Err(e) => Err(e),
        }
    }

    fn factor_dense<T: RealScalar>(matrix: &CsrMatrix<T>) -> Result<OwnedNumericLu<T>> {
        Ok(OwnedNumericLu {
            repr: OwnedLuRepr::Dense(lu_decompose(&super::csr_to_dense(matrix).view())?),
        })
    }
}
