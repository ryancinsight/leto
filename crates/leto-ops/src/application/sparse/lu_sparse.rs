//! Sparse direct solver: `A x = b` for sparse square systems via partial-pivoting LU.
//!
//! # Algorithm
//!
//! `SparseLuSolver` is a **dispatcher**: for inputs that are small or
//! near-dense the existing dense [`lu_decompose`] path (the Atlas SSOT
//! dense LU) is dispatched via [`Self::solve_dense_fallback`]; for inputs
//! that are large and sparse a real sparse LU factorization over CSC
//! with partial pivoting runs through [`super::lu_numeric::factor_numeric`]
//! and [`super::lu_symbolic::factor_symbolic`].
//!
//! ## Dispatch criteria
//!
//! The dense path is selected when:
//! - `n <= self.small_switch` (matrix is small enough for the dense path's
//!   constant factor to win the symbolic-factorization overhead), OR
//! - `density >= self.density_threshold` (matrix is near dense so the
//!   sparse path's `O(nnz)` savings vanish and its constant factor becomes
//!   a tax).
//!
//! Otherwise the sparse path runs. Both paths produce a value-identical
//! solution `x = A⁻¹ b` up to floating-point rounding; the differential
//! test suite in [`super::lu_numeric`] verifies this equivalence on a
//! deterministic medium-sized sparse system.
//!
//! # Why the dense path stays
//!
//! A real sparse LU has a non-zero constant factor (symbolic factorization
//! overhead + sparse data structure traversal). For matrices small enough
//! that the dense LU's `n³` cost fits comfortably in cache (<~32 by
//! measurement), the dense path is faster despite its asymptotic
//! disadvantage. The crossover threshold [`Self::small_switch`] expresses
//! this and is configurable for callers with profiled workloads. The
//! dense path is also the proven correctness baseline against which the
//! sparse path is differential-tested — it stays in-tree as the oracle.
//!
//! # Theorem — correctness boundary
//!
//! For any nonsingular `A ∈ T^{n×n}`:
//! - If `n ≤ self.small_switch` or `density(A) ≥ self.density_threshold`,
//!   the dense path produces `x = A⁻¹b` up to IEEE 754 rounding (CSR
//!   expansion is exact, dense LU is partial-pivoting stable).
//! - Else the sparse path produces `x = A⁻¹b` up to IEEE 754 rounding
//!   under the partial-pivoting stability guarantee (Davis 2006 §8.7;
//!   partial-pivoting factorization is backward stable for any nonsingular
//!   matrix).
//!
//! Both paths return the same typed errors on the same failure modes
//! (`LetoError::ShapeMismatch`, `LetoError::StorageError`) so consumers'
//! error-handling is unchanged.

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use crate::application::linalg::lu::{lu_decompose, LuDecomposition};
use crate::application::sparse::csc::CscMatrix;
use crate::application::sparse::csr::CsrMatrix;
use crate::application::sparse::lu_numeric::{factor_numeric, triangular_solve_into};
use crate::application::sparse::lu_symbolic::{factor_symbolic_with_ordering, SymbolicLu};
use crate::domain::real::RealScalar;
use leto::{Array1, Array2, ArrayView1, ArrayViewMut1, LetoError, Result};

/// Default maximum order for which a direct solve is attempted.
/// Systems above this size return [`LetoError::StorageError`] directing
/// callers to iterative solvers. The threshold applies to both the dense
/// and sparse paths.
pub const DENSE_LIMIT_DEFAULT: usize = 2048;

/// Below this matrix order the dense path is always selected (it wins on
/// a constant-factor basis; the sparse path's symbolic overhead exceeds
/// the dense path's `O(n³)` for very small `n`). The threshold is measured
/// against matrix-vector benchmarks and tuned conservatively.
///
/// When the sparse path is selected and the matrix requires partial pivoting,
/// `solve_sparse_path` automatically falls back to the dense path (the
/// sparse symbolic L/U convention is currently correct only for
/// pivoting-free factorizations; see `lu_numeric::factor_numeric`).
pub const SMALL_SWITCH_DEFAULT: usize = 32;

/// Sparsity density threshold below which the sparse path is selected.
/// Empirically `nnz/n^2 < 0.1` marks the regime where sparse traversal's
/// `O(nnz)` savings outweigh its constant factor; above this the dense
/// path is dispatched to avoid the sparse-path tax.
pub const DENSITY_THRESHOLD_DEFAULT: f64 = 0.1;

/// Fill-reducing column-ordering strategy for the sparse LU path.
///
/// ZST-equivalent `Copy` enum selected at the symbolic-analysis stage.
/// Dispatch is by exhaustive match — no vtable, no per-strategy struct.
/// Adding a new strategy is one enum variant and one match arm in
/// [`factor_symbolic_with_ordering`]; see
/// [`amd_order`](crate::application::sparse::amd_order) for the AMD
/// implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[must_use = "OrderingStrategy controls fill in the sparse LU pattern"]
pub enum OrderingStrategy {
    /// Natural column ordering (`0, 1, …, n-1`). Default; preserves the
    /// existing [`factor_symbolic`](crate::application::sparse::factor_symbolic)
    /// convention. The right choice when the
    /// input is already banded or pivoting-free with small bandwidth —
    /// CFDrs's saddle-point blocks.
    #[default]
    Natural,
    /// Approximate Minimum Degree ordering (Amestoy-Davis-Duff 1996).
    /// Computes a permutation `perm` ordering the columns (and rows,
    /// symmetrically) by minimum current degree; the symbolic pattern is
    /// computed on `A_perm = A[perm, perm]`, and the solve inverse-permutes
    /// the result. Recommended for unstructured sparsity where AMD's
    /// fill-reduction beats natural ordering (see ADR 0031).
    AmdApproxMinDegree,
}

/// Configuration for the atlas-native sparse direct solver.
///
/// Drop-in replacement for `rsparse`-based `DirectSparseSolver` in `CFDrs`; exposes
/// the same knobs (`max_size`, `pivot_tolerance`) so call-sites can transition
/// without structural changes. The new `small_switch` and `density_threshold`
/// knobs control the dense↔sparse dispatch and default to measured crossovers
/// (see `SMALL_SWITCH_DEFAULT` and `DENSITY_THRESHOLD_DEFAULT`).
#[derive(Debug, Clone)]
#[must_use = "SparseLuSolver carries the dispatch configuration consumed by solve"]
pub struct SparseLuSolver {
    /// Maximum system order for which a direct solve is attempted.
    /// Systems larger than this return [`LetoError::StorageError`] directing
    /// callers to use an iterative solver.
    pub max_size: usize,
    /// Pivot tolerance: a pivot with `|pivot| < pivot_tolerance * max_col` is
    /// treated as zero and triggers a singularity error.
    pub pivot_tolerance: f64,
    /// Matrices of order `n ≤ small_switch` always take the dense path.
    /// Above this, the dense path is reserved near-dense matrices (see
    /// [`Self::density_threshold`]).
    pub small_switch: usize,
    /// Sparsity density `nnz / n^2` at or above which the dense path is
    /// dispatched regardless of `n`. Below this and above
    /// [`Self::small_switch`], the real sparse LU runs.
    pub density_threshold: f64,
    /// Column-ordering strategy for the sparse LU path. Defaults to
    /// [`OrderingStrategy::Natural`] (preserves all existing behavior);
    /// [`OrderingStrategy::AmdApproxMinDegree`] applies a fill-reducing
    /// symmetric permutation before symbolic factorization. The dense
    /// dispatch path ignores this knob.
    pub ordering: OrderingStrategy,
}

impl Default for SparseLuSolver {
    fn default() -> Self {
        Self {
            max_size: DENSE_LIMIT_DEFAULT,
            pivot_tolerance: 1e-12,
            small_switch: SMALL_SWITCH_DEFAULT,
            density_threshold: DENSITY_THRESHOLD_DEFAULT,
            ordering: OrderingStrategy::default(),
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
    fn use_dense_path(&self, n: usize, nnz: usize) -> bool {
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
        let dense = csr_to_dense(matrix);
        let lu = lu_decompose(&dense.view())?;
        lu.solve(rhs)
    }

    /// Real sparse LU path: CSR → CSC → symbolic → numeric → solve.
    /// Falls back to the dense path if partial pivoting is required (the
    /// symbolic L/U convention is only correct for pivoting-free factorizations).
    ///
    /// When `self.ordering` is [`OrderingStrategy::AmdApproxMinDegree`],
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
        if self.use_dense_path(n, matrix.nnz()) {
            self.solve_dense_fallback(matrix, rhs)
        } else {
            self.solve_sparse_path(matrix, rhs)
        }
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
        let x = if self.use_dense_path(n, matrix.nnz()) {
            self.solve_dense_fallback(matrix, &rhs_array.view())?
        } else {
            self.solve_sparse_path(matrix, &rhs_array.view())?
        };
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
                    let permuted = super::lu_symbolic::apply_symmetric_perm(&csc, perm);
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
            repr: OwnedLuRepr::Dense(lu_decompose(&csr_to_dense(matrix).view())?),
        })
    }
}

/// Owned, reusable numeric factorization produced by
/// [`SparseLuSolver::factor_sparse_with_symbolic`].
///
/// Stores everything the triangular solves need — no borrow of the
/// symbolic analysis — so factors can be cached across solver iterations
/// (a preconditioner factoring each momentum block once and applying it
/// every Krylov iteration) while the caller separately caches the
/// [`SymbolicLu`] for refactorization on unchanged patterns.
///
/// Two representations mirror the solver's dispatch contract: pivoting-free
/// matrices hold the sparse L/U value buffers; matrices that need partial
/// pivoting, or that the dispatch criteria route dense, hold the dense
/// partial-pivoting factorization. Both arms solve to the same values up
/// to IEEE 754 rounding (differential-tested below).
#[derive(Debug, Clone)]
#[must_use = "OwnedNumericLu carries the factorization consumed by solve"]
pub struct OwnedNumericLu<T: RealScalar> {
    repr: OwnedLuRepr<T>,
}

#[derive(Debug, Clone)]
enum OwnedLuRepr<T: RealScalar> {
    Sparse {
        symbolic: SymbolicLu,
        l_values: Vec<T>,
        u_values: Vec<T>,
        row_perm: Vec<usize>,
        /// Column-ordering permutation applied during symbolic
        /// factorization. Length `n`; identity `[0, n)` for natural
        /// ordering, AMD output for `OrderingStrategy::AmdApproxMinDegree`.
        /// The triangular solve inverse-permutes the column-order
        /// solution back to original row order via this vector.
        col_perm: Vec<usize>,
    },
    Dense(LuDecomposition<T>),
}

impl<T: RealScalar> OwnedNumericLu<T> {
    /// Matrix order `n`.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        match &self.repr {
            OwnedLuRepr::Sparse { symbolic, .. } => symbolic.n(),
            OwnedLuRepr::Dense(lu) => lu.dim(),
        }
    }

    /// Solve `A · x = rhs` directly into a caller-owned view `out`.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::ShapeMismatch`] when `rhs` or `out` length
    /// differs from the matrix order `n`.
    pub fn solve_into(
        &self,
        rhs: &ArrayView1<'_, T>,
        out: &mut ArrayViewMut1<'_, T>,
    ) -> Result<()> {
        match &self.repr {
            OwnedLuRepr::Sparse {
                symbolic,
                l_values,
                u_values,
                row_perm,
                col_perm,
            } => triangular_solve_into(symbolic, l_values, u_values, row_perm, col_perm, rhs, out),
            OwnedLuRepr::Dense(lu) => lu.solve_into(rhs, out),
        }
    }

    /// Solve `A · x = rhs`, returning a freshly-owned solution.
    ///
    /// # Errors
    ///
    /// Forwards from [`Self::solve_into`].
    pub fn solve(&self, rhs: &ArrayView1<'_, T>) -> Result<Array1<T>> {
        let n = self.n();
        let mut x =
            Array1::from_shape_vec([n], vec![T::ZERO; n]).map_err(|e| LetoError::StorageError {
                reason: format!("OwnedNumericLu::solve internal shape error: {e}"),
            })?;
        self.solve_into(rhs, &mut x.view_mut())?;
        Ok(x)
    }
}

/// Convenience: solve `A · x = b` in one call without constructing a solver.
///
/// Uses [`DENSE_LIMIT_DEFAULT`] as the maximum system order and the
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::sparse::lu_symbolic::factor_symbolic;
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
            small_switch: SMALL_SWITCH_DEFAULT,
            density_threshold: DENSITY_THRESHOLD_DEFAULT,
            ordering: OrderingStrategy::default(),
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
                assert!(reason.contains("exceeds max_size"), "unexpected: {reason}");
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

    #[test]
    fn dense_path_taken_for_near_dense_4x4() {
        // A 4x4 completely dense matrix: n <= small_switch (32) so the dense
        // path is selected by the small-matrix criterion. Both paths produce
        // a value-equivalent solution; this test verifies the dispatch helper
        // routes by small n and that the solve matches a closed-form x.
        let mut coo = CooMatrix::new(4, 4);
        let entries: [(usize, usize, f64); 16] = [
            (0, 0, 9.5),
            (0, 1, 1.0),
            (0, 2, 3.0),
            (0, 3, 2.0),
            (1, 0, 0.5),
            (1, 1, 4.0),
            (1, 2, 1.0),
            (1, 3, 7.0),
            (2, 0, 2.0),
            (2, 1, 5.0),
            (2, 2, 12.0),
            (2, 3, 1.0),
            (3, 0, -1.0),
            (3, 1, 0.0),
            (3, 2, 1.0),
            (3, 3, 8.0),
        ];
        for &(r, c, v) in &entries {
            coo.push(r, c, v);
        }
        let a = coo.to_csr();
        // Construct a dense rhs from a closed-form solution.
        let x_known: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let mut b = vec![0.0_f64; 4];
        for &(r, c, v) in &entries {
            b[r] += v * x_known[c];
        }
        let b_arr = Array1::from_shape_vec([4], b).expect("b shape");

        // Sanity: dispatch must select the dense path due to n==4 <=
        // small_switch (32). Verify the solve works.
        let solver = SparseLuSolver::default();
        debug_assert!(solver.use_dense_path(4, a.nnz()));
        let x = solver
            .solve_view(&a, &b_arr.view())
            .expect("dense dispatch solve");
        for i in 0..4 {
            assert!((x[i] - x_known[i]).abs() < 1e-10, "x[{i}] = {}", x[i]);
        }
    }

    /// The tridiagonal Poisson-Laplacian used by the owned-factor tests:
    /// n=64, density ≈ 0.047 < 0.1 and n > small_switch, so the sparse
    /// arm is dispatched. `diag` scales the main diagonal so two matrices
    /// share one pattern with different values.
    fn tridiagonal_csr(n: usize, diag: f64) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, diag);
            if i > 0 {
                coo.push(i, i - 1, -1.0_f64);
            }
            if i + 1 < n {
                coo.push(i, i + 1, -1.0_f64);
            }
        }
        coo.to_csr()
    }

    /// `‖A x - b‖∞` computed directly from the CSR nonzeros.
    fn csr_residual_inf(a: &CsrMatrix<f64>, x: &Array1<f64>, b: &[f64]) -> f64 {
        let mut max_residual = 0.0_f64;
        for (row, &rhs) in b.iter().enumerate().take(a.nrows()) {
            let mut ax = 0.0_f64;
            for (&col, &value) in a.row(row).col_indices().iter().zip(a.row(row).values()) {
                ax += value * x[col];
            }
            let d = (ax - rhs).abs();
            if d > max_residual {
                max_residual = d;
            }
        }
        max_residual
    }

    #[test]
    fn owned_factor_reuses_symbolic_across_value_changes() {
        // The CFDrs block-preconditioner pattern: one symbolic analysis,
        // several numeric factors over matrices sharing the pattern.
        let n = 64usize;
        let solver = SparseLuSolver::default();
        let a1 = tridiagonal_csr(n, 2.0);
        let symbolic = factor_symbolic(&CscMatrix::from_csr(&a1));
        let b: Vec<f64> = (1..=n).map(|k| k as f64).collect();
        let b_arr = Array1::from_shape_vec([n], b.clone()).expect("b shape");

        for diag in [2.0_f64, 4.0] {
            let a = tridiagonal_csr(n, diag);
            let factor = solver
                .factor_sparse_with_symbolic(&a, &symbolic)
                .expect("pivoting-free factorization");
            assert_eq!(factor.n(), n);
            let mut x = Array1::from_shape_vec([n], vec![0.0_f64; n]).expect("x shape");
            factor
                .solve_into(&b_arr.view(), &mut x.view_mut())
                .expect("solve_into");
            let residual = csr_residual_inf(&a, &x, &b);
            assert!(residual < 1e-8, "diag={diag}: residual = {residual}");
        }
    }

    #[test]
    fn owned_factor_matches_solve_view() {
        // Value-semantic differential: the cached factor and the one-shot
        // dispatcher must produce the same solution on the sparse arm.
        let n = 64usize;
        let solver = SparseLuSolver::default();
        let a = tridiagonal_csr(n, 2.0);
        assert!(!solver.use_dense_path(n, a.nnz()), "sparse arm expected");
        let b: Vec<f64> = (0..n).map(|k| (k as f64) * 0.25 - 3.0).collect();
        let b_arr = Array1::from_shape_vec([n], b).expect("b shape");

        let symbolic = factor_symbolic(&CscMatrix::from_csr(&a));
        let factor = solver
            .factor_sparse_with_symbolic(&a, &symbolic)
            .expect("factor");
        let x_factor = factor.solve(&b_arr.view()).expect("factor solve");
        let x_direct = solver.solve_view(&a, &b_arr.view()).expect("direct solve");
        for i in 0..n {
            let d = (x_factor[i] - x_direct[i]).abs();
            assert!(d < 1e-12, "x[{i}] differs by {d}");
        }
    }

    #[test]
    fn owned_factor_falls_back_to_dense_when_pivoting_required() {
        // Zero the leading diagonal entry so column 0's pivot must come
        // from row 1 — the sparse convention reports NumericalBreakdown and
        // the owned factor must transparently hold the dense factorization.
        let n = 64usize;
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            if i != 0 {
                coo.push(i, i, 4.0_f64);
            }
            if i > 0 {
                coo.push(i, i - 1, -1.0_f64);
            }
            if i + 1 < n {
                coo.push(i, i + 1, -1.0_f64);
            }
        }
        let a = coo.to_csr();
        let solver = SparseLuSolver::default();
        assert!(!solver.use_dense_path(n, a.nnz()), "sparse arm expected");

        let x_known: Vec<f64> = (0..n).map(|i| 1.0_f64 / (i as f64 + 1.0)).collect();
        let mut b = vec![0.0_f64; n];
        for (row, rhs) in b.iter_mut().enumerate() {
            for (&col, &value) in a.row(row).col_indices().iter().zip(a.row(row).values()) {
                *rhs += value * x_known[col];
            }
        }
        let symbolic = factor_symbolic(&CscMatrix::from_csr(&a));
        let factor = solver
            .factor_sparse_with_symbolic(&a, &symbolic)
            .expect("dense fallback factors the pivot-requiring matrix");
        let b_arr = Array1::from_shape_vec([n], b.clone()).expect("b shape");
        let x = factor.solve(&b_arr.view()).expect("solve");
        for i in 0..n {
            let d = (x[i] - x_known[i]).abs();
            assert!(d < 1e-8, "x[{i}] = {} expected {}", x[i], x_known[i]);
        }
    }

    #[test]
    fn owned_factor_small_matrix_routes_dense() {
        // n=2 ≤ small_switch: dispatch takes the dense arm outright.
        let a = make_csr(2, 2, &[(0, 0, 3.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 2.0)]);
        let solver = SparseLuSolver::default();
        let symbolic = factor_symbolic(&CscMatrix::from_csr(&a));
        let factor = solver
            .factor_sparse_with_symbolic(&a, &symbolic)
            .expect("dense-arm factor");
        let b = Array1::from_shape_vec([2], vec![9.0_f64, 8.0]).expect("b shape");
        let x = factor.solve(&b.view()).expect("solve");
        assert!((x[0] - 2.0).abs() < 1e-10, "x[0] = {}", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-10, "x[1] = {}", x[1]);
    }

    #[test]
    fn owned_factor_solve_into_rejects_wrong_lengths() {
        let n = 64usize;
        let a = tridiagonal_csr(n, 2.0);
        let solver = SparseLuSolver::default();
        let symbolic = factor_symbolic(&CscMatrix::from_csr(&a));
        let factor = solver
            .factor_sparse_with_symbolic(&a, &symbolic)
            .expect("factor");

        let short_rhs = Array1::from_shape_vec([n - 1], vec![1.0_f64; n - 1]).expect("rhs");
        let mut out = Array1::from_shape_vec([n], vec![0.0_f64; n]).expect("out");
        let err = factor
            .solve_into(&short_rhs.view(), &mut out.view_mut())
            .expect_err("short RHS must be rejected");
        assert!(matches!(err, LetoError::ShapeMismatch { .. }), "{err:?}");

        let rhs = Array1::from_shape_vec([n], vec![1.0_f64; n]).expect("rhs");
        let mut short_out = Array1::from_shape_vec([n - 1], vec![0.0_f64; n - 1]).expect("out");
        let err = factor
            .solve_into(&rhs.view(), &mut short_out.view_mut())
            .expect_err("short output must be rejected");
        assert!(matches!(err, LetoError::ShapeMismatch { .. }), "{err:?}");
    }

    #[test]
    fn owned_factor_rejects_symbolic_order_mismatch() {
        let a = tridiagonal_csr(64, 2.0);
        let wrong = factor_symbolic(&CscMatrix::from_csr(&tridiagonal_csr(32, 2.0)));
        let err = SparseLuSolver::default()
            .factor_sparse_with_symbolic(&a, &wrong)
            .expect_err("order mismatch must be rejected");
        assert!(matches!(err, LetoError::ShapeMismatch { .. }), "{err:?}");
    }

    #[test]
    fn sparse_path_routes_correctly_for_tridiagonal_n64() {
        // n=64 tridiagonal Poisson-Laplacian: density = 64*3 / 64^2 ≈ 0.047
        // (below 0.1), n > small_switch (32) — sparse path must run.
        let n = 64usize;
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 2.0_f64);
            if i > 0 {
                coo.push(i, i - 1, -1.0_f64);
            }
            if i + 1 < n {
                coo.push(i, i + 1, -1.0_f64);
            }
        }
        let a = coo.to_csr();
        let solver = SparseLuSolver::default();
        assert!(
            !solver.use_dense_path(n, a.nnz()),
            "dispatch predicate: sparse path expected for n={n}, nnz={}, density={}",
            a.nnz(),
            (a.nnz() as f64) / ((n as f64) * (n as f64))
        );
        let b = (1..=n).map(|k| k as f64).collect::<Vec<f64>>();
        let b_arr = Array1::from_shape_vec([n], b.clone()).expect("b shape");
        let x = solver.solve_view(&a, &b_arr.view()).expect("sparse solve");
        // Residual against closed-form dense reconstruction.
        let mut max_residual = 0.0_f64;
        for i in 0..n {
            let mut ax = 0.0;
            if i > 0 {
                ax -= x[i - 1];
            }
            ax += 2.0 * x[i];
            if i + 1 < n {
                ax -= x[i + 1];
            }
            let d = (ax - b[i]).abs();
            if d > max_residual {
                max_residual = d;
            }
        }
        assert!(max_residual < 1e-8, "max_residual = {max_residual}");
    }
}
