//! Sparse matrices: automatic dense→sparse compression and sparsity-exploiting
//! kernels.
//!
//! A matrix is *sparse* when most entries are zero. Storing and operating only
//! on the nonzeros turns dense `O(n²)` (matrix–vector) and `O(n³)` (products)
//! work into `O(nnz)` and `O(nnz·…)`, where `nnz` is the nonzero count — a large
//! win once density `nnz/(n·m)` is small. This module owns the **Compressed
//! Sparse Row (CSR)** representation and the kernels over it; [`spmv`] is the
//! sparse matrix-vector product and [`spmm`] is the sparse-dense product.
//!
//! # Theorem (CSR exactly represents the matrix; SpMV is `O(nnz)`)
//! Let `A ∈ Tᵐˣⁿ` with `nnz` nonzeros. The CSR triple `(values, col_indices,
//! row_ptr)` — `values[p]` and `col_indices[p]` the value and column of the
//! `p`-th nonzero in row-major nonzero order, `row_ptr[i]` the index in `values`
//! where row `i` begins (`row_ptr[m] = nnz`) — satisfies, for every `(i, j)`,
//!
//! ```text
//! A[i,j] = Σ_{p ∈ [row_ptr[i], row_ptr[i+1])}  [col_indices[p] = j] · values[p]
//! ```
//!
//! *Proof.* `from_dense` appends, scanning row `i` left to right, exactly the
//! `(j, A[i,j])` with `A[i,j] ≠ 0` and stamps `row_ptr[i]`/`row_ptr[i+1]` around
//! that run; entries omitted are zero by construction, so the sum picks out the
//! single stored `p` with `col_indices[p] = j` (CSR holds at most one entry per
//! `(i,j)`) or is empty (value `0`). Hence the stored matrix equals `A`. The
//! product `y = A x` is `y[i] = Σ_j A[i,j] x[j] = Σ_{p ∈ row i} values[p]·
//! x[col_indices[p]]`, by the identity above — exactly the loop [`spmv`] runs,
//! touching each nonzero once: `Θ(nnz + m)` time, versus dense `Θ(m·n)`.
//! [`spmm`] extends the same identity across each dense RHS column in
//! `Θ(nnz·k + m·k)`. ∎
//!
//! Vertical structure (SoC): this module owns the [`CsrMatrix`] type and the
//! dense↔sparse conversions; [`spmv`] and [`spmm`] own sparse-dense kernels.
//! Generic over [`crate::domain::scalar::Scalar`]; native precision.

mod spmm;
mod spmv;

pub use spmm::{spmm, spmm_into};
pub use spmv::{spmv, spmv_into};

use crate::domain::scalar::Scalar;
use leto::{Array2, ArrayView2, LetoError, Result};

/// Compressed Sparse Row matrix: stores only the `nnz` nonzero entries.
///
/// Invariants (established by [`CsrMatrix::from_dense`] and required by
/// [`CsrMatrix::from_parts`]): `row_ptr.len() == nrows + 1`, `row_ptr` is
/// non-decreasing with `row_ptr[0] == 0` and `row_ptr[nrows] == values.len()`,
/// `col_indices.len() == values.len()`, every `col_indices[p] < ncols`, and
/// column indices are strictly increasing within each row.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix<T> {
    values: Vec<T>,
    col_indices: Vec<usize>,
    row_ptr: Vec<usize>,
    nrows: usize,
    ncols: usize,
}

impl<T: Scalar> CsrMatrix<T> {
    /// Compress a dense matrix view into CSR, dropping every exact-zero entry.
    ///
    /// This is the automatic dense→sparse path: it scans the matrix once
    /// (`O(n·m)`) and retains only nonzeros, after which every kernel over the
    /// result is `O(nnz)`. Worthwhile when [`density`](Self::density) is small.
    #[must_use = "from_dense returns the compressed matrix"]
    pub fn from_dense(matrix: &ArrayView2<'_, T>) -> Self {
        let [nrows, ncols] = matrix.shape();
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptr = Vec::with_capacity(nrows + 1);
        row_ptr.push(0);

        if let Some(dense) = matrix.as_slice() {
            for i in 0..nrows {
                let row = &dense[i * ncols..i * ncols + ncols];
                for (j, &value) in row.iter().enumerate() {
                    if value != T::ZERO {
                        values.push(value);
                        col_indices.push(j);
                    }
                }
                row_ptr.push(values.len());
            }
        } else {
            let strides = matrix.strides();
            let data = matrix.data();
            for i in 0..nrows {
                if ncols == 0 {
                    row_ptr.push(values.len());
                    continue;
                }
                let mut offset = matrix
                    .layout()
                    .offset_of([i, 0])
                    .expect("row start is in bounds") as isize;
                for j in 0..ncols {
                    let value = data[offset as usize];
                    if value != T::ZERO {
                        values.push(value);
                        col_indices.push(j);
                    }
                    offset += strides[1];
                }
                row_ptr.push(values.len());
            }
        }

        Self {
            values,
            col_indices,
            row_ptr,
            nrows,
            ncols,
        }
    }

    /// Construct CSR from raw parts, validating the structural invariants.
    ///
    /// # Errors
    /// [`LetoError::StorageError`] if any CSR invariant is violated (mismatched
    /// lengths, non-monotone `row_ptr`, or an out-of-range column index).
    pub fn from_parts(
        values: Vec<T>,
        col_indices: Vec<usize>,
        row_ptr: Vec<usize>,
        nrows: usize,
        ncols: usize,
    ) -> Result<Self> {
        let bad = |reason: &str| LetoError::StorageError {
            reason: format!("invalid CSR: {reason}"),
        };
        let expected_row_ptr_len = nrows
            .checked_add(1)
            .ok_or_else(|| LetoError::StorageError {
                reason: "invalid CSR: nrows + 1 overflows usize".to_string(),
            })?;
        if row_ptr.len() != expected_row_ptr_len {
            return Err(bad("row_ptr length must be nrows + 1"));
        }
        if col_indices.len() != values.len() {
            return Err(bad("col_indices and values lengths differ"));
        }
        if row_ptr[0] != 0 || *row_ptr.last().expect("nrows+1 >= 1") != values.len() {
            return Err(bad("row_ptr must start at 0 and end at nnz"));
        }
        if row_ptr.windows(2).any(|w| w[0] > w[1]) {
            return Err(bad("row_ptr must be non-decreasing"));
        }
        if col_indices.iter().any(|&j| j >= ncols) {
            return Err(bad("column index out of range"));
        }
        for window in row_ptr.windows(2) {
            let row_cols = &col_indices[window[0]..window[1]];
            if row_cols.windows(2).any(|cols| cols[0] >= cols[1]) {
                return Err(bad(
                    "column indices in each row must be strictly increasing",
                ));
            }
        }
        Ok(Self {
            values,
            col_indices,
            row_ptr,
            nrows,
            ncols,
        })
    }

    /// `(nrows, ncols)`.
    #[must_use]
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Number of stored nonzero entries.
    #[must_use]
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Fraction of entries that are nonzero, in `[0, 1]` (`0` for an empty matrix).
    #[must_use]
    #[inline]
    pub fn density(&self) -> f64 {
        let total = self.nrows * self.ncols;
        if total == 0 {
            0.0
        } else {
            self.nnz() as f64 / total as f64
        }
    }

    /// Borrowed CSR arrays `(values, col_indices, row_ptr)` — zero-copy, for
    /// kernels (e.g. [`spmv`]) and for bridging to an external SpMV backend.
    #[must_use]
    #[inline]
    pub fn as_parts(&self) -> (&[T], &[usize], &[usize]) {
        (&self.values, &self.col_indices, &self.row_ptr)
    }

    /// Reconstruct the dense matrix (`O(n·m)`; inverse of [`from_dense`](Self::from_dense)).
    #[must_use]
    pub fn to_dense(&self) -> Array2<T> {
        let mut dense = vec![T::ZERO; self.nrows * self.ncols];
        for i in 0..self.nrows {
            for p in self.row_ptr[i]..self.row_ptr[i + 1] {
                dense[i * self.ncols + self.col_indices[p]] = self.values[p];
            }
        }
        Array2::from_shape_vec([self.nrows, self.ncols], dense).expect("CSR dense shape is valid")
    }
}
