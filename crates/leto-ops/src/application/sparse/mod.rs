//! Sparse matrices: assembly, dense↔sparse compression, and sparsity-exploiting
//! kernels.
//!
//! A matrix is *sparse* when most entries are zero. Storing and operating only
//! on the nonzeros turns dense `O(n²)` (matrix–vector) and `O(n³)` (products)
//! work into `O(nnz)` and `O(nnz·…)`, where `nnz` is the nonzero count — a large
//! win once density `nnz/(n·m)` is small.
//!
//! Three storage formats, by lifecycle phase (SoC):
//! - [`CooMatrix`](crate::application::sparse::CooMatrix) —
//!   **coordinate/triplet** list, the assembly target: each contribution is one
//!   [`push`](crate::application::sparse::CooMatrix::push); duplicates
//!   accumulate.
//! - [`CsrMatrix`] — **compressed sparse row**, the solve/kernel target consumed
//!   by [`spmv`] (matrix–vector) and [`spmm`] (sparse–dense product).
//! - [`CscMatrix`](crate::application::sparse::CscMatrix) — **compressed
//!   sparse column**, the column-major analogue of CSR. Accessed
//!   column-by-column via
//!   [`CscColumn`](crate::application::sparse::CscColumn) views; consumed by
//!   [`csc_spmv`](crate::application::sparse::csc_spmv) (matrix–vector).
//!
//! The canonical pipeline is *assemble in COO →
//! [`to_csr`](crate::application::sparse::CooMatrix::to_csr) → run CSR kernels*
//! or *assemble in COO →
//! [`to_csc`](crate::application::sparse::CooMatrix::to_csc) → run CSC
//! kernels*.  Both formats are generic over [`crate::domain::scalar::Scalar`]
//! at native precision; the kernels are hermes-SIMD-backed.
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
//! `Θ(nnz·k + m·k)`.
//! The [`CooMatrix`](crate::application::sparse::CooMatrix) theorem covers the
//! assembly→CSR step. ∎

mod coo;
mod csc;
mod csc_spmv;
mod csr;
mod lu_numeric;
mod lu_sparse;
mod lu_symbolic;
mod spgemm;
mod spmm;
mod spmv;

pub use coo::CooMatrix;
pub use csc::{CscColumn, CscMatrix};
pub use csc_spmv::{csc_spmv, csc_spmv_into};
pub use csr::{CsrMatrix, CsrRow};
pub use lu_numeric::{factor_numeric, NumericLu};
pub use lu_sparse::{csr_to_dense, sparse_lu_solve, SparseLuSolver, DENSE_LIMIT_DEFAULT};
pub use lu_symbolic::{factor_symbolic, SymbolicLu};
pub use spgemm::spgemm;
pub use spmm::{spmm, spmm_into};
pub use spmv::{spmv, spmv_into};


