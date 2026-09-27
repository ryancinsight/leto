//! [`SparseLuSolver`] configuration, its dispatch constants, and
//! [`OrderingStrategy`].

/// Default maximum order for which a direct solve is attempted.
/// Systems above this size return [`LetoError::StorageError`](leto::LetoError::StorageError)
/// directing callers to iterative solvers. The threshold applies to both the
/// dense and sparse paths.
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
/// [`factor_symbolic_with_ordering`](super::super::lu_symbolic::factor_symbolic_with_ordering);
/// see [`amd_order`](crate::application::sparse::amd_order) for the AMD
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
    /// Systems larger than this return [`LetoError::StorageError`](leto::LetoError::StorageError)
    /// directing callers to use an iterative solver.
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
