//! LU factorization driver: input validation, working copy, and path dispatch.

use super::eliminate::{lu_eliminate_blocked, lu_eliminate_step};
use super::LuDecomposition;
use crate::domain::real::RealScalar;
use crate::infrastructure::cache::{
    cached_cache_geometry, working_set_spills_llc_for_geometry, CacheGeometry,
};
use leto::{Array2, ArrayView2, LetoError, Result};

/// Whether an `n × n` LU over `element_bytes`-wide scalars runs the blocked
/// (BLAS-3) path: exactly when the factor working set spills past the shared
/// LLC, the same cache-aware threshold the parallel policy uses
/// ([`working_set_spills_llc_for_geometry`]). Cache-resident sizes keep the
/// unblocked SIMD-axpy path, which runs at cache bandwidth there.
pub(crate) fn lu_uses_blocked_path(
    geometry: CacheGeometry,
    n: usize,
    element_bytes: usize,
) -> bool {
    let working_set = n.saturating_mul(n).saturating_mul(element_bytes);
    working_set_spills_llc_for_geometry(geometry, working_set)
}

/// Compute the partially pivoted LU decomposition of a square matrix.
///
/// The input may be strided/transposed; it is copied once into row-major
/// working storage. Elimination dispatches on the factor working set
/// (the crate-private `lu_uses_blocked_path` gate): past the shared LLC it
/// runs right-looking blocked LU with BLAS-3 trailing updates, while
/// cache-resident sizes keep
/// the unblocked SIMD-axpy path, which runs at cache bandwidth there
/// (benchmarks.md "Blocked LU cache-resident regression"). Returns
/// [`LetoError::ShapeMismatch`] for non-square input and
/// [`LetoError::StorageError`] when a pivot column is exactly zero (the
/// matrix is singular to working precision).
pub fn lu_decompose<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<LuDecomposition<T>> {
    let [rows, cols] = matrix.shape();
    if rows != cols {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![rows, cols],
            rhs: vec![rows, rows],
        });
    }
    let n = rows;

    // One bulk row-major copy into working storage + a single finiteness scan
    // (replacing per-element bounds-checked gets); elimination is in place.
    let mut a = matrix.to_contiguous().into_storage().into_inner();
    if !a.iter().all(|value| value.is_finite()) {
        return Err(LetoError::StorageError {
            reason: "LU input contains a non-finite value".to_string(),
        });
    }

    let mut pivots: Vec<usize> = (0..n).collect();
    let mut sign = 1i8;

    if lu_uses_blocked_path(cached_cache_geometry(), n, core::mem::size_of::<T>()) {
        lu_eliminate_blocked(&mut a, n, &mut pivots, &mut sign)?;
    } else {
        for k in 0..n {
            lu_eliminate_step(&mut a, n, k, n, &mut pivots, &mut sign)?;
        }
    }

    Ok(LuDecomposition {
        factors: Array2::from_shape_vec([n, n], a).expect("square factor storage"),
        pivots,
        sign,
    })
}
