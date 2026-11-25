//! [`factor_numeric`]: left-looking numeric LU
//! factorization over a precomputed [`SymbolicLu`] pattern.

use super::super::lu_symbolic::SymbolicLu;
use super::super::CscMatrix;
use super::types::NumericLu;
use crate::domain::real::RealScalar;
use leto::{LetoError, Result};

/// Compute the numeric LU factorization of `A` over a precomputed symbolic
/// pattern.
///
/// The symbolic pattern uses natural row order. Each column tests the
/// largest-magnitude surviving entry. A required row exchange hands the
/// original matrix back to the dispatcher for pivoted dense factorization.
/// A successful sparse factor therefore has the identity row permutation.
///
/// # Errors
///
/// - [`LetoError::ShapeMismatch`] if `csc` is not square, or if
///   `symbolic.n() != csc.nrows()`.
/// - [`LetoError::StorageError`] with reason "matrix singular to working
///   precision at column `j`" if the pivot for column `j` falls below
///   `pivot_tolerance · max(|reduced_column|)`.
/// - [`LetoError::NumericalBreakdown`] when a row exchange is required.
///
/// # Examples
///
/// ```
/// use leto_ops::application::sparse::{CooMatrix, factor_numeric, factor_symbolic};
///
/// let mut coo = CooMatrix::new(2, 2);
/// coo.push(0, 0, 4.0_f64);
/// coo.push(0, 1, 1.0_f64);
/// coo.push(1, 0, 1.0_f64);
/// coo.push(1, 1, 3.0_f64);
/// let csc = coo.to_csc();
/// let symbolic = factor_symbolic(&csc);
/// let lu = factor_numeric(&csc, &symbolic, 1e-12).expect("2x2 SPD recovery");
///
/// use leto::{Array1};
/// let b = Array1::from_shape_vec([2], vec![11.0_f64, 11.0]).expect("b shape");
/// let x = lu.solve(&b.view()).expect("solve");
/// assert!((x[0] - 2.0_f64).abs() < 1e-10);
/// assert!((x[1] - 3.0_f64).abs() < 1e-10);
/// ```
pub fn factor_numeric<'a, T: RealScalar>(
    csc: &CscMatrix<T>,
    symbolic: &'a SymbolicLu,
    pivot_tolerance: f64,
) -> Result<NumericLu<'a, T>> {
    let (nrows, ncols) = csc.shape();
    if nrows != ncols {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![nrows, ncols],
            rhs: vec![nrows, nrows],
        });
    }
    let n = nrows;
    if symbolic.n() != n {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![symbolic.n()],
            rhs: vec![n],
        });
    }

    let tol = T::from_f64(pivot_tolerance.max(0.0));

    // CSC indices of A.
    let col_ptr = csc.col_ptr();
    let row_indices = csc.row_indices();
    let values = csc.values();

    // Symbolic L/U column pointers and row indices.
    let l_col_ptr: &Vec<usize> = &symbolic.l_col_ptr;
    let l_row_indices: &Vec<usize> = &symbolic.l_row_indices;
    let u_col_ptr: &Vec<usize> = &symbolic.u_col_ptr;
    let u_row_indices: &Vec<usize> = &symbolic.u_row_indices;

    // Allocate numeric value buffers matching the symbolic patterns.
    let mut l_values: Vec<T> = vec![T::ZERO; symbolic.l_row_indices.len()];
    let mut u_values: Vec<T> = vec![T::ZERO; symbolic.u_row_indices.len()];

    // Dense workspace for one natural-order column.
    let mut work: Vec<T> = vec![T::ZERO; n];
    // Sparse row-index tracker for the work column's nonzero positions
    // (avoids a full O(n) clear per step).
    let mut work_mark: Vec<usize> = vec![usize::MAX; n];
    let mut work_pattern: Vec<usize> = Vec::with_capacity(n);

    // Column j: gather A[:, j], eliminate prior U-column rows, pivot, store.
    for j in 0..n {
        work_pattern.clear();
        for p in col_ptr[j]..col_ptr[j + 1] {
            let slot = row_indices[p];
            work[slot] = values[p];
            if work_mark[slot] != j {
                work_mark[slot] = j;
                work_pattern.push(slot);
            }
        }

        // Discover all fill through previously factored columns before
        // elimination so every predecessor is visited in dependency order.
        {
            let mut fp_idx = 0;
            while fp_idx < work_pattern.len() {
                let k = work_pattern[fp_idx];
                fp_idx += 1;
                if k >= j {
                    continue; // only fan out from prior columns
                }
                for &slot in l_row_indices
                    .iter()
                    .take(l_col_ptr[k + 1])
                    .skip(l_col_ptr[k])
                {
                    if work_mark[slot] != j {
                        work_mark[slot] = j;
                        work_pattern.push(slot);
                    }
                }
            }
        }

        // Pass 2: sort the full reach and eliminate in order.
        work_pattern.sort_unstable();
        for &k in &work_pattern {
            if k >= j {
                continue;
            }
            let u_kj = work[k];
            for (lp, &slot) in l_row_indices
                .iter()
                .enumerate()
                .take(l_col_ptr[k + 1])
                .skip(l_col_ptr[k])
            {
                work[slot] -= l_values[lp] * u_kj;
            }
        }

        // Unvisited entries are zero. Keep the diagonal on magnitude ties.
        let mut pivot_slot = j;
        let mut pivot_mag = work[j].abs();
        for &slot in &work_pattern[work_pattern.partition_point(|&slot| slot <= j)..] {
            let mag = work[slot].abs();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot_slot = slot;
            }
        }
        // Under true partial pivoting the pivot IS the column max, so the
        // relative threshold below fires only at an exactly zero column for
        // tol < 1; the form is kept literal to honor the documented
        // `pivot_tolerance` contract.
        let max_in_col = pivot_mag;

        // Singularity check: pivot magnitude below tol * max_in_col.
        if max_in_col == T::ZERO || pivot_mag < tol * max_in_col {
            return Err(LetoError::StorageError {
                reason: format!(
                    "SparseLu: matrix singular to working precision at column {j} \
                     (pivot magnitude {pivot_mag:?} < tolerance * {max_in_col:?})"
                ),
            });
        }

        if pivot_slot != j {
            return Err(LetoError::NumericalBreakdown(
                "SparseLu: partial pivoting required for this matrix; \
                 use the dense LU path (SparseLuSolver dispatches automatically)"
                    .into(),
            ));
        }

        for p in u_col_ptr[j]..u_col_ptr[j + 1] {
            u_values[p] = work[u_row_indices[p]];
        }
        for p in l_col_ptr[j]..l_col_ptr[j + 1] {
            l_values[p] = work[l_row_indices[p]] / work[j];
        }

        // Clear work for next iteration (sparse reset via work_pattern).
        for &slot in &work_pattern {
            work[slot] = T::ZERO;
        }
    }

    Ok(NumericLu {
        symbolic,
        l_values,
        u_values,
        row_perm: (0..n).collect(),
    })
}
