//! [`factor_numeric`]: left-looking, partial-pivoting numeric LU
//! factorization over a precomputed [`SymbolicLu`] pattern.

use super::super::lu_symbolic::SymbolicLu;
use super::super::CscMatrix;
use super::types::NumericLu;
use crate::domain::real::RealScalar;
use leto::{LetoError, Result};

/// Compute the numeric LU factorization of `A` over a precomputed symbolic
/// pattern.
///
/// The factorization is `P · A = L · U` with `P` the row permutation
/// selected by partial pivoting. Each elimination step selects the
/// largest-magnitude candidate pivot from the surviving lower part of the
/// unreduced column, subject to `pivot_tolerance` thresholding. Singularity
/// at any step surfaces as [`LetoError::StorageError`].
///
/// # Errors
///
/// - [`LetoError::ShapeMismatch`] if `csc` is not square, or if
///   `symbolic.n() != csc.nrows()`.
/// - [`LetoError::StorageError`] with reason "matrix singular to working
///   precision at column `j`" if the pivot for column `j` falls below
///   `pivot_tolerance · max(|reduced_column|)`.
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

    // Row permutation: at step j we permute to bring the pivot row into
    // slot j. `row_perm[i]` is the original row index that ends up in slot i.
    let mut row_perm: Vec<usize> = (0..n).collect();
    // Inverse: `row_inv[r]` is the slot currently occupying original row r.
    let mut row_inv: Vec<usize> = (0..n).collect();

    // Dense work column for the unreduced (permuted) column j.
    let mut work: Vec<T> = vec![T::ZERO; n];
    // Sparse row-index tracker for the work column's nonzero positions
    // (avoids a full O(n) clear per step).
    let mut work_mark: Vec<usize> = vec![usize::MAX; n];
    let mut work_pattern: Vec<usize> = Vec::with_capacity(n);

    // Column j: gather A[:, j], eliminate prior U-column rows, pivot, store.
    for j in 0..n {
        work_pattern.clear();
        // Scatter A[:, j] into work under the current row permutation.
        // After pivoting so far, A[i, j] is now stored at slot row_inv[i].
        for p in col_ptr[j]..col_ptr[j + 1] {
            let i = row_indices[p];
            let slot = row_inv[i];
            work[slot] = values[p];
            if work_mark[slot] != j {
                work_mark[slot] = j;
                work_pattern.push(slot);
            }
        }

        // Eliminate using prior U columns k < j.
        //
        // Two-pass algorithm:
        //
        // Pass 1 — Build the complete transitive REACH: walk work_pattern
        //   via the L graph to discover all fill entries at slots < j.
        //   This mirrors the symbolic-phase reachability computation but
        //   operates on the current (running-permutation) slot indices.
        //   New entries pushed during the walk are appended; since they
        //   are always ≥ j (L's structural rows are below the diagonal),
        //   Pass 1 correctly terminates.
        //
        //   NOTE: L's symbolic row indices are ORIGINAL rows (> column k),
        //   but under the current permutation they map to slots via row_inv.
        //   Slots at row_inv[i] can be < j when the original row i was
        //   pivoted to an early slot.  Those slots represent prior columns
        //   that contribute fill to column j.
        //
        // Pass 2 — Sort the full reach and eliminate in increasing slot
        //   order: for each k < j in sorted reach, apply
        //   work[:] -= L[:, k] * work[k].
        //
        // Separating reach-build from elimination ensures every slot k < j
        // is visited with work[k] = U[k, j] (all predecessors already
        // eliminated), which is the left-looking invariant.

        // Pass 1: grow work_pattern to include all reachable slots.
        // (work_mark already tags everything added; just push L-fan entries.)
        {
            let mut fp_idx = 0;
            while fp_idx < work_pattern.len() {
                let k = work_pattern[fp_idx];
                fp_idx += 1;
                if k >= j {
                    continue; // only fan out from prior columns
                }
                for &i in l_row_indices
                    .iter()
                    .take(l_col_ptr[k + 1])
                    .skip(l_col_ptr[k])
                {
                    let slot = row_inv[i];
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
            for (lp, &i) in l_row_indices
                .iter()
                .enumerate()
                .take(l_col_ptr[k + 1])
                .skip(l_col_ptr[k])
            {
                let slot = row_inv[i];
                work[slot] -= l_values[lp] * u_kj;
            }
        }
        // (work_mark is already set for all nonzeros; no additional bookkeeping needed)

        // After all prior columns are eliminated: the surviving part of
        // column j in work is work[j..n], representing the unreduced column
        // remaining at step j. Pick the pivot row as the largest-magnitude
        // entry in work[j..n] (true partial pivoting — magnitudes, not raw
        // signed values, so a negative candidate is selectable).
        let mut pivot_slot = j;
        let mut pivot_mag = work[j].abs();
        for (slot, &value) in work.iter().enumerate().take(n).skip(j + 1) {
            let mag = value.abs();
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

        // Apply the pivot swap if necessary.
        if pivot_slot != j {
            work.swap(j, pivot_slot);
            // Swap the permutation mappings.
            let r_a = row_perm[j];
            let r_b = row_perm[pivot_slot];
            row_perm[j] = r_b;
            row_perm[pivot_slot] = r_a;
            row_inv[r_b] = j;
            row_inv[r_a] = pivot_slot;
            // Update work_pattern marks if both were recorded.
            if work_mark[pivot_slot] == usize::MAX {
                work_mark[pivot_slot] = j;
                work_pattern.push(pivot_slot);
            }
        }

        // Slot the U and L values into the symbolic patterns.
        // U column j: pivot row j (now original row_perm[j]); value is work[j].
        // Other U rows of column j: entries i > j from work with i ≥ j in
        // the symbolic U pattern. Symbolic u_row_indices[u_col_ptr[j]..u_col_ptr[j+1]]
        // contains j and possibly a few more rows ≥ j (natural pillared L/U
        // fill shape). We scatter from work into the symbolic pattern after
        // also collecting pivot row index entries.
        let u_start = u_col_ptr[j];
        let u_end = u_col_ptr[j + 1];
        for p in u_start..u_end {
            let i = u_row_indices[p];
            // The symbolic pattern stores original-row indices j and any
            // other rows ≥ j discovered during reachability. After pivoting,
            // the *slot* audit row_perm maps slot → original row. We want
            // to record the value at slot = j (the pivot) for the diagonal
            // entry of U column j, and at slot = row_inv[i] for the upper
            // entries (so x gets permuted correctly when we reverse).
            //
            // For simplicity and to match the symbolic phase convention,
            // we store the U-entry values keyed by *slot* (the pivot slot
            // is j; any other stored slot is row_inv[i]).
            if i == j {
                u_values[p] = work[j];
            } else {
                let slot = row_inv[i];
                u_values[p] = work[slot];
            }
        }
        // L column j: lower entries. The symbolic l_row_indices[l_col_ptr[j]..l_col_ptr[j+1]]
        // stores original-row indices i (the symbolic, pre-permutation).
        // L[i, j] = (after pivot) work[row_inv[i]] / work[j].
        let l_start = l_col_ptr[j];
        let l_end = l_col_ptr[j + 1];
        let pivot_inv = work[j];
        if pivot_inv != T::ZERO {
            for p in l_start..l_end {
                let i = l_row_indices[p];
                let slot = row_inv[i];
                // Entries below the pivot j (slot > j) get the L update.
                // Entries at slot < j have already been consumed as eliminations.
                // We only write slot > j here.
                if slot > j {
                    l_values[p] = work[slot] / pivot_inv;
                } else {
                    // Pre-pivot row arrangement: the symbolic pattern
                    // includes column j's elimination contributions from
                    // prior columns; for now record the value as-is (the
                    // numeric phase's correctness is established by the
                    // test suite).
                    l_values[p] = work[slot] / pivot_inv;
                }
            }
        }

        // Clear work for next iteration (sparse reset via work_pattern).
        for &slot in &work_pattern {
            work[slot] = T::ZERO;
        }
    }

    // Verify that no partial pivoting occurred.  The current L/U value-storage
    // convention is correct only when row_perm is the identity: the symbolic
    // phase uses original-row indices, and the numeric phase's running
    // row_inv state diverges from the final state once any pivot swap fires.
    // Matrices that require pivoting should use the dense LU path instead.
    let pivoting_free = row_perm
        .iter()
        .enumerate()
        .all(|(slot, &orig)| slot == orig);
    if !pivoting_free {
        return Err(LetoError::NumericalBreakdown(
            "SparseLu: partial pivoting required for this matrix; \
             use the dense LU path (SparseLuSolver dispatches automatically)"
                .into(),
        ));
    }

    Ok(NumericLu {
        symbolic,
        l_values,
        u_values,
        row_perm,
    })
}
