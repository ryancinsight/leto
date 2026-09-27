//! Triangular-solve path: [`NumericLu::solve`]/[`NumericLu::solve_into`] and
//! the shared [`triangular_solve_into`] core reused by `lu_sparse::OwnedNumericLu`.

use super::super::lu_symbolic::SymbolicLu;
use super::types::NumericLu;
use crate::domain::real::RealScalar;
use leto::{Array1, ArrayView1, ArrayViewMut1, LetoError, Result};

impl<'a, T: RealScalar> NumericLu<'a, T> {
    /// Solve `A · x = b` using the precomputed factorization.
    ///
    /// Returns a freshly-owned `Array1<T>`. The RHS is borrowed; no
    /// consumer-side staging vector is needed.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::ShapeMismatch`] if `rhs.len() != n`.
    pub fn solve(&self, rhs: &ArrayView1<'_, T>) -> Result<Array1<T>> {
        let n = self.n();
        let mut x =
            Array1::from_shape_vec([n], vec![T::ZERO; n]).map_err(|e| LetoError::StorageError {
                reason: format!("NumericLu::solve internal shape error: {e}"),
            })?;
        self.solve_into(rhs, &mut x.view_mut())?;
        Ok(x)
    }

    /// Solve `A · x = b` directly into a caller-owned view `out`.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::ShapeMismatch`] if `rhs` or `out` length
    /// differs from the matrix order `n`.
    pub fn solve_into(
        &self,
        rhs: &ArrayView1<'_, T>,
        out: &mut ArrayViewMut1<'_, T>,
    ) -> Result<()> {
        let n = self.symbolic.n;
        // If the symbolic carries an AMD column permutation, the solve
        // path needs it for the final inverse-scatter. For the natural
        // case `amd_col_perm` is `None` and we materialize an identity
        // `[0, n)` slice; the matrix inverse-scatter collapses to the
        // identity write.
        let identity_perm_owned;
        let col_perm: &[usize] = match &self.symbolic.amd_col_perm {
            Some(p) => p,
            None => {
                identity_perm_owned = (0..n).collect::<Vec<_>>();
                &identity_perm_owned
            }
        };
        triangular_solve_into(
            self.symbolic,
            &self.l_values,
            &self.u_values,
            &self.row_perm,
            col_perm,
            rhs,
            out,
        )
    }
}

/// Shared triangular-solve core: `y = P · rhs`, forward-substitute
/// `L · z = y`, back-substitute `U · x = z`, scatter `x` back to original
/// row order into `out`, then apply the column permutation inverse if one
/// was used during symbolic factorization.
///
/// One implementation serves both the borrowing [`NumericLu`] and the
/// owning `lu_sparse::OwnedNumericLu`; the storage convention (original-row
/// indices in the symbolic patterns, slot mapping via the inverse
/// permutation) is stated here once.
///
/// # Column-permutation contract
///
/// `col_perm: &[usize]` is the column-order permutation applied during
/// symbolic factorization (`perm[i]` = the original column/row that ended
/// up at slot `i` in `A_perm`). For natural ordering it is the identity
/// `0..n`. For AMD under [`crate::application::sparse::amd::amd_order`]
/// it is the AMD output, and `A_perm = A[perm, perm]` was the matrix
/// actually factorized. The slot-order solution `x_slot` produced by the
/// triangular solves satisfies `A_perm · x_slot = rhs_perm`; the original
/// solution is `x[perm[i]] = x_slot[i]`, computed here as a final
/// scatter that overwrites `out` in original order.
pub(in crate::application::sparse) fn triangular_solve_into<T: RealScalar>(
    symbolic: &SymbolicLu,
    l_values: &[T],
    u_values: &[T],
    row_perm: &[usize],
    col_perm: &[usize],
    rhs: &ArrayView1<'_, T>,
    out: &mut ArrayViewMut1<'_, T>,
) -> Result<()> {
    let n = symbolic.n();
    if rhs.shape()[0] != n {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![rhs.shape()[0]],
            rhs: vec![n],
        });
    }
    if out.shape()[0] != n {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![out.shape()[0]],
            rhs: vec![n],
        });
    }
    let mut y = vec![T::ZERO; n];

    // Step 1: y = P · b  (permute RHS into slot order)
    for (slot, &orig_row) in row_perm.iter().enumerate() {
        y[slot] = rhs.get([orig_row]).copied().unwrap_or(T::ZERO);
    }

    // Precompute inverse permutation: row_inv[orig_row] = slot.
    let mut row_inv = vec![0usize; n];
    for (slot, &orig) in row_perm.iter().enumerate() {
        row_inv[orig] = slot;
    }

    // Step 2: Forward-substitute L · z = y.
    // L is unit lower triangular; l_row_indices stores ORIGINAL row
    // indices i > j (in unpermuted order). Map i → slot via row_inv
    // so the update lands in the permuted y buffer.
    let l_col_ptr = &symbolic.l_col_ptr;
    let l_row_indices = &symbolic.l_row_indices;
    for j in 0..n {
        let yj = y[j];
        for (p, &orig_i) in l_row_indices
            .iter()
            .enumerate()
            .take(l_col_ptr[j + 1])
            .skip(l_col_ptr[j])
        {
            let slot_i = row_inv[orig_i];
            y[slot_i] -= l_values[p] * yj;
        }
    }

    // Step 3: Back-substitute U · x = y.
    // U is stored in CSC format with ORIGINAL row indices.  The diagonal
    // entries satisfy u_row_indices[p] == j (slot == original for pivots).
    // Off-diagonal entries at original row i < j are at slot row_inv[i].
    let u_col_ptr = &symbolic.u_col_ptr;
    let u_row_indices = &symbolic.u_row_indices;
    let mut x = y; // reuse buffer; overwritten column by column
    for j in (0..n).rev() {
        // Divide by diagonal U[j,j]  (u_row_indices[p] == j is slot j).
        let u_diag = u_row_indices
            .iter()
            .enumerate()
            .take(u_col_ptr[j + 1])
            .skip(u_col_ptr[j])
            .find(|(_, &r)| r == j)
            .map(|(p, _)| p);
        if let Some(p) = u_diag {
            x[j] = x[j] / u_values[p];
        }
        // Propagate: x[slot_i] -= U[orig_i, j] · x[j] for orig_i < j.
        let xj = x[j];
        for (p, &orig_i) in u_row_indices
            .iter()
            .enumerate()
            .take(u_col_ptr[j + 1])
            .skip(u_col_ptr[j])
        {
            if orig_i < j {
                let slot_i = row_inv[orig_i];
                x[slot_i] -= u_values[p] * xj;
            }
        }
    }

    // Step 4: Unscramble — x is in slot order; scatter back to the
    // column-permuted row order. row_perm[slot] is the (column-permuted)
    // original row at that slot, so `slot_x[row_perm[slot]] = x[slot]`
    // yields the solution to `A_perm · x_perm = b_perm` in `A_perm`'s
    // row/column order.
    //
    // If `col_perm` is the identity (natural ordering) this is the
    // original row order and we're done — write directly into `out`.
    // If `col_perm` is nontrivial (AMD), the natural row order here is
    // `A_perm`'s row order; we compose through `col_perm` to scatter into
    // the original `A`'s row order. The two cases share one write loop:
    // the identity `col_perm` is the no-op of the AMD scatter.
    let natural = col_perm.iter().enumerate().all(|(i, &p)| p == i);
    if natural {
        for (slot, &orig_row) in row_perm.iter().enumerate() {
            *out.get_mut([orig_row])
                .expect("invariant: orig_row < n and out length checked above") = x[slot];
        }
    } else {
        // AMD path: first scatter to slot order (col_perm-frame row view),
        // then compose to the original-row order through `col_perm`. We
        // reuse `row_inv` to mark the slot for each col-permuted row,
        // then map to the original row.
        //
        // slot_x[col_perm-slot-row] = x[slot]; we want `out[orig_row] = x[slot]`
        // where orig_row is the original matrix row equivalent to the
        // AMD-permuted row at `row_perm[slot]`. Under the symmetric AMD
        // permutation `A_perm = A[perm, perm]`, AMD-row `r` corresponds to
        // original row `perm[r]`. So `out[perm[row_perm[slot]]] = x[slot]`.
        for (slot, &perm_row) in row_perm.iter().enumerate() {
            let orig_row = col_perm[perm_row];
            *out.get_mut([orig_row])
                .expect("invariant: orig_row < n and out length checked above") = x[slot];
        }
    }
    Ok(())
}
