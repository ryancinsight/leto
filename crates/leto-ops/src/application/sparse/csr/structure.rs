//! CSR structure validation: the shared accept/reject SSOT for owned
//! and borrowed construction.

use super::policy::{address, canonical, checked, CsrIndex, CsrOrdering};
use leto::{LetoError, Result};

/// Validate CSR parts against the [`CsrMatrix`] invariants.
///
/// Shared SSOT of [`CsrMatrix::from_parts`] and [`CsrView::from_slices`]:
/// identical triples accept/reject identically with identical messages.
/// Slice-only, so owned and borrowed construction validate the same way.
pub(super) fn validate_csr_parts<I: CsrIndex, O: CsrOrdering>(
    nnz: usize,
    col_indices: &[I],
    row_ptr: &[I],
    nrows: usize,
    ncols: usize,
) -> Result<()> {
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
    if col_indices.len() != nnz {
        return Err(bad("col_indices and values lengths differ"));
    }
    if row_ptr.iter().any(|&index| checked(index).is_none()) {
        return Err(bad("row offset is negative or exceeds usize"));
    }
    if address(row_ptr[0]) != 0 || address(row_ptr[nrows]) != nnz {
        return Err(bad("row_ptr must start at 0 and end at nnz"));
    }
    if row_ptr.windows(2).any(|w| address(w[0]) > address(w[1])) {
        return Err(bad("row_ptr must be non-decreasing"));
    }
    if col_indices
        .iter()
        .any(|&j| checked(j).is_none_or(|index| index >= ncols))
    {
        return Err(bad("column index out of range"));
    }
    if canonical::<O>() {
        for window in row_ptr.windows(2) {
            let row_cols = &col_indices[address(window[0])..address(window[1])];
            if row_cols
                .windows(2)
                .any(|cols| address(cols[0]) >= address(cols[1]))
            {
                return Err(bad(
                    "column indices in each row must be strictly increasing",
                ));
            }
        }
    }
    Ok(())
}
