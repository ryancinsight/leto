use crate::application::index::validate_mutable_output;
use leto::{ArrayView, ArrayViewMut, LetoError, Result};

/// Nonzero density of `lhs` at or below which [`matmul_auto`] routes to the
/// sparse CSR kernel instead of dense [`matmul`].
///
/// Cost model: dense matmul is `Θ(m·s·n)`; the sparse route is one `O(m·s)`
/// compression plus `Θ(nnz·n) = Θ(density·m·s·n)` (`spmm`). Ignoring the
/// sub-dominant compression, sparse beats dense by ≈ `1/density`, discounted by
/// the CSR gather's larger per-flop constant. A conservative `0.1` keeps the
/// sparse path strictly winning (measured ~17× at `0.05`) and never regresses the
/// dense majority case, which pays only the `O(m·s)` density scan.
// The policy selects among these existing const-generic instantiations. The
// 32-row specialization remains the conservative common-shape fallback and is
// also used for the fixed-size alpha panel in the depth-batched path.
pub(super) const MATMUL_ROW_BLOCK: usize = 32;
pub(super) const MATMUL_DEPTH_BLOCK: usize = 4;
pub(super) const MATMUL_DEPTH_BATCH_ROWS: usize = 128;

#[derive(Clone, Copy)]
pub(super) struct MatmulLayout {
    pub(super) rows: usize,
    pub(super) shared: usize,
    pub(super) cols: usize,
    pub(super) lhs_stride_row: isize,
    pub(super) lhs_stride_col: isize,
    pub(super) rhs_stride_row: isize,
    pub(super) rhs_stride_col: isize,
    pub(super) out_stride_row: isize,
    pub(super) out_stride_col: isize,
    pub(super) lhs_offset: isize,
    pub(super) rhs_offset: isize,
    pub(super) out_offset: isize,
}

#[inline]
pub(super) fn validate_matmul<T>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &ArrayViewMut<'_, T, 2>,
) -> Result<MatmulLayout> {
    let [rows, lhs_shared] = lhs.shape();
    let [rhs_shared, cols] = rhs.shape();
    let [out_rows, out_cols] = out.shape();

    if lhs_shared != rhs_shared || rows != out_rows || cols != out_cols {
        return Err(LetoError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }

    lhs.layout().validate_storage_len(lhs.data().len())?;
    rhs.layout().validate_storage_len(rhs.data().len())?;
    validate_mutable_output(out, "matmul")?;

    Ok(MatmulLayout {
        rows,
        shared: lhs_shared,
        cols,
        lhs_stride_row: lhs.strides()[0],
        lhs_stride_col: lhs.strides()[1],
        rhs_stride_row: rhs.strides()[0],
        rhs_stride_col: rhs.strides()[1],
        out_stride_row: out.strides()[0],
        out_stride_col: out.strides()[1],
        lhs_offset: lhs.offset() as isize,
        rhs_offset: rhs.offset() as isize,
        out_offset: out.offset() as isize,
    })
}

/// Raw base pointers and offsets for one matmul's three operands.
///
/// Bundles the six fields `dot_matmul_row`/`outer_matmul_row` and their
/// callers all thread through, so a row kernel's argument count reflects its
/// own indices and dimensions rather than the operand plumbing every matmul
/// variant repeats identically.
#[derive(Clone, Copy)]
pub(super) struct MatmulPtrs<T> {
    pub(super) lhs: *const T,
    pub(super) rhs: *const T,
    pub(super) out: *mut T,
    pub(super) lhs_offset: usize,
    pub(super) rhs_offset: usize,
    pub(super) out_offset: usize,
}
