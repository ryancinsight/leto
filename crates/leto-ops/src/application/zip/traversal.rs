use crate::application::index::RowMajorTraversal;
use leto::Result;

#[inline]
pub(super) fn for_each_row_major_indexed<RowState, InitRow, Visit, const N: usize>(
    traversal: RowMajorTraversal<N>,
    mut init_row: InitRow,
    mut visit: Visit,
) -> Result<()>
where
    InitRow: FnMut([usize; N]) -> Result<RowState>,
    Visit: FnMut([usize; N], &mut RowState),
{
    for row in 0..traversal.rows() {
        let mut index = traversal.base_index(row);
        let mut row_state = init_row(index)?;
        for k in 0..traversal.inner() {
            if N > 0 {
                index[N - 1] = k;
            }
            visit(index, &mut row_state);
        }
    }
    Ok(())
}
