use crate::application::view::ArrayView;
use crate::domain::error::{LetoError, Result};
use crate::domain::slice::SliceArg;

/// Split `input` along `axis` into consecutive zero-copy subviews whose lengths
/// are given by `sizes`.
///
/// `sizes` must sum to the input's length along `axis`. Each returned view
/// shares the input's backing storage; no data is copied.
pub fn split<'a, T, const N: usize>(
    input: &ArrayView<'a, T, N>,
    axis: usize,
    sizes: &[usize],
) -> Result<Vec<ArrayView<'a, T, N>>> {
    if axis >= N {
        return Err(LetoError::StorageError {
            reason: format!("split axis {axis} is out of bounds for rank {N}"),
        });
    }
    let axis_len = input.shape()[axis];
    let total: usize = sizes.iter().sum();
    if total != axis_len {
        return Err(LetoError::StorageError {
            reason: format!("split sizes sum to {total} but axis {axis} has length {axis_len}"),
        });
    }

    let mut views = Vec::with_capacity(sizes.len());
    let mut start = 0usize;
    for &len in sizes {
        let end = start + len;
        let mut args = [SliceArg::All; N];
        args[axis] = SliceArg::range(Some(start as isize), Some(end as isize), 1);
        views.push(input.slice_with::<N>(&args)?);
        start = end;
    }
    Ok(views)
}
