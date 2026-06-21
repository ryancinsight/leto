use crate::application::array::Array;
use crate::application::view::ArrayView;
use crate::domain::error::Result;
use crate::domain::layout::Layout;
use crate::infrastructure::storage::VecStorage;

/// Per-axis `(before, after)` padding widths.
pub type PadWidth<const N: usize> = [(usize, usize); N];

/// Pad `input` with `fill` by `width` elements before and after on each axis,
/// allocating C-contiguous output.
///
/// Output dimension `d` is `before[d] + input[d] + after[d]`. Cells inside the
/// original region copy the source; cells in the pad margins take `fill`.
pub fn pad<T: Clone, const N: usize>(
    input: &ArrayView<'_, T, N>,
    width: PadWidth<N>,
    fill: T,
) -> Result<Array<T, VecStorage<T>, N>> {
    let in_shape = input.shape();
    let mut out_shape = [0usize; N];
    for d in 0..N {
        out_shape[d] = width[d].0 + in_shape[d] + width[d].1;
    }

    let out_layout = Layout::c_contiguous(out_shape)?;
    let size = out_layout.size();
    let mut values = vec![fill; size];

    for (src_index, val) in input.indexed_iter() {
        let mut out_index = [0usize; N];
        for d in 0..N {
            out_index[d] = src_index[d] + width[d].0;
        }
        let out_off = out_layout.offset_of(out_index)?;
        values[out_off] = val.clone();
    }

    Array::new(out_layout, VecStorage::new(values))
}
