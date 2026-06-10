use crate::application::array::Array;
use crate::application::index::index_from_flat;
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
    let mut values = Vec::with_capacity(size);

    for flat in 0..size {
        let out_index = index_from_flat(flat, &out_shape);
        let mut src_index = [0usize; N];
        let mut inside = true;
        for d in 0..N {
            let before = width[d].0;
            if out_index[d] < before || out_index[d] >= before + in_shape[d] {
                inside = false;
                break;
            }
            src_index[d] = out_index[d] - before;
        }
        if inside {
            values.push(input.get(src_index)?.clone());
        } else {
            values.push(fill.clone());
        }
    }

    Array::new(out_layout, VecStorage::new(values))
}
