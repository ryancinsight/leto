use crate::application::array::Array;
use crate::application::index::index_from_flat;
use crate::application::view::ArrayView;
use crate::domain::error::{LetoError, Result};
use crate::domain::insert_axis::InsertAxis;
use crate::domain::layout::Layout;
use crate::domain::remove_axis::RankMarker;
use crate::infrastructure::storage::VecStorage;

/// Stack equal-shaped rank-`N` views along a new `axis`, producing rank `M = N + 1`.
///
/// All inputs must share the same shape. The new axis has length equal to the
/// number of inputs and is inserted at position `axis` (valid range `0..=N`).
/// Output is C-contiguous, written in logical row-major order. The output rank
/// `M` is resolved at compile time through [`InsertAxis`]; call as
/// `stack::<T, N, M>(..)` where `M == N + 1`.
pub fn stack<T, const N: usize, const M: usize>(
    inputs: &[ArrayView<'_, T, N>],
    axis: usize,
) -> Result<Array<T, VecStorage<T>, M>>
where
    T: Clone,
    RankMarker<N>: InsertAxis<N, LargerShape = [usize; M]>,
{
    let Some((first, rest)) = inputs.split_first() else {
        return Err(LetoError::StorageError {
            reason: "stack requires at least one input".to_string(),
        });
    };

    let base_shape = first.shape();
    for view in rest {
        if view.shape() != base_shape {
            return Err(LetoError::ShapeMismatch {
                lhs: base_shape.to_vec(),
                rhs: view.shape().to_vec(),
            });
        }
    }

    let out_shape = RankMarker::<N>.insert_shape(base_shape, axis, inputs.len())?;
    let out_layout = Layout::c_contiguous(out_shape)?;
    let size = out_layout.size();
    let mut values = Vec::with_capacity(size);

    for flat in 0..size {
        let out_index = index_from_flat(flat, &out_shape);
        let which = out_index[axis];
        // Project the rank-M output index back to the rank-N source index by
        // dropping the stacked axis.
        let mut src_index = [0usize; N];
        for (j, slot) in src_index.iter_mut().enumerate() {
            *slot = if j < axis {
                out_index[j]
            } else {
                out_index[j + 1]
            };
        }
        values.push(inputs[which].get(src_index)?.clone());
    }

    Array::new(out_layout, VecStorage::new(values))
}
