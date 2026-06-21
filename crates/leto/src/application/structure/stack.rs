use crate::application::array::Array;
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
    if axis == 0 {
        let mut values: Vec<T> = Vec::with_capacity(size);
        for input_view in inputs {
            values.extend(input_view.iter().cloned());
        }
        return Array::new(out_layout, VecStorage::new(values));
    }

    let mut values: Vec<std::mem::MaybeUninit<T>> = Vec::with_capacity(size);
    unsafe { values.set_len(size); }

    for (which, input_view) in inputs.iter().enumerate() {
        for (src_index, val) in input_view.indexed_iter() {
            let mut out_index = [0usize; M];
            for j in 0..M {
                if j < axis {
                    out_index[j] = src_index[j];
                } else if j == axis {
                    out_index[j] = which;
                } else {
                    out_index[j] = src_index[j - 1];
                }
            }
            let out_off = out_layout.offset_of(out_index)?;
            values[out_off].write(val.clone());
        }
    }

    let values = unsafe {
        let mut values = std::mem::ManuallyDrop::new(values);
        Vec::from_raw_parts(
            values.as_mut_ptr() as *mut T,
            values.len(),
            values.capacity(),
        )
    };

    Array::new(out_layout, VecStorage::new(values))
}
