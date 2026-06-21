use crate::application::array::Array;
use crate::application::view::ArrayView;
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;
use crate::infrastructure::storage::VecStorage;

/// Concatenate views along an existing `axis`, allocating C-contiguous output.
///
/// Every input must share the same shape on all axes except `axis`. The output
/// axis length is the sum of the inputs' lengths along `axis`. Output values
/// are written exactly once in row-major order.
pub fn concat<T: Clone, const N: usize>(
    inputs: &[ArrayView<'_, T, N>],
    axis: usize,
) -> Result<Array<T, VecStorage<T>, N>> {
    if axis >= N {
        return Err(LetoError::StorageError {
            reason: format!("concat axis {axis} is out of bounds for rank {N}"),
        });
    }
    let Some((first, rest)) = inputs.split_first() else {
        return Err(LetoError::StorageError {
            reason: "concat requires at least one input".to_string(),
        });
    };

    let mut out_shape = first.shape();
    let mut axis_total = first.shape()[axis];
    for view in rest {
        let shape = view.shape();
        for (d, (&a, &b)) in out_shape.iter().zip(shape.iter()).enumerate() {
            if d != axis && a != b {
                return Err(LetoError::ShapeMismatch {
                    lhs: out_shape.to_vec(),
                    rhs: shape.to_vec(),
                });
            }
        }
        axis_total += shape[axis];
    }
    out_shape[axis] = axis_total;

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
    unsafe {
        values.set_len(size);
    }

    let mut base = 0usize;
    for input_view in inputs {
        let input_axis_len = input_view.shape()[axis];
        for (src_index, val) in input_view.indexed_iter() {
            let mut out_index = src_index;
            out_index[axis] += base;
            let out_off = out_layout.offset_of(out_index)?;
            values[out_off].write(val.clone());
        }
        base += input_axis_len;
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
