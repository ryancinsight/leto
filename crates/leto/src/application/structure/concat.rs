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

    let mut values: Vec<std::mem::MaybeUninit<T>> = Vec::with_capacity(size);
    unsafe {
        values.set_len(size);
    }

    struct ConcatDropGuard<T> {
        ptr: *mut std::mem::MaybeUninit<T>,
        initialized: usize,
    }

    impl<T> Drop for ConcatDropGuard<T> {
        fn drop(&mut self) {
            if self.initialized > 0 && std::mem::needs_drop::<T>() {
                for i in 0..self.initialized {
                    unsafe {
                        std::ptr::drop_in_place(self.ptr.add(i) as *mut T);
                    }
                }
            }
        }
    }

    let mut guard = ConcatDropGuard {
        ptr: values.as_mut_ptr(),
        initialized: 0,
    };

    let outer_size: usize = out_shape[0..axis].iter().product();
    let inner_size: usize = out_shape[axis + 1..].iter().product();

    // Precalculate block sizes and offsets along axis.
    let mut block_sizes = Vec::with_capacity(inputs.len());
    let mut base_offsets = Vec::with_capacity(inputs.len());
    let mut current_base = 0usize;
    for view in inputs {
        let input_axis_len = view.shape()[axis];
        block_sizes.push(input_axis_len * inner_size);
        base_offsets.push(current_base);
        current_base += input_axis_len;
    }
    let out_axis_total = current_base;

    let iters: Vec<_> = inputs.iter().map(|v| v.iter()).collect();
    let mut zipped_iters: Vec<(_, usize, usize)> = iters
        .into_iter()
        .zip(block_sizes.iter().copied())
        .zip(base_offsets.iter().copied())
        .map(|((iter, size), base)| (iter, size, base))
        .collect();

    for outer_idx in 0..outer_size {
        for (iter, block_size, base_offset) in &mut zipped_iters {
            let out_off = outer_idx * out_axis_total * inner_size + *base_offset * inner_size;
            for i in 0..*block_size {
                let val = iter.next().expect("iterator exhausted early");
                unsafe {
                    guard
                        .ptr
                        .add(out_off + i)
                        .write(std::mem::MaybeUninit::new(val.clone()));
                }
                guard.initialized += 1;
            }
        }
    }

    std::mem::forget(guard);

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
