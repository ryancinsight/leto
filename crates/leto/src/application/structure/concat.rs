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

    struct ConcatDropGuard<'a, T: Clone, const N: usize> {
        ptr: *mut std::mem::MaybeUninit<T>,
        out_layout: &'a Layout<N>,
        inputs: &'a [ArrayView<'a, T, N>],
        axis: usize,
        completed_inputs: usize,
        current_input_written: usize,
    }

    impl<'a, T: Clone, const N: usize> Drop for ConcatDropGuard<'a, T, N> {
        fn drop(&mut self) {
            if self.completed_inputs < self.inputs.len() {
                let mut base = 0usize;
                for i in 0..self.completed_inputs {
                    let input_view = &self.inputs[i];
                    let input_axis_len = input_view.shape()[self.axis];
                    for (src_index, _) in input_view.indexed_iter() {
                        let mut out_index = src_index;
                        out_index[self.axis] += base;
                        if let Ok(out_off) = self.out_layout.offset_of(out_index) {
                            unsafe {
                                std::ptr::drop_in_place(self.ptr.add(out_off) as *mut T);
                            }
                        }
                    }
                    base += input_axis_len;
                }
                if self.completed_inputs < self.inputs.len() {
                    let input_view = &self.inputs[self.completed_inputs];
                    for (idx, (src_index, _)) in input_view.indexed_iter().enumerate() {
                        if idx >= self.current_input_written {
                            break;
                        }
                        let mut out_index = src_index;
                        out_index[self.axis] += base;
                        if let Ok(out_off) = self.out_layout.offset_of(out_index) {
                            unsafe {
                                std::ptr::drop_in_place(self.ptr.add(out_off) as *mut T);
                            }
                        }
                    }
                }
            }
        }
    }

    let mut guard = ConcatDropGuard {
        ptr: values.as_mut_ptr(),
        out_layout: &out_layout,
        inputs,
        axis,
        completed_inputs: 0,
        current_input_written: 0,
    };

    let mut base = 0usize;
    for input_view in inputs {
        let input_axis_len = input_view.shape()[axis];
        guard.current_input_written = 0;
        for (src_index, val) in input_view.indexed_iter() {
            let mut out_index = src_index;
            out_index[axis] += base;
            let out_off = out_layout.offset_of(out_index)?;
            unsafe {
                guard
                    .ptr
                    .add(out_off)
                    .write(std::mem::MaybeUninit::new(val.clone()));
            }
            guard.current_input_written += 1;
        }
        base += input_axis_len;
        guard.completed_inputs += 1;
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
