use crate::application::array::Array;
use crate::application::view::{ArrayView, ArrayViewMut};
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
        axis_total = axis_total
            .checked_add(shape[axis])
            .ok_or(LetoError::Overflow {
                reason: "concatenated axis extent",
            })?;
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

/// Concatenate views along an existing `axis` into caller-owned `output`.
///
/// The direct-output dual of [`concat()`]: elements flow from the inputs into
/// `output` without an intermediate owned element array. Auxiliary metadata
/// uses O(input count) space. `output`'s shape must
/// equal the concatenated shape (inputs agreeing on every axis but `axis`,
/// `axis` extent equal to the sum of the inputs' extents there); its layout
/// must be injective, since every output element is written exactly once.
///
/// C-dense inputs with C-dense output copy block-wise; every other layout
/// (transposed, sliced, or offset views on either side) copies in logical
/// row-major order. `output` must not alias any input.
///
/// # Errors
///
/// Returns [`LetoError::StorageError`] when `axis` is out of bounds or no
/// input is given, [`LetoError::ShapeMismatch`] when the inputs disagree off
/// `axis` or `output`'s shape misses the concatenated shape, [`LetoError::Overflow`]
/// when the concatenated extent is unrepresentable, and whatever
/// [`ArrayViewMut::indexed_iter_mut`] reports when `output`'s layout aliases.
pub fn concat_into<T: Clone, const N: usize>(
    inputs: &[ArrayView<'_, T, N>],
    axis: usize,
    output: &mut ArrayViewMut<'_, T, N>,
) -> Result<()> {
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
        axis_total = axis_total
            .checked_add(shape[axis])
            .ok_or(LetoError::Overflow {
                reason: "concatenated axis extent",
            })?;
    }
    out_shape[axis] = axis_total;

    if output.shape() != out_shape {
        return Err(LetoError::ShapeMismatch {
            lhs: out_shape.to_vec(),
            rhs: output.shape().to_vec(),
        });
    }

    // Empty shapes contain no addressed elements; avoid irrelevant prefix products.
    if out_shape.contains(&0) {
        return Ok(());
    }

    // Dense fast path: every input's row-major order is its physical order
    // and so is the output's, so each input lands as whole block copies.
    if let Some(dst) = output.as_mut_slice() {
        let mut sources = Vec::with_capacity(inputs.len());
        let mut all_dense = true;
        for view in inputs {
            match view.as_slice() {
                Some(src) => sources.push(src),
                None => {
                    all_dense = false;
                    break;
                }
            }
        }
        if all_dense {
            let outer_size: usize = out_shape[0..axis].iter().product();
            let inner_size: usize = out_shape[axis + 1..].iter().product();
            let mut base = 0usize;
            for (view, src) in inputs.iter().zip(sources.iter()) {
                let block = view.shape()[axis] * inner_size;
                for outer in 0..outer_size {
                    let out_off = outer * axis_total * inner_size + base * inner_size;
                    let src_off = outer * block;
                    dst[out_off..out_off + block].clone_from_slice(&src[src_off..src_off + block]);
                }
                base += view.shape()[axis];
            }
            return Ok(());
        }
        // A strided input fell through with no copy performed yet, so the
        // general path below can rewrite every output element from scratch.
    }

    // General strided path: walk the output in logical row-major order and
    // pull each element from the input owning its `axis` lane.
    let mut bases = Vec::with_capacity(inputs.len());
    let mut base = 0usize;
    for view in inputs {
        bases.push(base);
        base += view.shape()[axis];
    }
    for (out_index, slot) in output.reborrow().indexed_iter_mut()? {
        let lane = out_index[axis];
        let mut src_index = out_index;
        for (view, &view_base) in inputs.iter().zip(bases.iter()) {
            if lane >= view_base && lane < view_base + view.shape()[axis] {
                src_index[axis] = lane - view_base;
                *slot = view.get(src_index)?.clone();
                break;
            }
        }
    }
    Ok(())
}
