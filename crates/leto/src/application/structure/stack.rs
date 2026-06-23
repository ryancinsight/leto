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
    let mut values: Vec<std::mem::MaybeUninit<T>> = Vec::with_capacity(size);
    unsafe {
        values.set_len(size);
    }

    struct StackDropGuard<T> {
        ptr: *mut std::mem::MaybeUninit<T>,
        initialized: usize,
    }

    impl<T> Drop for StackDropGuard<T> {
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

    let mut guard = StackDropGuard {
        ptr: values.as_mut_ptr(),
        initialized: 0,
    };

    let outer_size: usize = out_shape[0..axis].iter().product();
    let inner_size: usize = out_shape[axis + 1..].iter().product();

    let mut iters: Vec<_> = inputs.iter().map(|v| v.iter()).collect();
    for outer_idx in 0..outer_size {
        for (which, iter) in iters.iter_mut().enumerate() {
            let out_off = (outer_idx * inputs.len() + which) * inner_size;
            for inner_idx in 0..inner_size {
                let val = iter.next().expect("iterator exhausted early");
                unsafe {
                    guard
                        .ptr
                        .add(out_off + inner_idx)
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
