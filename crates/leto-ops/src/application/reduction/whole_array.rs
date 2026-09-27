//! All-elements reduction (no keep-dim axis) via [`AxisReduction`].

use super::strategies::AxisReduction;
use crate::application::index::{unit_stride_row_slice, RowMajorTraversal};
use crate::domain::scalar::Scalar;
use leto::{ArrayView, LetoError, Result};

/// Reduce every element of `input` into one scalar using an [`AxisReduction`]
/// marker.
///
/// The operation validates the input view storage, rejects empty inputs for
/// reductions without an identity (for example [`MinAxis`](super::MinAxis)
/// and [`MaxAxis`](super::MaxAxis)), and follows logical view
/// order for strided and sliced layouts without materializing a contiguous
/// copy.
pub fn reduce_all<Op, T, const N: usize>(input: &ArrayView<'_, T, N>) -> Result<T>
where
    Op: AxisReduction<T>,
    T: Scalar,
{
    input.layout().validate_storage_len(input.data().len())?;
    let size = input.layout().checked_size()?;
    if size == 0 {
        return if Op::ALLOW_EMPTY {
            Ok(Op::EMPTY)
        } else {
            Err(LetoError::StorageError {
                reason: "all-elements reduction requires a non-empty input".to_string(),
            })
        };
    }

    if let Some(slice) = input.as_slice_memory_order() {
        let acc = reduce_nonempty_slice::<Op, T>(slice);
        return Ok(Op::finalize(acc, size));
    }

    let shape = input.shape();
    let layout = input.layout();
    let data = input.data();

    let mut acc: Option<T> = None;
    if let Some(traversal) = RowMajorTraversal::new(size, shape) {
        let step = traversal.last_axis_stride(layout);
        for row in 0..traversal.rows() {
            let base_idx = traversal.base_index(row);
            let mut offset = layout.offset_of(base_idx)? as isize;
            if let Some(slice) = unit_stride_row_slice(data, offset, step, traversal.inner()) {
                let row_acc = reduce_nonempty_slice::<Op, T>(slice);
                acc = Some(match acc {
                    Some(value) => Op::fold(value, row_acc),
                    None => row_acc,
                });
                continue;
            }

            for _ in 0..traversal.inner() {
                let value = data[offset as usize];
                acc = Some(match acc {
                    Some(current) => Op::fold(current, value),
                    None => Op::initial(value),
                });
                offset += step;
            }
        }
    }

    let Some(acc) = acc else {
        return Err(LetoError::StorageError {
            reason: "all-elements reduction traversal produced no elements".to_string(),
        });
    };
    Ok(Op::finalize(acc, size))
}

#[inline]
fn reduce_nonempty_slice<Op, T>(slice: &[T]) -> T
where
    Op: AxisReduction<T>,
    T: Scalar,
{
    if let Some(acc) = Op::reduce_slice(slice) {
        return acc;
    }

    let mut iter = slice.iter().copied();
    let first = iter
        .next()
        .expect("invariant: reduce_nonempty_slice receives a non-empty slice");
    let mut acc = Op::initial(first);
    for value in iter {
        acc = Op::fold(acc, value);
    }
    acc
}
