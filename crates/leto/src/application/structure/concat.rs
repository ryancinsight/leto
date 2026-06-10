use crate::application::array::Array;
use crate::application::index::index_from_flat;
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
    let mut values = Vec::with_capacity(size);

    for flat in 0..size {
        let mut index = index_from_flat(flat, &out_shape);
        let along = index[axis];
        // Locate the source input owning logical position `along` on `axis`.
        let mut base = 0usize;
        let mut chosen: Option<&ArrayView<'_, T, N>> = None;
        let mut local = along;
        for view in inputs {
            let len = view.shape()[axis];
            if along < base + len {
                local = along - base;
                chosen = Some(view);
                break;
            }
            base += len;
        }
        let view = chosen.expect("axis position is covered by an input");
        index[axis] = local;
        values.push(view.get(index)?.clone());
    }

    Array::new(out_layout, VecStorage::new(values))
}
