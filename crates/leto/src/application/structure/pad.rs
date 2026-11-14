use crate::application::array::Array;
use crate::application::constructors::increment_index;
use crate::application::view::{ArrayView, ArrayViewMut};
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;
use crate::infrastructure::storage::VecStorage;

/// Per-axis `(before, after)` padding widths.
pub type PadWidth<const N: usize> = [(usize, usize); N];

/// The padded shape of `in_shape` under `width`, with checked arithmetic.
///
/// # Errors
///
/// Returns [`LetoError::Overflow`] when `before + extent + after` overflows
/// a `usize` on any axis, instead of panicking or wrapping.
pub fn padded_shape<const N: usize>(
    in_shape: [usize; N],
    width: PadWidth<N>,
) -> Result<[usize; N]> {
    let mut out_shape = [0usize; N];
    for d in 0..N {
        out_shape[d] = width[d]
            .0
            .checked_add(in_shape[d])
            .and_then(|v| v.checked_add(width[d].1))
            .ok_or(LetoError::Overflow {
                reason: "pad width overflows axis extent",
            })?;
    }
    Ok(out_shape)
}

/// Pad `input` with `fill` by `width` elements before and after on each axis,
/// writing into caller-owned `output`.
///
/// Output dimension `d` must equal `before[d] + input[d] + after[d]`. Cells
/// inside the original region copy the source; cells in the pad margins take
/// `fill`. Both views may be strided: transposed, sliced, and offset views
/// dispatch without a contiguous staging copy, mirroring
/// `hephaestus_core::PadOps`.
///
/// # Errors
///
/// Returns [`LetoError::Overflow`] when the padded shape overflows, or
/// [`LetoError::ShapeMismatch`] when `output`'s shape does not match it.
pub fn pad_into<T: Clone, const N: usize>(
    input: &ArrayView<'_, T, N>,
    width: PadWidth<N>,
    fill: T,
    output: &mut ArrayViewMut<'_, T, N>,
) -> Result<()> {
    let in_shape = input.shape();
    let expected = padded_shape(in_shape, width)?;
    if output.shape() != expected {
        return Err(LetoError::ShapeMismatch {
            lhs: expected.to_vec(),
            rhs: output.shape().to_vec(),
        });
    }

    for (out_index, slot) in output.reborrow().indexed_iter_mut()? {
        *slot = padded_value(input, width, out_index, &fill)?;
    }
    Ok(())
}

/// The padded element at `out_index`: the source cell inside the original
/// region, `fill` in the margins.
///
/// One selection rule for [`pad_into`] and [`pad`], so the caller-owned and
/// allocating paths cannot diverge.
fn padded_value<T: Clone, const N: usize>(
    input: &ArrayView<'_, T, N>,
    width: PadWidth<N>,
    out_index: [usize; N],
    fill: &T,
) -> Result<T> {
    let in_shape = input.shape();
    let mut src_index = [0usize; N];
    for d in 0..N {
        let Some(s) = out_index[d]
            .checked_sub(width[d].0)
            .filter(|&s| s < in_shape[d])
        else {
            return Ok(fill.clone());
        };
        src_index[d] = s;
    }
    Ok(input.get(src_index)?.clone())
}

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
    let out_shape = padded_shape(input.shape(), width)?;
    let out_layout = Layout::c_contiguous(out_shape)?;
    let size = out_layout.size();
    // Pushed one value at a time in row-major order: the vector length is the
    // live prefix, so an error drops exactly the elements already built.
    let mut data = Vec::with_capacity(size);
    let mut out_index = [0usize; N];
    for _ in 0..size {
        data.push(padded_value(input, width, out_index, &fill)?);
        increment_index(&mut out_index, &out_shape);
    }
    Array::new(out_layout, VecStorage::new(data))
}
