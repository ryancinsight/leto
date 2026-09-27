//! Keep-dim axis reduction: [`reduce_axis_into`] and [`reduce_axis`].

use super::strategies::AxisReduction;
use crate::application::index::{index_from_flat, validate_mutable_output};
use crate::domain::scalar::Scalar;
use leto::{Array, ArrayView, ArrayViewMut, Layout, LetoError, Result, VecStorage};

#[cfg(feature = "parallel")]
use crate::infrastructure::parallel::PARALLEL_MIN_REDUCTION_OUTPUTS;

#[cfg(feature = "parallel")]
struct AxisReductionContext<'a, T, const N: usize> {
    out_size: usize,
    out_shape: [usize; N],
    axis: usize,
    axis_len: usize,
    input_layout: leto::Layout<N>,
    output_layout: leto::Layout<N>,
    input_data: &'a [T],
    output_data: &'a mut [T],
}

#[inline]
fn output_shape<const N: usize>(input_shape: [usize; N], axis: usize) -> Result<[usize; N]> {
    if axis >= N {
        return Err(LetoError::StorageError {
            reason: format!("axis {axis} out of bounds for rank {N}"),
        });
    }

    let mut shape = input_shape;
    shape[axis] = 1;
    Ok(shape)
}

#[inline(always)]
fn reduce_nonempty_axis_values<Op, T, F>(
    axis_len: usize,
    contiguous: Option<&[T]>,
    mut value_at: F,
) -> T
where
    Op: AxisReduction<T>,
    T: Scalar,
    F: FnMut(usize) -> T,
{
    if let Some(slice) = contiguous {
        if let Some(acc) = Op::reduce_slice(slice) {
            return acc;
        }
    }

    let mut acc = Op::initial(value_at(0));
    for axis_idx in 1..axis_len {
        acc = Op::fold(acc, value_at(axis_idx));
    }
    acc
}

/// Apply a keep-dim axis reduction into caller-owned output storage.
pub fn reduce_axis_into<Op, T, const N: usize>(
    input: &ArrayView<'_, T, N>,
    axis: usize,
    output: &mut ArrayViewMut<'_, T, N>,
) -> Result<()>
where
    Op: AxisReduction<T>,
    T: Scalar,
{
    let expected_shape = output_shape(input.shape(), axis)?;
    if output.shape() != expected_shape {
        return Err(LetoError::ShapeMismatch {
            lhs: expected_shape.to_vec(),
            rhs: output.shape().to_vec(),
        });
    }

    input.layout().validate_storage_len(input.data().len())?;
    validate_mutable_output(output, "axis reduction")?;

    let axis_len = input.shape()[axis];
    if axis_len == 0 && !Op::ALLOW_EMPTY {
        return Err(LetoError::StorageError {
            reason: format!("axis {axis} has zero length for non-empty reduction"),
        });
    }

    let out_size = output.layout().checked_size()?;
    let out_shape = output.shape();
    let input_layout = input.layout();
    let output_layout = output.layout();
    let input_data = input.data();
    let output_data = output.data_mut();

    if N == 2
        && axis == 0
        && input_layout.strides()[1] == 1
        && input_layout.strides()[0] == input.shape()[1] as isize
        && output_layout.strides()[1] == 1
        && output_layout.strides()[0] == out_shape[1] as isize
    {
        let rows = input.shape()[0];
        let cols = input.shape()[1];
        let row_stride = input_layout.strides()[0];
        let col_stride = input_layout.strides()[1];
        let input_base = input_layout.offset();
        let output_base = output_layout.offset();
        let output_col_stride = output_layout.strides()[1];

        if rows == 0 {
            for col in 0..cols {
                let out_off = (output_base as isize + col as isize * output_col_stride) as usize;
                output_data[out_off] = Op::EMPTY;
            }
            return Ok(());
        }

        for col in 0..cols {
            let input_off = (input_base as isize + col as isize * col_stride) as usize;
            let out_off = (output_base as isize + col as isize * output_col_stride) as usize;
            output_data[out_off] = Op::initial(input_data[input_off]);
        }

        for row in 1..rows {
            let row_base = input_base as isize + row as isize * row_stride;
            for col in 0..cols {
                let input_off = (row_base + col as isize * col_stride) as usize;
                let out_off = (output_base as isize + col as isize * output_col_stride) as usize;
                output_data[out_off] = Op::fold(output_data[out_off], input_data[input_off]);
            }
        }

        for col in 0..cols {
            let out_off = (output_base as isize + col as isize * output_col_stride) as usize;
            output_data[out_off] = Op::finalize(output_data[out_off], rows);
        }
        return Ok(());
    }

    #[cfg(feature = "parallel")]
    {
        // Output injectivity is established by `validate_mutable_output`, so
        // parallel workers' logical rows map to disjoint physical elements.
        if out_size >= PARALLEL_MIN_REDUCTION_OUTPUTS {
            parallel_reduce_axis_into::<Op, T, N>(AxisReductionContext {
                out_size,
                out_shape,
                axis,
                axis_len,
                input_layout,
                output_layout,
                input_data,
                output_data,
            });
            return Ok(());
        }
    }

    let is_axis_contiguous = input_layout.strides()[axis] == 1;
    let axis_stride = input_layout.strides()[axis];

    for flat_idx in 0..out_size {
        let out_idx = index_from_flat(flat_idx, &out_shape);
        let out_off = output_layout.offset_of(out_idx)?;
        if axis_len == 0 {
            output_data[out_off] = Op::EMPTY;
            continue;
        }

        let mut input_idx = out_idx;
        input_idx[axis] = 0;
        let first_off = input_layout.offset_of(input_idx)?;

        let acc = reduce_nonempty_axis_values::<Op, T, _>(
            axis_len,
            if is_axis_contiguous {
                Some(&input_data[first_off..first_off + axis_len])
            } else {
                None
            },
            |axis_idx| {
                let input_off = (first_off as isize + axis_idx as isize * axis_stride) as usize;
                input_data[input_off]
            },
        );

        output_data[out_off] = Op::finalize(acc, axis_len);
    }

    Ok(())
}

/// Apply a keep-dim axis reduction into newly allocated C-contiguous output storage.
pub fn reduce_axis<Op, T, const N: usize>(
    input: &ArrayView<'_, T, N>,
    axis: usize,
) -> Result<Array<T, VecStorage<T>, N>>
where
    Op: AxisReduction<T>,
    T: Scalar,
{
    let shape = output_shape(input.shape(), axis)?;
    let layout = Layout::c_contiguous(shape)?;
    let size = layout.checked_size()?;
    let storage = VecStorage::uninit(size);
    let mut output = Array::new(layout, storage)?;
    reduce_axis_into::<Op, T, N>(input, axis, &mut output.view_mut())?;
    Ok(output)
}

#[cfg(feature = "parallel")]
fn parallel_reduce_axis_into<Op, T, const N: usize>(ctx: AxisReductionContext<'_, T, N>)
where
    Op: AxisReduction<T>,
    T: Scalar,
{
    let input_ptr = ctx.input_data.as_ptr() as usize;
    let output_ptr = ctx.output_data.as_mut_ptr() as usize;
    let axis_stride = ctx.input_layout.strides()[ctx.axis];
    // One unit is one output element: it reads the whole reduced axis and
    // writes once.
    let unit_bytes = ctx
        .axis_len
        .saturating_add(1)
        .saturating_mul(core::mem::size_of::<T>());

    crate::infrastructure::parallel::for_each_unit_range(
        ctx.out_size,
        unit_bytes,
        move |first, count| {
            let is_axis_contiguous = axis_stride == 1;
            for flat_idx in first..first + count {
                let out_idx = index_from_flat(flat_idx, &ctx.out_shape);
                let out_off = ctx
                    .output_layout
                    .offset_of(out_idx)
                    .expect("validated output layout must map every logical index");
                if ctx.axis_len == 0 {
                    // SAFETY: each worker writes a distinct logical output element.
                    unsafe {
                        *(output_ptr as *mut T).add(out_off) = Op::EMPTY;
                    }
                    continue;
                }

                let mut input_idx = out_idx;
                input_idx[ctx.axis] = 0;
                let first_off = ctx
                    .input_layout
                    .offset_of(input_idx)
                    .expect("validated input layout must map every logical index");

                let acc = reduce_nonempty_axis_values::<Op, T, _>(
                    ctx.axis_len,
                    if is_axis_contiguous {
                        // SAFETY: input slice bounds are validated.
                        Some(unsafe {
                            std::slice::from_raw_parts(
                                (input_ptr as *const T).add(first_off),
                                ctx.axis_len,
                            )
                        })
                    } else {
                        None
                    },
                    |axis_idx| unsafe {
                        let input_off =
                            (first_off as isize + axis_idx as isize * axis_stride) as usize;
                        *(input_ptr as *const T).add(input_off)
                    },
                );

                // SAFETY: each worker writes a distinct logical output element.
                unsafe {
                    *(output_ptr as *mut T).add(out_off) = Op::finalize(acc, ctx.axis_len);
                }
            }
        },
    );
}
