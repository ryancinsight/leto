use crate::application::index::{index_from_flat, unit_stride_row_slice, RowMajorTraversal};
use crate::domain::scalar::Scalar;
use leto::{Array, ArrayView, ArrayViewMut, Layout, LetoError, Result, VecStorage};

#[cfg(feature = "parallel")]
// Reductions benefit from parallelism at lower element counts than unary ops
// because each output element requires a full axis scan (O(N/out_size) reads).
const PARALLEL_THRESHOLD: usize = 32768;

mod sealed {
    pub trait Sealed {}
}

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

/// Zero-sized axis-reduction contract.
pub trait AxisReduction<T: Scalar>: sealed::Sealed + Copy + Send + Sync + 'static {
    /// Initial accumulator for non-empty reductions.
    fn initial(first: T) -> T;
    /// Fold one value into the accumulator.
    fn fold(acc: T, value: T) -> T;
    /// Finalize the accumulator after `axis_len` elements.
    fn finalize(acc: T, axis_len: usize) -> T;
    /// Whether an empty reduction has a defined value.
    const ALLOW_EMPTY: bool;
    /// Empty reduction value when `ALLOW_EMPTY` is true.
    const EMPTY: T;
    /// Try a fast-path slice-based reduction.
    fn reduce_slice(slice: &[T]) -> Option<T>;
}

/// Sum axis-reduction marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SumAxis;

/// Mean axis-reduction marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeanAxis;

/// Minimum axis-reduction marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct MinAxis;

/// Maximum axis-reduction marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaxAxis;

impl sealed::Sealed for SumAxis {}
impl sealed::Sealed for MeanAxis {}
impl sealed::Sealed for MinAxis {}
impl sealed::Sealed for MaxAxis {}

impl<T: Scalar> AxisReduction<T> for SumAxis {
    #[inline(always)]
    fn initial(first: T) -> T {
        first
    }

    #[inline(always)]
    fn fold(acc: T, value: T) -> T {
        acc.add(value)
    }

    #[inline(always)]
    fn finalize(acc: T, _axis_len: usize) -> T {
        acc
    }

    const ALLOW_EMPTY: bool = true;
    const EMPTY: T = T::ZERO;

    #[inline(always)]
    fn reduce_slice(slice: &[T]) -> Option<T> {
        Some(T::sum_slice(slice))
    }
}

impl<T: Scalar> AxisReduction<T> for MeanAxis {
    #[inline(always)]
    fn initial(first: T) -> T {
        first
    }

    #[inline(always)]
    fn fold(acc: T, value: T) -> T {
        acc.add(value)
    }

    #[inline(always)]
    fn finalize(acc: T, axis_len: usize) -> T {
        acc.div(T::from_usize(axis_len))
    }

    const ALLOW_EMPTY: bool = false;
    const EMPTY: T = T::ZERO;

    #[inline(always)]
    fn reduce_slice(slice: &[T]) -> Option<T> {
        Some(T::sum_slice(slice))
    }
}

impl<T: Scalar> AxisReduction<T> for MinAxis {
    #[inline(always)]
    fn initial(first: T) -> T {
        first
    }

    #[inline(always)]
    fn fold(acc: T, value: T) -> T {
        if value < acc {
            value
        } else {
            acc
        }
    }

    #[inline(always)]
    fn finalize(acc: T, _axis_len: usize) -> T {
        acc
    }

    const ALLOW_EMPTY: bool = false;
    const EMPTY: T = T::ZERO;

    #[inline(always)]
    fn reduce_slice(slice: &[T]) -> Option<T> {
        Some(T::min_slice(slice))
    }
}

impl<T: Scalar> AxisReduction<T> for MaxAxis {
    #[inline(always)]
    fn initial(first: T) -> T {
        first
    }

    #[inline(always)]
    fn fold(acc: T, value: T) -> T {
        if value > acc {
            value
        } else {
            acc
        }
    }

    #[inline(always)]
    fn finalize(acc: T, _axis_len: usize) -> T {
        acc
    }

    const ALLOW_EMPTY: bool = false;
    const EMPTY: T = T::ZERO;

    #[inline(always)]
    fn reduce_slice(slice: &[T]) -> Option<T> {
        Some(T::max_slice(slice))
    }
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

/// Reduce every element of `input` into one scalar using an [`AxisReduction`]
/// marker.
///
/// The operation validates the input view storage, rejects empty inputs for
/// reductions without an identity (for example [`MinAxis`] and [`MaxAxis`]),
/// and follows logical view order for strided and sliced layouts without
/// materializing a contiguous copy.
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
    output.layout().validate_storage_len(output.data().len())?;

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
        && input_layout.strides[1] == 1
        && input_layout.strides[0] == input.shape()[1] as isize
        && output_layout.strides[1] == 1
        && output_layout.strides[0] == out_shape[1] as isize
        && !output_layout.has_zero_stride_aliasing()
    {
        let rows = input.shape()[0];
        let cols = input.shape()[1];
        let row_stride = input_layout.strides[0];
        let col_stride = input_layout.strides[1];
        let input_base = input_layout.offset;
        let output_base = output_layout.offset;
        let output_col_stride = output_layout.strides[1];

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
        if out_size >= PARALLEL_THRESHOLD && !output_layout.has_zero_stride_aliasing() {
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

    let is_axis_contiguous = input_layout.strides[axis] == 1;

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

        let acc = if is_axis_contiguous {
            if let Some(slice_res) = Op::reduce_slice(&input_data[first_off..first_off + axis_len])
            {
                slice_res
            } else {
                let mut a = Op::initial(input_data[first_off]);
                for axis_idx in 1..axis_len {
                    let input_off = (first_off as isize + axis_idx as isize) as usize;
                    a = Op::fold(a, input_data[input_off]);
                }
                a
            }
        } else {
            let mut a = Op::initial(input_data[first_off]);
            let axis_stride = input_layout.strides[axis];
            for axis_idx in 1..axis_len {
                let input_off = (first_off as isize + axis_idx as isize * axis_stride) as usize;
                a = Op::fold(a, input_data[input_off]);
            }
            a
        };

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
    let storage = VecStorage::fill(size, T::ZERO);
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
    let chunk_size = 512;

    crate::infrastructure::parallel::parallel_for_chunks(
        ctx.out_size,
        chunk_size,
        move |start, end| {
            let is_axis_contiguous = ctx.input_layout.strides[ctx.axis] == 1;
            for flat_idx in start..end {
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

                let acc = if is_axis_contiguous {
                    // SAFETY: input slice bounds are validated.
                    let slice = unsafe {
                        std::slice::from_raw_parts(
                            (input_ptr as *const T).add(first_off),
                            ctx.axis_len,
                        )
                    };
                    if let Some(slice_res) = Op::reduce_slice(slice) {
                        slice_res
                    } else {
                        let mut a = unsafe { Op::initial(*(input_ptr as *const T).add(first_off)) };
                        for axis_idx in 1..ctx.axis_len {
                            let input_off = (first_off as isize + axis_idx as isize) as usize;
                            let value = unsafe { *(input_ptr as *const T).add(input_off) };
                            a = Op::fold(a, value);
                        }
                        a
                    }
                } else {
                    let mut a = unsafe { Op::initial(*(input_ptr as *const T).add(first_off)) };
                    let axis_stride = ctx.input_layout.strides[ctx.axis];
                    for axis_idx in 1..ctx.axis_len {
                        let input_off =
                            (first_off as isize + axis_idx as isize * axis_stride) as usize;
                        let value = unsafe { *(input_ptr as *const T).add(input_off) };
                        a = Op::fold(a, value);
                    }
                    a
                };

                // SAFETY: each worker writes a distinct logical output element.
                unsafe {
                    *(output_ptr as *mut T).add(out_off) = Op::finalize(acc, ctx.axis_len);
                }
            }
        },
    );
}

/// Sum `input` along `axis`, keeping the reduced axis as length one.
#[inline]
pub fn sum_axis_into<T: Scalar, const N: usize>(
    input: &ArrayView<'_, T, N>,
    axis: usize,
    output: &mut ArrayViewMut<'_, T, N>,
) -> Result<()> {
    reduce_axis_into::<SumAxis, T, N>(input, axis, output)
}

/// Sum `input` along `axis` into newly allocated C-contiguous output storage.
#[inline]
pub fn sum_axis<T: Scalar, const N: usize>(
    input: &ArrayView<'_, T, N>,
    axis: usize,
) -> Result<Array<T, VecStorage<T>, N>> {
    reduce_axis::<SumAxis, T, N>(input, axis)
}

/// Mean-reduce `input` along `axis`, keeping the reduced axis as length one.
#[inline]
pub fn mean_axis_into<T: Scalar, const N: usize>(
    input: &ArrayView<'_, T, N>,
    axis: usize,
    output: &mut ArrayViewMut<'_, T, N>,
) -> Result<()> {
    reduce_axis_into::<MeanAxis, T, N>(input, axis, output)
}

/// Mean-reduce `input` along `axis` into newly allocated C-contiguous output storage.
#[inline]
pub fn mean_axis<T: Scalar, const N: usize>(
    input: &ArrayView<'_, T, N>,
    axis: usize,
) -> Result<Array<T, VecStorage<T>, N>> {
    reduce_axis::<MeanAxis, T, N>(input, axis)
}

/// Min-reduce `input` along `axis`, keeping the reduced axis as length one.
#[inline]
pub fn min_axis_into<T: Scalar, const N: usize>(
    input: &ArrayView<'_, T, N>,
    axis: usize,
    output: &mut ArrayViewMut<'_, T, N>,
) -> Result<()> {
    reduce_axis_into::<MinAxis, T, N>(input, axis, output)
}

/// Min-reduce `input` along `axis` into newly allocated C-contiguous output storage.
#[inline]
pub fn min_axis<T: Scalar, const N: usize>(
    input: &ArrayView<'_, T, N>,
    axis: usize,
) -> Result<Array<T, VecStorage<T>, N>> {
    reduce_axis::<MinAxis, T, N>(input, axis)
}

/// Max-reduce `input` along `axis`, keeping the reduced axis as length one.
#[inline]
pub fn max_axis_into<T: Scalar, const N: usize>(
    input: &ArrayView<'_, T, N>,
    axis: usize,
    output: &mut ArrayViewMut<'_, T, N>,
) -> Result<()> {
    reduce_axis_into::<MaxAxis, T, N>(input, axis, output)
}

/// Max-reduce `input` along `axis` into newly allocated C-contiguous output storage.
#[inline]
pub fn max_axis<T: Scalar, const N: usize>(
    input: &ArrayView<'_, T, N>,
    axis: usize,
) -> Result<Array<T, VecStorage<T>, N>> {
    reduce_axis::<MaxAxis, T, N>(input, axis)
}

/// Minimum over all elements of `input`.
#[inline]
pub fn min<T: Scalar, const N: usize>(input: &ArrayView<'_, T, N>) -> Result<T> {
    reduce_all::<MinAxis, T, N>(input)
}

/// Maximum over all elements of `input`.
#[inline]
pub fn max<T: Scalar, const N: usize>(input: &ArrayView<'_, T, N>) -> Result<T> {
    reduce_all::<MaxAxis, T, N>(input)
}
