use crate::application::index::index_from_flat;
use leto::{Array, ArrayView, ArrayViewMut, LetoError, Result, VecStorage};

#[cfg(feature = "parallel")]
const PARALLEL_THRESHOLD: usize = 8192;

#[cfg(feature = "parallel")]
struct StridedMapContext<'a, T, U, const N: usize> {
    size: usize,
    shape: [usize; N],
    input_layout: leto::Layout<N>,
    output_layout: leto::Layout<N>,
    input_data: &'a [T],
    output_data: &'a mut [U],
}

#[inline]
fn validate_unary_storage<T, U, const N: usize>(
    input: &ArrayView<'_, T, N>,
    output: &ArrayViewMut<'_, U, N>,
) -> Result<()> {
    input.layout().validate_storage_len(input.data().len())?;
    output.layout().validate_storage_len(output.data().len())?;
    if output.layout().has_zero_stride_aliasing() {
        return Err(LetoError::StorageError {
            reason: "map output layout must not contain zero-stride aliasing".to_string(),
        });
    }
    Ok(())
}

/// Map every element from `input` into caller-owned `output`.
///
/// The output shape must match the input shape. The closure executes on values
/// in the input scalar type and returns the output scalar type; precision
/// changes are therefore explicit at the call site.
pub fn map_into<T, U, F, const N: usize>(
    input: &ArrayView<'_, T, N>,
    output: &mut ArrayViewMut<'_, U, N>,
    f: F,
) -> Result<()>
where
    T: Copy + Send + Sync + 'static,
    U: Copy + Send + Sync + 'static,
    F: Fn(T) -> U + Copy + Send + Sync + 'static,
{
    if input.shape() != output.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: input.shape().to_vec(),
            rhs: output.shape().to_vec(),
        });
    }

    if let (Some(input_slice), Some(output_slice)) = (input.as_slice(), output.as_mut_slice()) {
        #[cfg(feature = "parallel")]
        {
            if input_slice.len() >= PARALLEL_THRESHOLD {
                parallel_map_slice(input_slice, output_slice, f);
                return Ok(());
            }
        }

        for (out, &value) in output_slice.iter_mut().zip(input_slice.iter()) {
            *out = f(value);
        }
        return Ok(());
    }

    validate_unary_storage(input, output)?;
    let size = input.layout().checked_size()?;
    let shape = input.shape();
    let input_layout = input.layout();
    let output_layout = output.layout();
    let input_data = input.data();
    let output_data = output.data_mut();

    #[cfg(feature = "parallel")]
    {
        if size >= PARALLEL_THRESHOLD {
            parallel_map_strided(
                StridedMapContext {
                    size,
                    shape,
                    input_layout,
                    output_layout,
                    input_data,
                    output_data,
                },
                f,
            );
            return Ok(());
        }
    }

    for flat_idx in 0..size {
        let index = index_from_flat(flat_idx, &shape);
        let input_offset = input_layout.offset_of(index)?;
        let output_offset = output_layout.offset_of(index)?;
        output_data[output_offset] = f(input_data[input_offset]);
    }

    Ok(())
}

/// Allocate a C-contiguous output array and map every input element into it.
pub fn mapv<T, U, F, const N: usize>(
    input: &ArrayView<'_, T, N>,
    f: F,
) -> Result<Array<U, VecStorage<U>, N>>
where
    T: Copy + Send + Sync + 'static,
    U: Copy + Send + Sync + 'static,
    F: Fn(T) -> U + Copy + Send + Sync + 'static,
{
    input.layout().validate_storage_len(input.data().len())?;
    let size = input.layout().checked_size()?;
    let shape = input.shape();
    let layout = leto::Layout::c_contiguous(shape)?;
    let mut values = Vec::with_capacity(size);

    if let Some(input_slice) = input.as_slice() {
        values.extend(input_slice.iter().copied().map(f));
    } else {
        let input_layout = input.layout();
        let input_data = input.data();
        for flat_idx in 0..size {
            let index = index_from_flat(flat_idx, &shape);
            let input_offset = input_layout.offset_of(index)?;
            values.push(f(input_data[input_offset]));
        }
    }

    Array::new(layout, VecStorage::new(values))
}

/// Alias for `mapv` matching ndarray's borrowed-value naming.
#[inline]
pub fn map<T, U, F, const N: usize>(
    input: &ArrayView<'_, T, N>,
    f: F,
) -> Result<Array<U, VecStorage<U>, N>>
where
    T: Copy + Send + Sync + 'static,
    U: Copy + Send + Sync + 'static,
    F: Fn(T) -> U + Copy + Send + Sync + 'static,
{
    mapv(input, f)
}

#[cfg(feature = "parallel")]
fn parallel_map_slice<T, U, F>(input: &[T], output: &mut [U], f: F)
where
    T: Copy + Send + Sync + 'static,
    U: Copy + Send + Sync + 'static,
    F: Fn(T) -> U + Copy + Send + Sync + 'static,
{
    let len = input.len();
    let input_ptr = input.as_ptr() as usize;
    let output_ptr = output.as_mut_ptr() as usize;
    let chunk_size = 4096;

    crate::infrastructure::parallel::parallel_for_chunks(len, chunk_size, move |start, end| {
        for index in start..end {
            // SAFETY: each worker writes a unique element in `start..end`; input and
            // output slices have equal length by `map_into` shape validation.
            unsafe {
                let value = *(input_ptr as *const T).add(index);
                *(output_ptr as *mut U).add(index) = f(value);
            }
        }
    });
}

#[cfg(feature = "parallel")]
fn parallel_map_strided<T, U, F, const N: usize>(ctx: StridedMapContext<'_, T, U, N>, f: F)
where
    T: Copy + Send + Sync + 'static,
    U: Copy + Send + Sync + 'static,
    F: Fn(T) -> U + Copy + Send + Sync + 'static,
{
    let input_ptr = ctx.input_data.as_ptr() as usize;
    let output_ptr = ctx.output_data.as_mut_ptr() as usize;
    let chunk_size = 512;

    crate::infrastructure::parallel::parallel_for_chunks(
        ctx.size,
        chunk_size,
        move |start, end| {
            for flat_idx in start..end {
                let index = index_from_flat(flat_idx, &ctx.shape);
                let input_offset = ctx
                    .input_layout
                    .offset_of(index)
                    .expect("validated input layout must map every logical index");
                let output_offset = ctx
                    .output_layout
                    .offset_of(index)
                    .expect("validated output layout must map every logical index");

                // SAFETY: storage spans are validated before dispatch and zero-stride
                // output aliasing is rejected before this path.
                unsafe {
                    let value = *(input_ptr as *const T).add(input_offset);
                    *(output_ptr as *mut U).add(output_offset) = f(value);
                }
            }
        },
    );
}
