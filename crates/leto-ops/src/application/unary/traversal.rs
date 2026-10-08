use crate::application::index::{line_elements_for, validate_mutable_output, RowMajorTraversal};
use crate::infrastructure::cache::{cached_cache_geometry, CacheGeometry};
#[cfg(feature = "parallel")]
use crate::infrastructure::parallel::{parallelize_bandwidth_bound, parallelize_compute_bound};
use leto::{Array, ArrayView, ArrayViewMut, LetoError, Result, VecStorage};

#[inline]
fn validate_unary_storage<T, U, const N: usize>(
    input: &ArrayView<'_, T, N>,
    output: &ArrayViewMut<'_, U, N>,
) -> Result<()> {
    input.layout().validate_storage_len(input.data().len())?;
    validate_mutable_output(output, "map")?;
    Ok(())
}

/// Whether a map over `len` elements of `T` should run in parallel.
///
/// Compute-bound ops (heavy per-element arithmetic) pay off past a small fixed
/// element count; bandwidth-bound ops parallelize only once their working set
/// (input + output) spills past the shared last-level cache, mirroring
/// `binary_map`. Both gates live in the single `infrastructure::parallel`
/// policy.
#[cfg(feature = "parallel")]
fn should_parallelize<T>(len: usize, compute_bound: bool) -> bool {
    if compute_bound {
        parallelize_compute_bound(len)
    } else {
        parallelize_bandwidth_bound::<T>(len, 2)
    }
}

/// Apply `f` elementwise into caller-owned output. Raw closures are treated as
/// compute-bound (the historical eager-parallel default, preserved so existing
/// callers do not regress); the typed [`unary_map_into`](crate::application::unary::unary_map_into) passes the op's
/// [`UnaryOp::COMPUTE_BOUND`](crate::application::unary::UnaryOp::COMPUTE_BOUND) instead.
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
    map_into_gated(input, output, f, true)
}

/// Apply `f` elementwise using an explicit cache geometry policy.
///
/// The geometry's cache-line width selects the micro-tile side for strided
/// views. This is intended for callers with topology information that is more
/// authoritative than the process-local [`cached_cache_geometry`] probe; the
/// default [`map_into`] path remains automatically detected. Cache capacities
/// in `geometry` do not affect this operation.
pub fn map_into_with_cache_geometry<T, U, F, const N: usize>(
    input: &ArrayView<'_, T, N>,
    output: &mut ArrayViewMut<'_, U, N>,
    f: F,
    geometry: CacheGeometry,
) -> Result<()>
where
    T: Copy + Send + Sync + 'static,
    U: Copy + Send + Sync + 'static,
    F: Fn(T) -> U + Copy + Send + Sync + 'static,
{
    map_into_gated_with_cache_line(input, output, f, true, Some(geometry.cache_line_bytes()))
}

/// [`map_into`] with an explicit compute-bound flag driving the parallel gate.
pub(crate) fn map_into_gated<T, U, F, const N: usize>(
    input: &ArrayView<'_, T, N>,
    output: &mut ArrayViewMut<'_, U, N>,
    f: F,
    compute_bound: bool,
) -> Result<()>
where
    T: Copy + Send + Sync + 'static,
    U: Copy + Send + Sync + 'static,
    F: Fn(T) -> U + Copy + Send + Sync + 'static,
{
    map_into_gated_with_cache_line(input, output, f, compute_bound, None)
}

fn map_into_gated_with_cache_line<T, U, F, const N: usize>(
    input: &ArrayView<'_, T, N>,
    output: &mut ArrayViewMut<'_, U, N>,
    f: F,
    compute_bound: bool,
    cache_line_bytes: Option<usize>,
) -> Result<()>
where
    T: Copy + Send + Sync + 'static,
    U: Copy + Send + Sync + 'static,
    F: Fn(T) -> U + Copy + Send + Sync + 'static,
{
    #[cfg(not(feature = "parallel"))]
    let _ = compute_bound;
    if input.shape() != output.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: input.shape().to_vec(),
            rhs: output.shape().to_vec(),
        });
    }

    // Identical strides make both dense memory-order blocks enumerate the
    // same logical element at each position, so any shared dense order (C, F,
    // or permuted-contiguous) feeds the slice kernel — not only canonical C.
    if input.strides() == output.strides() {
        if let (Some(input_slice), Some(output_slice)) = (
            input.as_slice_memory_order(),
            output.as_mut_slice_memory_order(),
        ) {
            #[cfg(feature = "parallel")]
            {
                if should_parallelize::<T>(input_slice.len(), compute_bound) {
                    crate::application::strided::parallel_slice_into::<T, U, 1, _>(
                        [input_slice],
                        output_slice,
                        core::mem::size_of::<T>() + core::mem::size_of::<U>(),
                        |inputs, out| {
                            for (cell, &value) in out.iter_mut().zip(inputs[0]) {
                                *cell = f(value);
                            }
                        },
                    );
                    return Ok(());
                }
            }

            for (out, &value) in output_slice.iter_mut().zip(input_slice.iter()) {
                *out = f(value);
            }
            return Ok(());
        }
    }

    validate_unary_storage(input, output)?;
    let size = input.layout().checked_size()?;
    let shape = input.shape();
    let input_layout = input.layout();
    let output_layout = output.layout();
    let input_data = input.data();
    let output_data = output.data_mut();
    let cache_line_bytes =
        cache_line_bytes.unwrap_or_else(|| cached_cache_geometry().cache_line_bytes());
    let input_tile = line_elements_for::<T>(cache_line_bytes);
    let output_tile = line_elements_for::<U>(cache_line_bytes);

    #[cfg(feature = "parallel")]
    {
        if should_parallelize::<T>(size, compute_bound) {
            let input_ptr = input_data.as_ptr() as usize;
            let output_ptr = output_data.as_mut_ptr() as usize;
            crate::application::strided::strided_parallel::<N, 1, 1, _>(
                crate::application::strided::StridedLayout::new(
                    size,
                    shape,
                    [input_layout],
                    [output_layout],
                    input_tile,
                    output_tile,
                ),
                core::mem::size_of::<T>() + core::mem::size_of::<U>(),
                move |in_off: [isize; 1], out_off: [isize; 1]| {
                    // SAFETY: storage spans are validated before dispatch; every
                    // walked offset equals offset_of of a validated logical
                    // index; each worker owns disjoint rows and the output
                    // layout has no zero-stride aliasing, so no two workers
                    // write one element.
                    unsafe {
                        let value = *(input_ptr as *const T).offset(in_off[0]);
                        *(output_ptr as *mut U).offset(out_off[0]) = f(value);
                    }
                },
            );
            return Ok(());
        }
    }

    // Row-walk traversal: one offset computation per innermost row, then a
    // stride-increment walk. Column-walk views use cache-line micro-tiles
    // through the same shared `application::strided` traversal as binary_map.
    crate::application::strided::strided_serial::<N, 1, 1, _>(
        crate::application::strided::StridedLayout::new(
            size,
            shape,
            [input_layout],
            [output_layout],
            input_tile,
            output_tile,
        ),
        |in_off: [isize; 1], out_off: [isize; 1]| {
            output_data[out_off[0] as usize] = f(input_data[in_off[0] as usize]);
        },
    )
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
    } else if size > 0 {
        let input_layout = input.layout();
        let input_data = input.data();
        // Row-walk read traversal (output is push-sequential by construction).
        if let Some(traversal) = RowMajorTraversal::new(size, shape) {
            let in_step = traversal.last_axis_stride(input_layout);
            for row in 0..traversal.rows() {
                let base_idx = traversal.base_index(row);
                let mut input_offset = input_layout.offset_of(base_idx)? as isize;
                for _ in 0..traversal.inner() {
                    values.push(f(input_data[input_offset as usize]));
                    input_offset += in_step;
                }
            }
        }
    }

    Array::new(layout, VecStorage::new(values))
}

/// Alias for `mapv` matching leto's borrowed-value naming.
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

/// Apply `f` to every element of `view` in place.
///
/// This is the `leto::mapv_inplace` analogue. Elementwise in-place mutation
/// is memory-order independent, so the contiguous fast path accepts any dense
/// block (C or F). Zero-stride write aliasing is rejected because it would
/// apply `f` to a single physical element more than once.
///
/// # Parallel dispatch
///
/// A raw closure's arithmetic intensity is not knowable, so `f` is treated as
/// compute-bound and parallelized past a fixed element count, matching
/// [`map_into`]. There is no in-place counterpart to [`unary_map_into`](crate::application::unary::unary_map_into), so a
/// bandwidth-bound op cannot currently reach the cache-residency gate that
/// [`UnaryOp::COMPUTE_BOUND`](crate::application::unary::UnaryOp::COMPUTE_BOUND) selects for the into-output form.
///
/// This costs a bandwidth-bound closure real time while the data is still
/// cache-resident. Measured on a 36 MiB-L3 AVX2 host against an identical
/// sequential loop over the same slice and closure, `|x| x * c` runs
/// **5.9-6.4x slower** for `f64` and **8.5-9.8x slower** for `f32` from 65536
/// elements through roughly 512 KiB, and only becomes profitable past about
/// 1M elements. Prefer [`unary_map_into`](crate::application::unary::unary_map_into) with a `COMPUTE_BOUND = false` op
/// when the destination may differ, or keep such a map sequential by slicing
/// below the gate. Tracked as `LETO-INPLACE-INTENSITY-GATE-2026-09-01`.
pub fn map_inplace<T, F, const N: usize>(view: &mut ArrayViewMut<'_, T, N>, f: F) -> Result<()>
where
    T: Copy + Send + Sync + 'static,
    F: Fn(T) -> T + Copy + Send + Sync + 'static,
{
    validate_mutable_output(view, "in-place map")?;

    if let Some(slice) = view.as_mut_slice_memory_order() {
        #[cfg(feature = "parallel")]
        {
            if parallelize_compute_bound(slice.len()) {
                parallel_map_inplace_slice(slice, f);
                return Ok(());
            }
        }

        for value in slice.iter_mut() {
            *value = f(*value);
        }
        return Ok(());
    }

    // Row-walk traversal (shared RowMajorTraversal policy; see binary_map).
    let size = view.layout().checked_size()?;
    let shape = view.shape();
    let layout = view.layout();
    let data = view.data_mut();
    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(());
    };
    let step = traversal.last_axis_stride(layout);
    for row in 0..traversal.rows() {
        let base = traversal.base_index(row);
        let mut offset = layout.offset_of(base)? as isize;
        for _ in 0..traversal.inner() {
            data[offset as usize] = f(data[offset as usize]);
            offset += step;
        }
    }

    Ok(())
}

#[cfg(feature = "parallel")]
fn parallel_map_inplace_slice<T, F>(slice: &mut [T], f: F)
where
    T: Copy + Send + Sync + 'static,
    F: Fn(T) -> T + Copy + Send + Sync + 'static,
{
    // One unit is read and written in place, so it moves twice its own size.
    crate::infrastructure::parallel::for_each_unit_run_mut(
        slice,
        2 * core::mem::size_of::<T>(),
        |_, run| {
            for cell in run {
                *cell = f(*cell);
            }
        },
    );
}
