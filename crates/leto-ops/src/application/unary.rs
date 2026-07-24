use crate::application::index::{line_elements, RowMajorTraversal, TileGeometry};
use crate::domain::RealScalar;
use leto::{Array, ArrayView, ArrayViewMut, LetoError, Result, VecStorage};

// Parallel execution only pays off when the working set exceeds L2 cache
// (~256 KB per core on modern x86). Below this threshold the thread-pool
// overhead dominates; above it NUMA-aware parallel execution gains bandwidth.
// 256 KB / 4 bytes (f32) = 65536 elements.
#[cfg(feature = "parallel")]
const PARALLEL_THRESHOLD: usize = 65536;

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
/// Whether a map over `len` elements of `T` should run in parallel. Compute-bound
/// ops (heavy per-element arithmetic) pay off past a small fixed element count;
/// bandwidth-bound ops parallelize only once their working set (input + output)
/// spills past the shared last-level cache, mirroring `binary_map`.
#[cfg(feature = "parallel")]
fn should_parallelize<T>(len: usize, compute_bound: bool) -> bool {
    if compute_bound {
        len >= PARALLEL_THRESHOLD
    } else {
        crate::application::map::parallelize_bandwidth_bound::<T>(len, 2)
    }
}

/// Apply `f` elementwise into caller-owned output. Raw closures are treated as
/// compute-bound (the historical eager-parallel default, preserved so existing
/// callers do not regress); the typed [`unary_map_into`] passes the op's
/// [`UnaryOp::COMPUTE_BOUND`] instead.
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
    #[cfg(not(feature = "parallel"))]
    let _ = compute_bound;
    if input.shape() != output.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: input.shape().to_vec(),
            rhs: output.shape().to_vec(),
        });
    }

    if let (Some(input_slice), Some(output_slice)) = (input.as_slice(), output.as_mut_slice()) {
        #[cfg(feature = "parallel")]
        {
            if should_parallelize::<T>(input_slice.len(), compute_bound) {
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
        if should_parallelize::<T>(size, compute_bound) {
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

    // Row-walk traversal: one offset computation per innermost row, then a
    // stride-increment walk. Column-walk views use cache-line micro-tiles from
    // the same geometry policy as binary_map.
    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(());
    };
    let in_step = traversal.last_axis_stride(input_layout);
    let out_step = traversal.last_axis_stride(output_layout);
    let input_tile = line_elements::<T>();
    let output_tile = line_elements::<U>();
    if in_step.unsigned_abs() >= input_tile || out_step.unsigned_abs() >= output_tile {
        let tile = input_tile.min(output_tile);
        if let Some(geometry) = TileGeometry::new(size, shape, tile) {
            let input_row_step = input_layout.strides[N - 2];
            let output_row_step = output_layout.strides[N - 2];
            for slab in 0..geometry.slabs() {
                let slab_idx = geometry.slab_base_index(slab);
                let input_slab_base = input_layout.offset_of(slab_idx)? as isize;
                let output_slab_base = output_layout.offset_of(slab_idx)? as isize;
                for row_block in (0..geometry.height()).step_by(geometry.tile()) {
                    let row_end = (row_block + geometry.tile()).min(geometry.height());
                    for col_block in (0..geometry.width()).step_by(geometry.tile()) {
                        let col_end = (col_block + geometry.tile()).min(geometry.width());
                        for row in row_block..row_end {
                            let mut input_offset = input_slab_base
                                + (row as isize * input_row_step)
                                + (col_block as isize * in_step);
                            let mut output_offset = output_slab_base
                                + (row as isize * output_row_step)
                                + (col_block as isize * out_step);
                            for _ in col_block..col_end {
                                output_data[output_offset as usize] =
                                    f(input_data[input_offset as usize]);
                                input_offset += in_step;
                                output_offset += out_step;
                            }
                        }
                    }
                }
            }
            return Ok(());
        }
    }
    for row in 0..traversal.rows() {
        let base_idx = traversal.base_index(row);
        let mut input_offset = input_layout.offset_of(base_idx)? as isize;
        let mut output_offset = output_layout.offset_of(base_idx)? as isize;
        for _ in 0..traversal.inner() {
            output_data[output_offset as usize] = f(input_data[input_offset as usize]);
            input_offset += in_step;
            output_offset += out_step;
        }
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

/// Apply `f` to every element of `view` in place.
///
/// This is the `ndarray::mapv_inplace` analogue. Elementwise in-place mutation
/// is memory-order independent, so the contiguous fast path accepts any dense
/// block (C or F). Zero-stride write aliasing is rejected because it would
/// apply `f` to a single physical element more than once.
pub fn map_inplace<T, F, const N: usize>(view: &mut ArrayViewMut<'_, T, N>, f: F) -> Result<()>
where
    T: Copy + Send + Sync + 'static,
    F: Fn(T) -> T + Copy + Send + Sync + 'static,
{
    view.layout().validate_storage_len(view.data().len())?;
    if view.layout().has_zero_stride_aliasing() {
        return Err(LetoError::StorageError {
            reason: "in-place map layout must not contain zero-stride aliasing".to_string(),
        });
    }

    if let Some(slice) = view.as_mut_slice_memory_order() {
        #[cfg(feature = "parallel")]
        {
            if slice.len() >= PARALLEL_THRESHOLD {
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
    let len = slice.len();
    let ptr = slice.as_mut_ptr() as usize;
    let chunk_size = 4096;

    crate::infrastructure::parallel::parallel_for_chunks(len, chunk_size, move |start, end| {
        for index in start..end {
            // SAFETY: each worker mutates a unique element in `start..end` of a
            // validated dense block.
            unsafe {
                let cell = (ptr as *mut T).add(index);
                *cell = f(*cell);
            }
        }
    });
}

/// Zero-sized (or value-carrying) named real unary operation contract.
///
/// Implementors route through the shared [`map_into`]/[`mapv`] traversal via
/// [`unary_map_into`]/[`unary_map`]; no implementor defines its own traversal.
pub trait UnaryOp<T: RealScalar>: Copy + Send + Sync + 'static {
    /// Apply the scalar operation.
    fn apply(&self, x: T) -> T;

    /// Whether this op is compute-bound — heavy enough per element that
    /// parallelism pays even while the data is cache-resident (transcendentals,
    /// `powf`). Bandwidth-bound ops (`neg`, `abs`: a trivial op over each element)
    /// override to `false` and gate parallelism on working-set-vs-LLC, like
    /// binary elementwise ops, so they are not parallelized into a slowdown.
    const COMPUTE_BOUND: bool = true;
}

macro_rules! define_unary_op {
    ($(#[$meta:meta])* $name:ident => $method:ident) => {
        define_unary_op!($(#[$meta])* $name => $method, true);
    };
    ($(#[$meta:meta])* $name:ident => $method:ident, $compute_bound:expr) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl<T: RealScalar> UnaryOp<T> for $name {
            const COMPUTE_BOUND: bool = $compute_bound;
            #[inline(always)]
            fn apply(&self, x: T) -> T {
                x.$method()
            }
        }
    };
}

define_unary_op!(/// `e^x` operation marker.
    ExpOp => exp);
define_unary_op!(/// Natural logarithm operation marker.
    LnOp => ln);
define_unary_op!(/// Gauss error function operation marker.
    ErfOp => erf);
define_unary_op!(/// Complementary error function operation marker.
    ErfcOp => erfc);
define_unary_op!(/// Natural logarithm of the absolute gamma function operation marker.
    LgammaOp => lgamma);
define_unary_op!(/// Sine operation marker.
    SinOp => sin);
define_unary_op!(/// Cosine operation marker.
    CosOp => cos);
define_unary_op!(/// Square-root operation marker.
    SqrtOp => sqrt);
define_unary_op!(/// Absolute-value operation marker.
    AbsOp => abs, false);
define_unary_op!(/// Additive-inverse operation marker.
    NegOp => neg, false);
define_unary_op!(/// Reciprocal operation marker.
    RecipOp => recip);

/// Power operation carrying its exponent. Zero-cost: monomorphizes to a direct
/// `powf` call with the captured exponent.
#[derive(Clone, Copy, Debug)]
pub struct PowfOp<T: RealScalar> {
    /// The exponent applied to every element.
    pub exponent: T,
}

impl<T: RealScalar> UnaryOp<T> for PowfOp<T> {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        x.powf(self.exponent)
    }
}

/// Apply a named unary operation into caller-owned output through the shared
/// traversal kernel.
#[inline]
pub fn unary_map_into<T, Op, const N: usize>(
    op: Op,
    input: &ArrayView<'_, T, N>,
    output: &mut ArrayViewMut<'_, T, N>,
) -> Result<()>
where
    T: RealScalar,
    Op: UnaryOp<T>,
{
    map_into_gated(input, output, move |x| op.apply(x), Op::COMPUTE_BOUND)
}

/// Apply a named unary operation, allocating a C-contiguous output, through the
/// shared traversal kernel.
#[inline]
pub fn unary_map<T, Op, const N: usize>(
    op: Op,
    input: &ArrayView<'_, T, N>,
) -> Result<Array<T, VecStorage<T>, N>>
where
    T: RealScalar,
    Op: UnaryOp<T>,
{
    mapv(input, move |x| op.apply(x))
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

    // Row-walk parallel traversal: workers own disjoint innermost rows; one
    // offset computation per row, then stride-increment walks. Column-walk
    // views use the same cache-line micro-tile geometry as binary_map.
    let Some(traversal) = RowMajorTraversal::new(ctx.size, ctx.shape) else {
        return;
    };
    let in_step = traversal.last_axis_stride(ctx.input_layout);
    let out_step = traversal.last_axis_stride(ctx.output_layout);
    let input_tile = line_elements::<T>();
    let output_tile = line_elements::<U>();
    if in_step.unsigned_abs() >= input_tile || out_step.unsigned_abs() >= output_tile {
        let tile = input_tile.min(output_tile);
        if let Some(geometry) = TileGeometry::new(ctx.size, ctx.shape, tile) {
            let input_row_step = ctx.input_layout.strides[N - 2];
            let output_row_step = ctx.output_layout.strides[N - 2];
            let blocks = geometry.slabs() * geometry.row_blocks();
            let block_chunk = (4096 / (geometry.tile() * geometry.width()).max(1)).max(1);

            crate::infrastructure::parallel::parallel_for_chunks(
                blocks,
                block_chunk,
                move |start, end| {
                    for block in start..end {
                        let slab = block / geometry.row_blocks();
                        let row_block = block % geometry.row_blocks();
                        let slab_idx = geometry.slab_base_index(slab);
                        let input_slab_base = ctx
                            .input_layout
                            .offset_of(slab_idx)
                            .expect("validated input layout must map every slab base")
                            as isize;
                        let output_slab_base = ctx
                            .output_layout
                            .offset_of(slab_idx)
                            .expect("validated output layout must map every slab base")
                            as isize;
                        let row_start = row_block * geometry.tile();
                        let row_end = (row_start + geometry.tile()).min(geometry.height());

                        // SAFETY: storage spans are validated before dispatch;
                        // each offset is the affine image of a validated logical
                        // index; workers own disjoint output row blocks and
                        // zero-stride output aliasing is rejected.
                        unsafe {
                            for col_block in (0..geometry.width()).step_by(geometry.tile()) {
                                let col_end = (col_block + geometry.tile()).min(geometry.width());
                                for row in row_start..row_end {
                                    let mut input_offset = input_slab_base
                                        + (row as isize * input_row_step)
                                        + (col_block as isize * in_step);
                                    let mut output_offset = output_slab_base
                                        + (row as isize * output_row_step)
                                        + (col_block as isize * out_step);
                                    for _ in col_block..col_end {
                                        let value = *(input_ptr as *const T).offset(input_offset);
                                        *(output_ptr as *mut U).offset(output_offset) = f(value);
                                        input_offset += in_step;
                                        output_offset += out_step;
                                    }
                                }
                            }
                        }
                    }
                },
            );
            return;
        }
    }
    let row_chunk = traversal.chunk_rows_for(4096);

    crate::infrastructure::parallel::parallel_for_chunks(
        traversal.rows(),
        row_chunk,
        move |start, end| {
            for row in start..end {
                let base_idx = traversal.base_index(row);
                let mut input_offset = ctx
                    .input_layout
                    .offset_of(base_idx)
                    .expect("validated input layout must map every logical index")
                    as isize;
                let mut output_offset = ctx
                    .output_layout
                    .offset_of(base_idx)
                    .expect("validated output layout must map every logical index")
                    as isize;

                // SAFETY: storage spans are validated before dispatch; every walked
                // offset equals offset_of of a validated logical index; workers own
                // disjoint rows and zero-stride output aliasing is rejected.
                unsafe {
                    for _ in 0..traversal.inner() {
                        let value = *(input_ptr as *const T).offset(input_offset);
                        *(output_ptr as *mut U).offset(output_offset) = f(value);
                        input_offset += in_step;
                        output_offset += out_step;
                    }
                }
            }
        },
    );
}
