use crate::application::index::{
    line_elements, unit_stride_row_slice, RowMajorTraversal, TileGeometry,
};
use crate::domain::scalar::Scalar;
use leto::{Array, ArrayView, ArrayViewMut, LetoError, Result, VecStorage};

/// Whether a bandwidth-bound elementwise op over `len` elements of `T` should
/// run in parallel. A binary map reads two operands and writes one, so its
/// working set is `operands · len · size_of::<T>()`. The op is memory-bandwidth-
/// bound, and one core nearly saturates that bandwidth while the data is
/// resident in the shared last-level cache; parallelism pays only once the
/// working set spills past the LLC, where additional cores add DRAM bandwidth.
/// Below that, thread-dispatch overhead is pure loss.
///
/// Replaces a fixed 65536-element gate (`256 KB` L2 / `4 B` f32) that ignored
/// both element size and arithmetic intensity, parallelizing bandwidth-bound
/// ops far too eagerly — a 64k `f64` `add` (1.5 MB working set) ran ~3x slower
/// parallel than serial. See gap_audit `2026-07-19 Parallel Threshold`.
#[cfg(feature = "parallel")]
pub(crate) fn parallelize_bandwidth_bound<T>(len: usize, operands: usize) -> bool {
    let working_set = len
        .saturating_mul(operands)
        .saturating_mul(core::mem::size_of::<T>());
    working_set > crate::infrastructure::cache::cached_cache_geometry().l3_bytes()
}

mod sealed {
    pub trait Sealed {}
}

#[cfg(feature = "parallel")]
struct StridedBinaryContext<'a, T, const N: usize> {
    size: usize,
    shape: [usize; N],
    lhs_layout: leto::Layout<N>,
    rhs_layout: leto::Layout<N>,
    out_layout: leto::Layout<N>,
    lhs_data: &'a [T],
    rhs_data: &'a [T],
    out_data: &'a mut [T],
}

/// Zero-sized binary operation contract for element-wise kernels.
pub trait BinaryOp<T: Scalar>: sealed::Sealed + Copy + Send + Sync + 'static {
    /// Apply the scalar operation.
    fn apply(lhs: T, rhs: T) -> T;

    /// Apply the operation to three same-length contiguous slices.
    fn apply_slice(lhs: &[T], rhs: &[T], out: &mut [T]);
}

/// Addition operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AddOp;

/// Subtraction operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SubOp;

/// Multiplication operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct MulOp;

/// Division operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct DivOp;

impl sealed::Sealed for AddOp {}
impl sealed::Sealed for SubOp {}
impl sealed::Sealed for MulOp {}
impl sealed::Sealed for DivOp {}

macro_rules! define_comparison_op {
    ($name:ident, $comparison:tt) => {
        #[doc = concat!(stringify!($name), " applies the ", stringify!($comparison), " comparison and writes a zero-or-one mask.")]
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl sealed::Sealed for $name {}

        impl<T: Scalar> BinaryOp<T> for $name {
            #[inline(always)]
            fn apply(lhs: T, rhs: T) -> T {
                if lhs $comparison rhs {
                    T::ONE
                } else {
                    T::ZERO
                }
            }

            #[inline(always)]
            fn apply_slice(lhs: &[T], rhs: &[T], out: &mut [T]) {
                for ((out, lhs), rhs) in out.iter_mut().zip(lhs).zip(rhs) {
                    *out = Self::apply(*lhs, *rhs);
                }
            }
        }
    };
}

define_comparison_op!(EqOp, ==);
define_comparison_op!(NeOp, !=);
define_comparison_op!(LtOp, <);
define_comparison_op!(GtOp, >);
define_comparison_op!(LeOp, <=);
define_comparison_op!(GeOp, >=);

impl<T: Scalar> BinaryOp<T> for AddOp {
    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> T {
        lhs.add(rhs)
    }

    #[inline(always)]
    fn apply_slice(lhs: &[T], rhs: &[T], out: &mut [T]) {
        T::add_slice(lhs, rhs, out);
    }
}

impl<T: Scalar> BinaryOp<T> for SubOp {
    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> T {
        lhs.sub(rhs)
    }

    #[inline(always)]
    fn apply_slice(lhs: &[T], rhs: &[T], out: &mut [T]) {
        T::sub_slice(lhs, rhs, out);
    }
}

impl<T: Scalar> BinaryOp<T> for MulOp {
    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> T {
        lhs.mul(rhs)
    }

    #[inline(always)]
    fn apply_slice(lhs: &[T], rhs: &[T], out: &mut [T]) {
        T::mul_slice(lhs, rhs, out);
    }
}

impl<T: Scalar> BinaryOp<T> for DivOp {
    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> T {
        lhs.div(rhs)
    }

    #[inline(always)]
    fn apply_slice(lhs: &[T], rhs: &[T], out: &mut [T]) {
        T::div_slice(lhs, rhs, out);
    }
}

#[inline]
fn validate_binary_shapes<T, const N: usize>(
    lhs: &ArrayView<'_, T, N>,
    rhs: &ArrayView<'_, T, N>,
    out: &ArrayViewMut<'_, T, N>,
) -> Result<()> {
    lhs.layout().broadcast(out.shape())?;
    rhs.layout().broadcast(out.shape())?;
    Ok(())
}

#[inline]
fn validate_binary_storage<T, const N: usize>(
    lhs: &ArrayView<'_, T, N>,
    rhs: &ArrayView<'_, T, N>,
    out: &ArrayViewMut<'_, T, N>,
) -> Result<()> {
    lhs.layout().validate_storage_len(lhs.data().len())?;
    rhs.layout().validate_storage_len(rhs.data().len())?;
    out.layout().validate_storage_len(out.data().len())?;
    Ok(())
}

/// Apply a binary element-wise operation to two input views and one mutable output view.
pub fn binary_map<Op, T, const N: usize>(
    lhs: &ArrayView<'_, T, N>,
    rhs: &ArrayView<'_, T, N>,
    out: &mut ArrayViewMut<'_, T, N>,
) -> Result<()>
where
    Op: BinaryOp<T>,
    T: Scalar,
{
    validate_binary_shapes(lhs, rhs, out)?;
    let out_shape = out.shape();

    if let (Some(lhs_slice), Some(rhs_slice), Some(out_slice)) =
        (lhs.as_slice(), rhs.as_slice(), out.as_mut_slice())
    {
        if lhs.shape() == out_shape && rhs.shape() == out_shape {
            debug_assert_eq!(lhs_slice.len(), rhs_slice.len());
            debug_assert_eq!(lhs_slice.len(), out_slice.len());

            #[cfg(feature = "parallel")]
            {
                if parallelize_bandwidth_bound::<T>(lhs_slice.len(), 3) {
                    parallel_binary_map_slice::<Op, T>(lhs_slice, rhs_slice, out_slice);
                    return Ok(());
                }
            }

            Op::apply_slice(lhs_slice, rhs_slice, out_slice);
            return Ok(());
        }
    }

    validate_binary_storage(lhs, rhs, out)?;
    if out.layout().has_zero_stride_aliasing() {
        return Err(LetoError::StorageError {
            reason: "binary output layout must not contain zero-stride aliasing".to_string(),
        });
    }

    let size = out.layout().checked_size()?;
    let shape = out.shape();
    let lhs_layout = lhs.layout().broadcast(shape)?;
    let rhs_layout = rhs.layout().broadcast(shape)?;
    let out_layout = out.layout();

    let lhs_data = lhs.data();
    let rhs_data = rhs.data();
    let out_data = out.data_mut();

    #[cfg(feature = "parallel")]
    {
        if parallelize_bandwidth_bound::<T>(size, 3) && !out_layout.has_zero_stride_aliasing() {
            parallel_binary_map_strided::<Op, T, N>(StridedBinaryContext {
                size,
                shape,
                lhs_layout,
                rhs_layout,
                out_layout,
                lhs_data,
                rhs_data,
                out_data,
            });
            return Ok(());
        }
    }

    // Row-walk traversal: one offset computation per innermost row, then a
    // pure stride-increment walk along the last axis. Removes the per-element
    // div/mod index decomposition and the three per-element offset products
    // (the measured ~87x strided-vs-contiguous gap; see gap_audit.md).
    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(());
    };
    let lhs_step = traversal.last_axis_stride(lhs_layout);
    let rhs_step = traversal.last_axis_stride(rhs_layout);
    let out_step = traversal.last_axis_stride(out_layout);

    // Cache-line micro-tiling pays exactly when some operand's last-axis
    // walk skips whole lines (|stride| >= elements-per-line); unit and
    // reverse-unit strides already consume lines fully and keep row-walk.
    let tile = line_elements::<T>();
    let column_walk = lhs_step.unsigned_abs() >= tile
        || rhs_step.unsigned_abs() >= tile
        || out_step.unsigned_abs() >= tile;
    if column_walk {
        if let Some(geometry) = TileGeometry::new(size, shape, tile) {
            let (lhs_rs, rhs_rs, out_rs) = (
                lhs_layout.strides[N - 2],
                rhs_layout.strides[N - 2],
                out_layout.strides[N - 2],
            );
            for slab in 0..geometry.slabs() {
                let base_idx = geometry.slab_base_index(slab);
                let lhs_base = lhs_layout.offset_of(base_idx)? as isize;
                let rhs_base = rhs_layout.offset_of(base_idx)? as isize;
                let out_base = out_layout.offset_of(base_idx)? as isize;
                let mut rb = 0;
                while rb < geometry.height() {
                    let rend = (rb + geometry.tile()).min(geometry.height());
                    let mut cb = 0;
                    while cb < geometry.width() {
                        let cend = (cb + geometry.tile()).min(geometry.width());
                        for r in rb..rend {
                            let r = r as isize;
                            let c0 = cb as isize;
                            let mut lhs_off = lhs_base + r * lhs_rs + c0 * lhs_step;
                            let mut rhs_off = rhs_base + r * rhs_rs + c0 * rhs_step;
                            let mut out_off = out_base + r * out_rs + c0 * out_step;
                            for _ in cb..cend {
                                out_data[out_off as usize] = Op::apply(
                                    lhs_data[lhs_off as usize],
                                    rhs_data[rhs_off as usize],
                                );
                                lhs_off += lhs_step;
                                rhs_off += rhs_step;
                                out_off += out_step;
                            }
                        }
                        cb = cend;
                    }
                    rb = rend;
                }
            }
            return Ok(());
        }
    }

    for row in 0..traversal.rows() {
        let base_idx = traversal.base_index(row);
        let mut lhs_off = lhs_layout.offset_of(base_idx)? as isize;
        let mut rhs_off = rhs_layout.offset_of(base_idx)? as isize;
        let mut out_off = out_layout.offset_of(base_idx)? as isize;
        for _ in 0..traversal.inner() {
            // Every walked offset equals offset_of of a validated logical
            // index, so the usize casts are in-bounds by the storage-span
            // validation above; safe indexing still guards against defects.
            out_data[out_off as usize] =
                Op::apply(lhs_data[lhs_off as usize], rhs_data[rhs_off as usize]);
            lhs_off += lhs_step;
            rhs_off += rhs_step;
            out_off += out_step;
        }
    }

    Ok(())
}

#[cfg(feature = "parallel")]
fn parallel_binary_map_slice<Op, T>(lhs: &[T], rhs: &[T], out: &mut [T])
where
    Op: BinaryOp<T>,
    T: Scalar,
{
    let numel = lhs.len();
    let lhs_ptr = lhs.as_ptr() as usize;
    let rhs_ptr = rhs.as_ptr() as usize;
    let out_ptr = out.as_mut_ptr() as usize;
    let chunk_size = 4096;

    crate::infrastructure::parallel::parallel_for_chunks(numel, chunk_size, move |start, end| {
        // SAFETY: each worker writes to a distinct range of `out` corresponding to `start..end`.
        // The slices are valid for the lifetime of parallel execution.
        unsafe {
            let lhs_chunk =
                std::slice::from_raw_parts((lhs_ptr as *const T).add(start), end - start);
            let rhs_chunk =
                std::slice::from_raw_parts((rhs_ptr as *const T).add(start), end - start);
            let out_chunk =
                std::slice::from_raw_parts_mut((out_ptr as *mut T).add(start), end - start);
            Op::apply_slice(lhs_chunk, rhs_chunk, out_chunk);
        }
    });
}

#[cfg(feature = "parallel")]
fn parallel_binary_map_strided<Op, T, const N: usize>(ctx: StridedBinaryContext<'_, T, N>)
where
    Op: BinaryOp<T>,
    T: Scalar,
{
    let lhs_ptr = ctx.lhs_data.as_ptr() as usize;
    let rhs_ptr = ctx.rhs_data.as_ptr() as usize;
    let out_ptr = ctx.out_data.as_mut_ptr() as usize;

    // Row-walk parallel traversal: workers own disjoint ranges of innermost
    // rows; each row costs one offset computation plus a stride-increment
    // walk (see the serial path for the rationale and baseline numbers).
    let Some(traversal) = RowMajorTraversal::new(ctx.size, ctx.shape) else {
        return;
    };
    let lhs_step = traversal.last_axis_stride(ctx.lhs_layout);
    let rhs_step = traversal.last_axis_stride(ctx.rhs_layout);
    let out_step = traversal.last_axis_stride(ctx.out_layout);

    // Cache-line micro-tiling for column walks (see the serial path): workers
    // own disjoint (slab, row-block) pairs, so no two workers share an output
    // row and the aliasing-rejection guarantee carries over unchanged.
    let tile = line_elements::<T>();
    let column_walk = lhs_step.unsigned_abs() >= tile
        || rhs_step.unsigned_abs() >= tile
        || out_step.unsigned_abs() >= tile;
    if column_walk {
        if let Some(geometry) = TileGeometry::new(ctx.size, ctx.shape, tile) {
            let (lhs_rs, rhs_rs, out_rs) = (
                ctx.lhs_layout.strides[N - 2],
                ctx.rhs_layout.strides[N - 2],
                ctx.out_layout.strides[N - 2],
            );
            let blocks = geometry.slabs() * geometry.row_blocks();
            let block_chunk = (4096 / (geometry.tile() * geometry.width()).max(1)).max(1);
            crate::infrastructure::parallel::parallel_for_chunks(
                blocks,
                block_chunk,
                move |start, end| {
                    for block in start..end {
                        let slab = block / geometry.row_blocks();
                        let rb = (block % geometry.row_blocks()) * geometry.tile();
                        let rend = (rb + geometry.tile()).min(geometry.height());
                        let base_idx = geometry.slab_base_index(slab);
                        let lhs_base = ctx
                            .lhs_layout
                            .offset_of(base_idx)
                            .expect("validated lhs layout must map every logical index")
                            as isize;
                        let rhs_base = ctx
                            .rhs_layout
                            .offset_of(base_idx)
                            .expect("validated rhs layout must map every logical index")
                            as isize;
                        let out_base = ctx
                            .out_layout
                            .offset_of(base_idx)
                            .expect("validated output layout must map every logical index")
                            as isize;
                        let mut cb = 0;
                        while cb < geometry.width() {
                            let cend = (cb + geometry.tile()).min(geometry.width());
                            for r in rb..rend {
                                let r = r as isize;
                                let c0 = cb as isize;
                                let mut lhs_off = lhs_base + r * lhs_rs + c0 * lhs_step;
                                let mut rhs_off = rhs_base + r * rhs_rs + c0 * rhs_step;
                                let mut out_off = out_base + r * out_rs + c0 * out_step;
                                // SAFETY: spans validated before dispatch; every
                                // walked offset equals offset_of of a validated
                                // logical index; workers own disjoint row blocks
                                // and zero-stride output aliasing is rejected.
                                unsafe {
                                    for _ in cb..cend {
                                        let lhs_val = *(lhs_ptr as *const T).offset(lhs_off);
                                        let rhs_val = *(rhs_ptr as *const T).offset(rhs_off);
                                        *(out_ptr as *mut T).offset(out_off) =
                                            Op::apply(lhs_val, rhs_val);
                                        lhs_off += lhs_step;
                                        rhs_off += rhs_step;
                                        out_off += out_step;
                                    }
                                }
                            }
                            cb = cend;
                        }
                    }
                },
            );
            return;
        }
    }
    // Keep roughly the previous elements-per-chunk granularity.
    let row_chunk = traversal.chunk_rows_for(4096);

    crate::infrastructure::parallel::parallel_for_chunks(
        traversal.rows(),
        row_chunk,
        move |start, end| {
            for row in start..end {
                let base_idx = traversal.base_index(row);
                let mut lhs_off = ctx
                    .lhs_layout
                    .offset_of(base_idx)
                    .expect("validated lhs layout must map every logical index")
                    as isize;
                let mut rhs_off = ctx
                    .rhs_layout
                    .offset_of(base_idx)
                    .expect("validated rhs layout must map every logical index")
                    as isize;
                let mut out_off = ctx
                    .out_layout
                    .offset_of(base_idx)
                    .expect("validated output layout must map every logical index")
                    as isize;

                // SAFETY: storage spans are validated before dispatch; every
                // walked offset equals offset_of of a validated logical index;
                // each worker owns disjoint rows and the output layout has no
                // zero-stride aliasing, so no two workers write one element.
                unsafe {
                    for _ in 0..traversal.inner() {
                        let lhs_val = *(lhs_ptr as *const T).offset(lhs_off);
                        let rhs_val = *(rhs_ptr as *const T).offset(rhs_off);
                        *(out_ptr as *mut T).offset(out_off) = Op::apply(lhs_val, rhs_val);
                        lhs_off += lhs_step;
                        rhs_off += rhs_step;
                        out_off += out_step;
                    }
                }
            }
        },
    );
}

/// Element-wise array addition: `out = lhs + rhs`.
#[inline]
pub fn add<T: Scalar, const N: usize>(
    lhs: &ArrayView<'_, T, N>,
    rhs: &ArrayView<'_, T, N>,
    out: &mut ArrayViewMut<'_, T, N>,
) -> Result<()> {
    binary_map::<AddOp, T, N>(lhs, rhs, out)
}

/// Element-wise array subtraction: `out = lhs - rhs`.
#[inline]
pub fn sub<T: Scalar, const N: usize>(
    lhs: &ArrayView<'_, T, N>,
    rhs: &ArrayView<'_, T, N>,
    out: &mut ArrayViewMut<'_, T, N>,
) -> Result<()> {
    binary_map::<SubOp, T, N>(lhs, rhs, out)
}

/// Element-wise array multiplication: `out = lhs * rhs`.
#[inline]
pub fn mul<T: Scalar, const N: usize>(
    lhs: &ArrayView<'_, T, N>,
    rhs: &ArrayView<'_, T, N>,
    out: &mut ArrayViewMut<'_, T, N>,
) -> Result<()> {
    binary_map::<MulOp, T, N>(lhs, rhs, out)
}

/// Element-wise array division: `out = lhs / rhs`.
#[inline]
pub fn div<T: Scalar, const N: usize>(
    lhs: &ArrayView<'_, T, N>,
    rhs: &ArrayView<'_, T, N>,
    out: &mut ArrayViewMut<'_, T, N>,
) -> Result<()> {
    binary_map::<DivOp, T, N>(lhs, rhs, out)
}

// -- Scalar broadcast --

/// Apply a binary operation between every element and a single scalar,
/// allocating a C-contiguous output: `out = op(input, scalar)`.
///
/// Reuses the [`BinaryOp`] markers and the shared allocating traversal
/// ([`crate::application::unary::mapv`]) so no scalar-specific kernel exists.
/// `add`/`sub`/`mul`/`div` against a scalar are therefore `scalar_map::<AddOp>`
/// and friends.
#[inline]
pub fn scalar_map<Op, T, const N: usize>(
    input: &ArrayView<'_, T, N>,
    scalar: T,
) -> Result<Array<T, VecStorage<T>, N>>
where
    Op: BinaryOp<T>,
    T: Scalar,
{
    crate::application::unary::mapv(input, move |x| Op::apply(x, scalar))
}

/// Apply a binary operation between every element and a single scalar into
/// caller-owned output: `out = op(input, scalar)`.
///
/// Scalar arithmetic (`add`/`sub`/`mul`/`div` against a constant) is bandwidth-
/// bound like [`binary_map`], so parallelism is gated on the working set versus
/// the LLC rather than the eager compute-bound default — passing `false` to
/// the internal `map_into_gated` kernel.
#[inline]
pub fn scalar_map_into<Op, T, const N: usize>(
    input: &ArrayView<'_, T, N>,
    scalar: T,
    output: &mut ArrayViewMut<'_, T, N>,
) -> Result<()>
where
    Op: BinaryOp<T>,
    T: Scalar,
{
    crate::application::unary::map_into_gated(input, output, move |x| Op::apply(x, scalar), false)
}

// -- Reductions --

/// Sum reduction over all elements of the view.
pub fn sum<T: Scalar, const N: usize>(arr: &ArrayView<'_, T, N>) -> T {
    // Sum is logically order-independent, so any dense memory-order slice
    // (C, F, or permuted-contiguous) feeds the vectorized reduction directly.
    if let Some(slice) = arr.as_slice_memory_order() {
        return T::sum_slice(slice);
    }

    let size = arr.size();
    let shape = arr.shape();
    let layout = arr.layout();
    let data = arr.data();

    let mut total = T::ZERO;
    if let Some(traversal) = RowMajorTraversal::new(size, shape) {
        let step = traversal.last_axis_stride(layout);
        for row in 0..traversal.rows() {
            let base_idx = traversal.base_index(row);
            if let Ok(mut offset) = layout.offset_of(base_idx).map(|offset| offset as isize) {
                if let Some(slice) = unit_stride_row_slice(data, offset, step, traversal.inner()) {
                    total = total.add(T::sum_slice(slice));
                    continue;
                }
                for _ in 0..traversal.inner() {
                    if let Ok(index) = usize::try_from(offset) {
                        if let Some(value) = data.get(index) {
                            total = total.add(*value);
                        }
                    }
                    offset += step;
                }
            }
        }
    }
    total
}
