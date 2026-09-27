use crate::application::index::{line_elements_for, validate_mutable_output};
use crate::domain::scalar::Scalar;
use crate::infrastructure::cache::{cached_cache_geometry, CacheGeometry};
#[cfg(feature = "parallel")]
use crate::infrastructure::parallel::parallelize_bandwidth_bound;
use leto::{Array, ArrayView, ArrayViewMut, Result, VecStorage};

mod sealed {
    pub trait Sealed {}
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
    validate_mutable_output(out, "binary map")?;
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
    binary_map_with_cache_line::<Op, T, N>(lhs, rhs, out, None)
}

/// Apply a binary element-wise operation using an explicit cache geometry.
///
/// The geometry's cache-line width selects the micro-tile side for strided
/// views. This is intended for callers with topology information that is more
/// authoritative than the process-local [`cached_cache_geometry`] probe; the
/// default [`binary_map`] path remains automatically detected. Cache
/// capacities in `geometry` do not affect this operation.
pub fn binary_map_with_cache_geometry<Op, T, const N: usize>(
    lhs: &ArrayView<'_, T, N>,
    rhs: &ArrayView<'_, T, N>,
    out: &mut ArrayViewMut<'_, T, N>,
    geometry: CacheGeometry,
) -> Result<()>
where
    Op: BinaryOp<T>,
    T: Scalar,
{
    binary_map_with_cache_line::<Op, T, N>(lhs, rhs, out, Some(geometry.cache_line_bytes()))
}

fn binary_map_with_cache_line<Op, T, const N: usize>(
    lhs: &ArrayView<'_, T, N>,
    rhs: &ArrayView<'_, T, N>,
    out: &mut ArrayViewMut<'_, T, N>,
    cache_line_bytes: Option<usize>,
) -> Result<()>
where
    Op: BinaryOp<T>,
    T: Scalar,
{
    validate_binary_shapes(lhs, rhs, out)?;
    let out_shape = out.shape();

    // Equal shapes and identical strides make the three dense memory-order
    // blocks enumerate the same logical element at each position, so any
    // shared dense order (C, F, or permuted-contiguous) feeds the slice
    // kernel — not only the canonical C order.
    if lhs.shape() == out_shape
        && rhs.shape() == out_shape
        && lhs.strides() == out.strides()
        && rhs.strides() == out.strides()
    {
        if let (Some(lhs_slice), Some(rhs_slice), Some(out_slice)) = (
            lhs.as_slice_memory_order(),
            rhs.as_slice_memory_order(),
            out.as_mut_slice_memory_order(),
        ) {
            debug_assert_eq!(lhs_slice.len(), rhs_slice.len());
            debug_assert_eq!(lhs_slice.len(), out_slice.len());

            #[cfg(feature = "parallel")]
            {
                if parallelize_bandwidth_bound::<T>(lhs_slice.len(), 3) {
                    crate::application::strided::parallel_slice_into::<T, T, 2, _>(
                        [lhs_slice, rhs_slice],
                        out_slice,
                        3 * core::mem::size_of::<T>(),
                        |inputs, out| Op::apply_slice(inputs[0], inputs[1], out),
                    );
                    return Ok(());
                }
            }

            Op::apply_slice(lhs_slice, rhs_slice, out_slice);
            return Ok(());
        }
    }

    validate_binary_storage(lhs, rhs, out)?;

    let size = out.layout().checked_size()?;
    let shape = out.shape();
    let lhs_layout = lhs.layout().broadcast(shape)?;
    let rhs_layout = rhs.layout().broadcast(shape)?;
    let out_layout = out.layout();

    let lhs_data = lhs.data();
    let rhs_data = rhs.data();
    let out_data = out.data_mut();
    let cache_line_bytes =
        cache_line_bytes.unwrap_or_else(|| cached_cache_geometry().cache_line_bytes());
    let in_tile = line_elements_for::<T>(cache_line_bytes);

    #[cfg(feature = "parallel")]
    {
        // Output injectivity is established by `validate_binary_storage`, so
        // parallel workers' logical rows map to disjoint physical elements.
        if parallelize_bandwidth_bound::<T>(size, 3) {
            let lhs_ptr = lhs_data.as_ptr() as usize;
            let rhs_ptr = rhs_data.as_ptr() as usize;
            let out_ptr = out_data.as_mut_ptr() as usize;
            crate::application::strided::strided_parallel::<N, 2, 1, _>(
                crate::application::strided::StridedLayout::new(
                    size,
                    shape,
                    [lhs_layout, rhs_layout],
                    [out_layout],
                    in_tile,
                    in_tile,
                ),
                3 * core::mem::size_of::<T>(),
                move |in_off: [isize; 2], out_off: [isize; 1]| {
                    // SAFETY: spans validated before dispatch; every walked
                    // offset equals offset_of of a validated logical index;
                    // workers own disjoint (slab, row-block) pairs and the
                    // output layout has no zero-stride aliasing, so no two
                    // workers write one element.
                    unsafe {
                        let lhs_value = *(lhs_ptr as *const T).offset(in_off[0]);
                        let rhs_value = *(rhs_ptr as *const T).offset(in_off[1]);
                        *(out_ptr as *mut T).offset(out_off[0]) = Op::apply(lhs_value, rhs_value);
                    }
                },
            );
            return Ok(());
        }
    }

    // Row-walk traversal: one offset computation per innermost row, then a pure
    // stride-increment walk along the last axis, removing the per-element
    // div/mod index decomposition and the three per-element offset products
    // (the measured ~87x strided-vs-contiguous gap; see gap_audit.md). Column-
    // walk views use cache-line micro-tiles. Both arms live in
    // `application::strided`, shared with the unary map.
    crate::application::strided::strided_serial::<N, 2, 1, _>(
        crate::application::strided::StridedLayout::new(
            size,
            shape,
            [lhs_layout, rhs_layout],
            [out_layout],
            in_tile,
            in_tile,
        ),
        |in_off: [isize; 2], out_off: [isize; 1]| {
            // Every walked offset equals offset_of of a validated logical index,
            // so the usize casts are in-bounds by the storage-span validation
            // above; safe indexing still guards against defects.
            out_data[out_off[0] as usize] =
                Op::apply(lhs_data[in_off[0] as usize], rhs_data[in_off[1] as usize]);
        },
    )
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
///
/// One implementation with [`reduce_all`](crate::application::reduction::reduce_all): this is
/// `reduce_all::<SumAxis>` with the failure that cannot happen removed from
/// the signature. Sum has an identity, so an empty view is `T::ZERO` rather
/// than an error, and the only remaining failure is a malformed view, which
/// is a programmer error rather than an input class. `min`/`max` keep their
/// `Result` because an empty input is a genuine domain failure for them.
///
/// # Panics
///
/// Panics when the view's layout addresses offsets outside its storage (a
/// malformed view built by [`ArrayView::new`] without validation); a silently
/// truncated sum is never returned. Validated construction via
/// [`ArrayView::try_new`] cannot reach this panic.
pub fn sum<T: Scalar, const N: usize>(arr: &ArrayView<'_, T, N>) -> T {
    crate::application::reduction::reduce_all::<crate::application::reduction::SumAxis, T, N>(arr)
        .expect("invariant: sum requires a view whose layout fits its storage")
}
