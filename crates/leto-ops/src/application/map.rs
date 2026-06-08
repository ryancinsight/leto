use crate::domain::scalar::Scalar;
use leto::{ArrayView, ArrayViewMut, LetoError, Result};

#[cfg(feature = "parallel")]
const PARALLEL_THRESHOLD: usize = 8192;

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

// Helper to convert flat 1D index to N-dimensional index.
#[inline(always)]
fn index_from_flat<const N: usize>(flat: usize, shape: &[usize; N]) -> [usize; N] {
    let mut index = [0usize; N];
    let mut temp = flat;
    for i in (0..N).rev() {
        if shape[i] > 0 {
            index[i] = temp % shape[i];
            temp /= shape[i];
        }
    }
    index
}

#[inline]
fn validate_binary_shapes<T, const N: usize>(
    lhs: &ArrayView<'_, T, N>,
    rhs: &ArrayView<'_, T, N>,
    out: &ArrayViewMut<'_, T, N>,
) -> Result<()> {
    if lhs.shape() != rhs.shape() || lhs.shape() != out.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }
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

    if let (Some(lhs_slice), Some(rhs_slice), Some(out_slice)) =
        (lhs.as_slice(), rhs.as_slice(), out.as_mut_slice())
    {
        debug_assert_eq!(lhs_slice.len(), rhs_slice.len());
        debug_assert_eq!(lhs_slice.len(), out_slice.len());

        #[cfg(feature = "parallel")]
        {
            if lhs_slice.len() >= PARALLEL_THRESHOLD {
                parallel_binary_map_slice::<Op, T>(lhs_slice, rhs_slice, out_slice);
                return Ok(());
            }
        }

        Op::apply_slice(lhs_slice, rhs_slice, out_slice);
        return Ok(());
    }

    validate_binary_storage(lhs, rhs, out)?;
    let size = lhs.layout().checked_size()?;
    let shape = lhs.shape();
    let lhs_layout = lhs.layout();
    let rhs_layout = rhs.layout();
    let out_layout = out.layout();

    let lhs_data = lhs.data();
    let rhs_data = rhs.data();
    let out_data = out.data_mut();

    #[cfg(feature = "parallel")]
    {
        if size >= PARALLEL_THRESHOLD && !out_layout.has_zero_stride_aliasing() {
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

    for flat_idx in 0..size {
        let multi_idx = index_from_flat(flat_idx, &shape);
        let lhs_off = lhs_layout.offset_of(multi_idx)?;
        let rhs_off = rhs_layout.offset_of(multi_idx)?;
        let out_off = out_layout.offset_of(multi_idx)?;
        out_data[out_off] = Op::apply(lhs_data[lhs_off], rhs_data[rhs_off]);
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

    crate::infrastructure::parallel::parallel_for(0, numel, move |i| {
        // SAFETY: each worker writes a unique `i` in `0..numel`; all slices
        // have equal length by `binary_map` validation and are alive for the
        // duration of `parallel_for`.
        unsafe {
            let lhs_val = *(lhs_ptr as *const T).add(i);
            let rhs_val = *(rhs_ptr as *const T).add(i);
            *(out_ptr as *mut T).add(i) = Op::apply(lhs_val, rhs_val);
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

    crate::infrastructure::parallel::parallel_for(0, ctx.size, move |flat_idx| {
        let multi_idx = index_from_flat(flat_idx, &ctx.shape);
        let lhs_off = ctx
            .lhs_layout
            .offset_of(multi_idx)
            .expect("validated lhs layout must map every logical index");
        let rhs_off = ctx
            .rhs_layout
            .offset_of(multi_idx)
            .expect("validated rhs layout must map every logical index");
        let out_off = ctx
            .out_layout
            .offset_of(multi_idx)
            .expect("validated output layout must map every logical index");

        // SAFETY: storage spans are validated before dispatch; each logical
        // flat index maps to one output offset. Mutable views that can alias
        // through broadcast zero strides are rejected by Leto view construction.
        unsafe {
            let lhs_val = *(lhs_ptr as *const T).add(lhs_off);
            let rhs_val = *(rhs_ptr as *const T).add(rhs_off);
            *(out_ptr as *mut T).add(out_off) = Op::apply(lhs_val, rhs_val);
        }
    });
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

// -- Reductions --

/// Sum reduction over all elements of the view.
pub fn sum<T: Scalar, const N: usize>(arr: &ArrayView<'_, T, N>) -> T {
    if let Some(slice) = arr.as_slice() {
        return T::sum_slice(slice);
    }

    let size = arr.size();
    let shape = arr.shape();
    let layout = arr.layout();
    let data = arr.data();

    let mut total = T::ZERO;
    for flat_idx in 0..size {
        let multi_idx = index_from_flat(flat_idx, &shape);
        if let Ok(off) = layout.offset_of(multi_idx) {
            total = total.add(data[off]);
        }
    }
    total
}

// -- Matrix Multiplication --

/// Perform matrix multiplication `out = lhs * rhs` for 2D views.
///
/// Uses cache-friendly loop ordering and optional row-wise Moirai parallelization.
pub fn matmul<T: Scalar>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &mut ArrayViewMut<'_, T, 2>,
) -> Result<()> {
    let [m, k1] = lhs.shape();
    let [k2, n] = rhs.shape();
    let [out_m, out_n] = out.shape();

    if k1 != k2 || m != out_m || n != out_n {
        return Err(LetoError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }
    lhs.layout().validate_storage_len(lhs.data().len())?;
    rhs.layout().validate_storage_len(rhs.data().len())?;
    out.layout().validate_storage_len(out.data().len())?;

    // Zero out initial output view.
    for r in 0..m {
        for c in 0..n {
            *out.get_mut([r, c])? = T::ZERO;
        }
    }

    #[cfg(feature = "parallel")]
    {
        if m >= 16 && !out.layout().has_zero_stride_aliasing() {
            let lhs_ptr = lhs.data().as_ptr() as usize;
            let rhs_ptr = rhs.data().as_ptr() as usize;
            let out_ptr = out.data_mut().as_mut_ptr() as usize;

            let lhs_layout = lhs.layout();
            let rhs_layout = rhs.layout();
            let out_layout = out.layout();

            crate::infrastructure::parallel::parallel_for(0, m, move |i| {
                for k in 0..k1 {
                    let lhs_off = lhs_layout
                        .offset_of([i, k])
                        .expect("validated lhs matrix layout");
                    // SAFETY: matrix layout validation happens through `get_mut`
                    // during zeroing and every row task writes a distinct row.
                    let lhs_val = unsafe { *(lhs_ptr as *const T).add(lhs_off) };
                    if lhs_val == T::ZERO {
                        continue;
                    }
                    for j in 0..n {
                        let rhs_off = rhs_layout
                            .offset_of([k, j])
                            .expect("validated rhs matrix layout");
                        let out_off = out_layout
                            .offset_of([i, j])
                            .expect("validated output matrix layout");
                        // SAFETY: row `i` is exclusive to this worker, and
                        // offsets are validated by `offset_of`.
                        unsafe {
                            let rhs_val = *(rhs_ptr as *const T).add(rhs_off);
                            let out_ref = &mut *(out_ptr as *mut T).add(out_off);
                            *out_ref = out_ref.add(lhs_val.mul(rhs_val));
                        }
                    }
                }
            });
            return Ok(());
        }
    }

    // Cache-efficient sequential loop ordering.
    for i in 0..m {
        for k in 0..k1 {
            let lhs_val = *lhs.get([i, k])?;
            if lhs_val == T::ZERO {
                continue;
            }
            for j in 0..n {
                let rhs_val = *rhs.get([k, j])?;
                let out_ref = out.get_mut([i, j])?;
                *out_ref = out_ref.add(lhs_val.mul(rhs_val));
            }
        }
    }

    Ok(())
}
