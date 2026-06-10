use crate::domain::scalar::Scalar;
use leto::{Array, ArrayView, ArrayViewMut, LetoError, Result, VecStorage};

/// Direction of a prefix scan along an axis.
///
/// A descriptive two-variant enum rather than a bare boolean, so call sites
/// read `ScanDirection::Reverse` instead of `true`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanDirection {
    /// Accumulate from index 0 upward (prefix scan).
    Forward,
    /// Accumulate from the last index downward (suffix scan).
    Reverse,
}

/// Associative scan operation contract: an identity and a combine step.
///
/// Implemented by zero-sized markers so `scan_axis` monomorphizes to a direct
/// inlined accumulation with no indirection.
pub trait ScanOp<T: Scalar>: Copy + Send + Sync + 'static {
    /// The scan identity (`0` for sum, `1` for product).
    fn identity() -> T;
    /// Combine the running accumulator with the next element.
    fn combine(acc: T, x: T) -> T;
}

/// Cumulative-sum scan marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct CumSumOp;

/// Cumulative-product scan marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct CumProdOp;

impl<T: Scalar> ScanOp<T> for CumSumOp {
    #[inline(always)]
    fn identity() -> T {
        T::ZERO
    }
    #[inline(always)]
    fn combine(acc: T, x: T) -> T {
        acc.add(x)
    }
}

impl<T: Scalar> ScanOp<T> for CumProdOp {
    #[inline(always)]
    fn identity() -> T {
        T::ONE
    }
    #[inline(always)]
    fn combine(acc: T, x: T) -> T {
        acc.mul(x)
    }
}

#[inline]
fn fill_outer_index<const N: usize>(
    outer_flat: usize,
    shape: &[usize; N],
    axis: usize,
    index: &mut [usize; N],
) {
    let mut rem = outer_flat;
    for d in (0..N).rev() {
        if d == axis {
            continue;
        }
        index[d] = rem % shape[d];
        rem /= shape[d];
    }
}

/// Scan `input` along `axis` into caller-owned `output` (same shape).
///
/// Accumulation runs in the native precision of `T`. Output layouts that alias
/// writes through zero strides are rejected.
pub fn scan_axis_into<Op, T, const N: usize>(
    input: &ArrayView<'_, T, N>,
    axis: usize,
    direction: ScanDirection,
    output: &mut ArrayViewMut<'_, T, N>,
) -> Result<()>
where
    Op: ScanOp<T>,
    T: Scalar,
{
    if axis >= N {
        return Err(LetoError::StorageError {
            reason: format!("scan axis {axis} is out of bounds for rank {N}"),
        });
    }
    if input.shape() != output.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: input.shape().to_vec(),
            rhs: output.shape().to_vec(),
        });
    }
    input.layout().validate_storage_len(input.data().len())?;
    output.layout().validate_storage_len(output.data().len())?;
    if output.layout().has_zero_stride_aliasing() {
        return Err(LetoError::StorageError {
            reason: "scan output layout must not contain zero-stride aliasing".to_string(),
        });
    }

    let shape = input.shape();
    let len = shape[axis];
    if len == 0 {
        return Ok(());
    }
    let outer: usize = shape
        .iter()
        .enumerate()
        .filter(|&(d, _)| d != axis)
        .map(|(_, &s)| s)
        .product();

    let mut index = [0usize; N];
    for outer_flat in 0..outer {
        fill_outer_index(outer_flat, &shape, axis, &mut index);
        let mut acc = Op::identity();
        match direction {
            ScanDirection::Forward => {
                for k in 0..len {
                    index[axis] = k;
                    acc = Op::combine(acc, *input.get(index)?);
                    *output.get_mut(index)? = acc;
                }
            }
            ScanDirection::Reverse => {
                for k in (0..len).rev() {
                    index[axis] = k;
                    acc = Op::combine(acc, *input.get(index)?);
                    *output.get_mut(index)? = acc;
                }
            }
        }
    }
    Ok(())
}

/// Scan `input` along `axis`, allocating a C-contiguous output of equal shape.
pub fn scan_axis<Op, T, const N: usize>(
    input: &ArrayView<'_, T, N>,
    axis: usize,
    direction: ScanDirection,
) -> Result<Array<T, VecStorage<T>, N>>
where
    Op: ScanOp<T>,
    T: Scalar,
{
    let layout = leto::Layout::c_contiguous(input.shape())?;
    let mut output = Array::new(layout, VecStorage::fill(layout.size(), T::ZERO))?;
    scan_axis_into::<Op, T, N>(input, axis, direction, &mut output.view_mut())?;
    Ok(output)
}

/// Forward cumulative sum along `axis`, allocating a C-contiguous output.
#[inline]
pub fn cumsum<T: Scalar, const N: usize>(
    input: &ArrayView<'_, T, N>,
    axis: usize,
) -> Result<Array<T, VecStorage<T>, N>> {
    scan_axis::<CumSumOp, T, N>(input, axis, ScanDirection::Forward)
}

/// Forward cumulative sum along `axis` into caller-owned output.
#[inline]
pub fn cumsum_into<T: Scalar, const N: usize>(
    input: &ArrayView<'_, T, N>,
    axis: usize,
    output: &mut ArrayViewMut<'_, T, N>,
) -> Result<()> {
    scan_axis_into::<CumSumOp, T, N>(input, axis, ScanDirection::Forward, output)
}
