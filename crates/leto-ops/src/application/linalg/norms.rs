use crate::application::index::{unit_stride_row_slice, RowMajorTraversal};
use crate::domain::real::RealScalar;
use leto::{Array, ArrayView, ArrayViewMut, LetoError, Result, VecStorage};

/// Zero-sized norm-kind contract: an accumulation step and a finishing map.
///
/// One generic [`norm`] traversal monomorphizes per `(K, T)`; each kind
/// contributes only its fold step and post-reduction map, mirroring the
/// `BinaryOp`/`UnaryOp`/`ScanOp` ZST pattern used across leto-ops.
///
/// Elementwise norms are traversal-order independent, so the contiguous fast
/// path may run in physical memory order over any dense block.
pub trait NormKind<T: RealScalar>: Copy + Send + Sync + 'static {
    /// Optional dense-slice fast path for norms with a fused reduction kernel.
    #[inline]
    fn accumulate_slice(_slice: &[T]) -> Option<T> {
        None
    }
    /// Fold one element's contribution into the accumulator.
    fn accumulate(acc: T, x: T) -> T;
    /// Combine two already-accumulated partial rows.
    #[inline]
    fn combine(acc: T, row_acc: T) -> T {
        Self::accumulate(acc, Self::finish(row_acc))
    }
    /// Map the final accumulator to the norm value.
    fn finish(acc: T) -> T;
}

/// L1 norm marker: `Σ |x|`.
#[derive(Clone, Copy, Debug, Default)]
pub struct NormL1;

/// L2 / Frobenius norm marker: `sqrt(Σ x²)`. Over rank-1 this is the
/// Euclidean vector norm; over rank-2+ it is the Frobenius norm — one
/// generic entry point covers both.
#[derive(Clone, Copy, Debug, Default)]
pub struct NormL2;

/// Max (infinity) norm marker: `max |x|`.
#[derive(Clone, Copy, Debug, Default)]
pub struct NormMax;

impl<T: RealScalar> NormKind<T> for NormL1 {
    #[inline]
    fn accumulate_slice(slice: &[T]) -> Option<T> {
        Some(T::abs_sum_slice(slice))
    }

    #[inline(always)]
    fn accumulate(acc: T, x: T) -> T {
        acc.add(x.abs())
    }
    #[inline(always)]
    fn combine(acc: T, row_acc: T) -> T {
        acc.add(row_acc)
    }
    #[inline(always)]
    fn finish(acc: T) -> T {
        acc
    }
}

impl<T: RealScalar> NormKind<T> for NormL2 {
    #[inline]
    fn accumulate_slice(slice: &[T]) -> Option<T> {
        Some(T::dot_slice(slice, slice))
    }

    #[inline(always)]
    fn accumulate(acc: T, x: T) -> T {
        acc.add(x.mul(x))
    }
    #[inline(always)]
    fn combine(acc: T, row_acc: T) -> T {
        acc.add(row_acc)
    }
    #[inline(always)]
    fn finish(acc: T) -> T {
        acc.sqrt()
    }
}

impl<T: RealScalar> NormKind<T> for NormMax {
    #[inline]
    fn accumulate_slice(slice: &[T]) -> Option<T> {
        Some(T::abs_max_slice(slice))
    }

    #[inline(always)]
    fn accumulate(acc: T, x: T) -> T {
        let magnitude = x.abs();
        if magnitude > acc {
            magnitude
        } else {
            acc
        }
    }
    #[inline(always)]
    fn combine(acc: T, row_acc: T) -> T {
        if row_acc > acc {
            row_acc
        } else {
            acc
        }
    }
    #[inline(always)]
    fn finish(acc: T) -> T {
        acc
    }
}

/// Compute a norm of every element of `view`, selected by the `K` marker.
///
/// Accumulation runs in the native precision of `T` (no hidden widening; a
/// caller needing a wider accumulator converts the input explicitly). All
/// norms start from `T::ZERO`, which is also the mathematically correct
/// result for an empty view. Contiguous views (any dense memory order, since
/// elementwise norms are order-independent) take a slice fast path; strided
/// views fall back to logical index traversal without materializing a copy.
pub fn norm<K, T, const N: usize>(view: &ArrayView<'_, T, N>) -> Result<T>
where
    K: NormKind<T>,
    T: RealScalar,
{
    view.layout().validate_storage_len(view.data().len())?;

    if let Some(slice) = view.as_slice_memory_order() {
        if let Some(acc) = K::accumulate_slice(slice) {
            return Ok(K::finish(acc));
        }
        let mut acc = T::ZERO;
        for &x in slice {
            acc = K::accumulate(acc, x);
        }
        return Ok(K::finish(acc));
    }

    let size = view.layout().checked_size()?;
    let shape = view.shape();
    let layout = view.layout();
    let data = view.data();

    let mut acc = T::ZERO;
    if let Some(traversal) = RowMajorTraversal::new(size, shape) {
        let step = traversal.last_axis_stride(layout);
        for row in 0..traversal.rows() {
            let base_idx = traversal.base_index(row);
            let mut offset = layout.offset_of(base_idx)? as isize;
            if let Some(slice) = unit_stride_row_slice(data, offset, step, traversal.inner()) {
                if let Some(row_acc) = K::accumulate_slice(slice) {
                    acc = K::combine(acc, row_acc);
                    continue;
                }
            }
            for _ in 0..traversal.inner() {
                acc = K::accumulate(acc, data[offset as usize]);
                offset += step;
            }
        }
    }
    Ok(K::finish(acc))
}

/// L1 norm: `Σ |x|`.
#[inline]
pub fn norm_l1<T: RealScalar, const N: usize>(view: &ArrayView<'_, T, N>) -> Result<T> {
    norm::<NormL1, T, N>(view)
}

/// L2 (rank-1) / Frobenius (rank-2+) norm: `sqrt(Σ x²)`.
#[inline]
pub fn norm_l2<T: RealScalar, const N: usize>(view: &ArrayView<'_, T, N>) -> Result<T> {
    norm::<NormL2, T, N>(view)
}

/// Max (infinity) norm: `max |x|`.
#[inline]
pub fn norm_max<T: RealScalar, const N: usize>(view: &ArrayView<'_, T, N>) -> Result<T> {
    norm::<NormMax, T, N>(view)
}

/// L2 normalize every element of `input` directly into caller-owned `output`.
pub fn l2_normalize_into<T: RealScalar, const N: usize>(
    input: &ArrayView<'_, T, N>,
    output: &mut ArrayViewMut<'_, T, N>,
    epsilon: T,
) -> Result<()> {
    if input.shape() != output.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: input.shape().to_vec(),
            rhs: output.shape().to_vec(),
        });
    }
    input.layout().validate_storage_len(input.data().len())?;
    output.layout().validate_storage_len(output.data().len())?;

    let l2 = norm_l2(input)?;
    let denom = l2.add(epsilon);
    if denom == T::ZERO {
        crate::application::unary::map_into(input, output, |_| T::ZERO)?;
        return Ok(());
    }

    let scale = T::ONE.div(denom);
    crate::application::unary::map_into(input, output, move |val| val.mul(scale))?;
    Ok(())
}

/// L2 normalize every element of `input` and return the owned array.
pub fn l2_normalize<T: RealScalar, const N: usize>(
    input: &ArrayView<'_, T, N>,
    epsilon: T,
) -> Result<Array<T, VecStorage<T>, N>> {
    let mut output = Array::from_elem(input.shape(), T::ZERO);
    l2_normalize_into(input, &mut output.view_mut(), epsilon)?;
    Ok(output)
}
