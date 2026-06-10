use crate::application::index::index_from_flat;
use crate::domain::real::RealScalar;
use leto::{ArrayView, Result};

/// Zero-sized norm-kind contract: an accumulation step and a finishing map.
///
/// One generic [`norm`] traversal monomorphizes per `(K, T)`; each kind
/// contributes only its fold step and post-reduction map, mirroring the
/// `BinaryOp`/`UnaryOp`/`ScanOp` ZST pattern used across leto-ops.
///
/// Elementwise norms are traversal-order independent, so the contiguous fast
/// path may run in physical memory order over any dense block.
pub trait NormKind<T: RealScalar>: Copy + Send + Sync + 'static {
    /// Fold one element's contribution into the accumulator.
    fn accumulate(acc: T, x: T) -> T;
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
    #[inline(always)]
    fn accumulate(acc: T, x: T) -> T {
        acc.add(x.abs())
    }
    #[inline(always)]
    fn finish(acc: T) -> T {
        acc
    }
}

impl<T: RealScalar> NormKind<T> for NormL2 {
    #[inline(always)]
    fn accumulate(acc: T, x: T) -> T {
        acc.add(x.mul(x))
    }
    #[inline(always)]
    fn finish(acc: T) -> T {
        acc.sqrt()
    }
}

impl<T: RealScalar> NormKind<T> for NormMax {
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
    for flat_idx in 0..size {
        let index = index_from_flat(flat_idx, &shape);
        let offset = layout.offset_of(index)?;
        acc = K::accumulate(acc, data[offset]);
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
