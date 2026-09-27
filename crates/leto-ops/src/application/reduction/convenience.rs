//! Named per-strategy convenience wrappers over [`reduce_axis`]/[`reduce_all`].

use super::axis::{reduce_axis, reduce_axis_into};
use super::strategies::{MaxAxis, MeanAxis, MinAxis, ProductAxis, SumAxis};
use super::whole_array::reduce_all;
use crate::domain::scalar::Scalar;
use leto::{Array, ArrayView, ArrayViewMut, Result, VecStorage};

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

/// Product-reduce `input` along `axis`, keeping the reduced axis as length one.
#[inline]
pub fn product_axis_into<T: Scalar, const N: usize>(
    input: &ArrayView<'_, T, N>,
    axis: usize,
    output: &mut ArrayViewMut<'_, T, N>,
) -> Result<()> {
    reduce_axis_into::<ProductAxis, T, N>(input, axis, output)
}

/// Product-reduce `input` along `axis` into newly allocated C-contiguous output storage.
#[inline]
pub fn product_axis<T: Scalar, const N: usize>(
    input: &ArrayView<'_, T, N>,
    axis: usize,
) -> Result<Array<T, VecStorage<T>, N>> {
    reduce_axis::<ProductAxis, T, N>(input, axis)
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
