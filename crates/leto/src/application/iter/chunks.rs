//! Non-overlapping chunk iteration over array views.
//!
//! An exact chunk of shape `c` is a zero-copy subview whose origin lies on the
//! chunk grid `kᵢ·cᵢ`. Remainders are skipped: only starts satisfying
//! `(kᵢ + 1)·cᵢ ≤ sᵢ` are yielded. Each chunk inherits the parent strides and
//! backing storage, so traversal streams block views without copying elements.
//!
//! # Theorem (exact chunk count)
//! For parent shape `s` and chunk shape `c` with every `cᵢ > 0`, the number of
//! chunks along axis `i` is `qᵢ = floor(sᵢ / cᵢ)`, and the total number of
//! yielded chunks is `∏ᵢ qᵢ`.
//!
//! *Proof.* A chunk start on axis `i` has form `kᵢ·cᵢ`. The whole chunk fits iff
//! `kᵢ·cᵢ + cᵢ ≤ sᵢ`, equivalent to `kᵢ < sᵢ / cᵢ`. Since `kᵢ` is an integer,
//! there are `floor(sᵢ / cᵢ)` valid starts. Axes are independent, so the product
//! rule gives `∏ᵢ qᵢ`. The chunk at grid coordinate `k` has physical origin
//! `offset_of(k * c)` and reuses the parent strides, so the yielded value is a
//! view into the original buffer. ∎

use crate::application::view::ArrayView;
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;

fn index_from_linear<const N: usize>(mut linear: usize, shape: [usize; N]) -> [usize; N] {
    let mut index = [0usize; N];
    for axis in (0..N).rev() {
        let extent = shape[axis];
        if extent != 0 {
            index[axis] = linear % extent;
            linear /= extent;
        }
    }
    index
}

fn chunk_origin<const N: usize>(
    chunk_index: [usize; N],
    chunk_shape: [usize; N],
) -> Result<[usize; N]> {
    let mut origin = [0usize; N];
    for axis in 0..N {
        origin[axis] =
            chunk_index[axis]
                .checked_mul(chunk_shape[axis])
                .ok_or(LetoError::Overflow {
                    reason: "exact chunk origin calculation",
                })?;
    }
    Ok(origin)
}

/// Iterator over non-overlapping exact chunks in row-major chunk-grid order.
///
/// Yields `ArrayView<'a, T, N>` values of exactly `chunk_shape`. Construct via
/// [`ArrayView::exact_chunks`](crate::application::view::ArrayView::exact_chunks)
/// or [`Array::exact_chunks`](crate::application::array::Array::exact_chunks).
pub struct ExactChunks<'a, T, const N: usize> {
    data: &'a [T],
    base_layout: Layout<N>,
    chunk_shape: [usize; N],
    chunk_counts: [usize; N],
    front: usize,
    back: usize,
}

impl<'a, T, const N: usize> ExactChunks<'a, T, N> {
    /// Build an exact chunk iterator over `view`.
    ///
    /// # Errors
    /// [`LetoError`] if any chunk extent is zero or the chunk grid overflows
    /// `usize`.
    pub(crate) fn new(view: &ArrayView<'a, T, N>, chunk_shape: [usize; N]) -> Result<Self> {
        let shape = view.shape();
        let mut chunk_counts = [0usize; N];
        let mut total = 1usize;
        for axis in 0..N {
            if chunk_shape[axis] == 0 {
                return Err(LetoError::StorageError {
                    reason: format!("exact chunk extent on axis {axis} must be non-zero"),
                });
            }
            chunk_counts[axis] = shape[axis] / chunk_shape[axis];
            total = total
                .checked_mul(chunk_counts[axis])
                .ok_or(LetoError::Overflow {
                    reason: "exact chunk count calculation",
                })?;
        }
        view.layout().validate_storage_len(view.data().len())?;

        Ok(Self {
            data: view.data(),
            base_layout: view.layout(),
            chunk_shape,
            chunk_counts,
            front: 0,
            back: total,
        })
    }

    #[inline]
    fn view_at(&self, linear: usize) -> ArrayView<'a, T, N> {
        let chunk_index = index_from_linear(linear, self.chunk_counts);
        let origin = chunk_origin(chunk_index, self.chunk_shape)
            .expect("invariant: chunk origin is inside validated chunk grid");
        let offset = self
            .base_layout
            .offset_of(origin)
            .expect("invariant: chunk origin is inside parent shape");
        let layout = Layout::new(self.chunk_shape, self.base_layout.strides, offset);
        ArrayView::new(layout, self.data)
    }
}

impl<'a, T, const N: usize> Iterator for ExactChunks<'a, T, N> {
    type Item = ArrayView<'a, T, N>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        let view = self.view_at(self.front);
        self.front += 1;
        Some(view)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;
        (remaining, Some(remaining))
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for ExactChunks<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        Some(self.view_at(self.back))
    }
}

impl<'a, T, const N: usize> ExactSizeIterator for ExactChunks<'a, T, N> {}

/// Iterator over axis-aligned chunks in row-major axis-chunk order.
///
/// Yields `ArrayView<'a, T, N>` values whose shape matches the parent view
/// except along `axis`, where each full chunk has extent `chunk_len` and the
/// final chunk carries the remainder when present. Construct via
/// [`ArrayView::axis_chunks_iter`](crate::application::view::ArrayView::axis_chunks_iter)
/// or [`Array::axis_chunks_iter`](crate::application::array::Array::axis_chunks_iter).
///
/// # Theorem (axis chunk coverage)
/// For axis length `s` and chunk length `c > 0`, the iterator yields
/// `ceil(s / c)` chunks. Chunk `k` covers logical coordinates
/// `k·c..min((k + 1)·c, s)` on the selected axis and the full extent of every
/// other axis. These intervals are disjoint and their union is `0..s`, so every
/// logical element appears in exactly one yielded view.
pub struct AxisChunks<'a, T, const N: usize> {
    data: &'a [T],
    base_layout: Layout<N>,
    axis: usize,
    chunk_len: usize,
    front: usize,
    back: usize,
}

impl<'a, T, const N: usize> AxisChunks<'a, T, N> {
    /// Build an axis chunk iterator over `view`.
    ///
    /// # Errors
    /// [`LetoError`] if `axis` is out of bounds or `chunk_len` is zero.
    pub(crate) fn new(view: &ArrayView<'a, T, N>, axis: usize, chunk_len: usize) -> Result<Self> {
        if axis >= N {
            return Err(LetoError::ShapeMismatch {
                lhs: vec![N],
                rhs: vec![axis],
            });
        }
        if chunk_len == 0 {
            return Err(LetoError::StorageError {
                reason: format!("axis chunk length on axis {axis} must be non-zero"),
            });
        }

        view.layout().validate_storage_len(view.data().len())?;
        let axis_len = view.shape()[axis];
        let chunk_count = axis_len.div_ceil(chunk_len);

        Ok(Self {
            data: view.data(),
            base_layout: view.layout(),
            axis,
            chunk_len,
            front: 0,
            back: chunk_count,
        })
    }

    #[inline]
    fn view_at(&self, chunk_index: usize) -> ArrayView<'a, T, N> {
        let start = chunk_index
            .checked_mul(self.chunk_len)
            .expect("invariant: axis chunk start fits in usize");
        let mut origin = [0usize; N];
        origin[self.axis] = start;
        let offset = self
            .base_layout
            .offset_of(origin)
            .expect("invariant: axis chunk origin is inside parent shape");

        let mut shape = self.base_layout.shape;
        shape[self.axis] = self
            .chunk_len
            .min(self.base_layout.shape[self.axis] - start);
        let layout = Layout::new(shape, self.base_layout.strides, offset);
        ArrayView::new(layout, self.data)
    }
}

impl<'a, T, const N: usize> Iterator for AxisChunks<'a, T, N> {
    type Item = ArrayView<'a, T, N>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        let view = self.view_at(self.front);
        self.front += 1;
        Some(view)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;
        (remaining, Some(remaining))
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for AxisChunks<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        Some(self.view_at(self.back))
    }
}

impl<'a, T, const N: usize> ExactSizeIterator for AxisChunks<'a, T, N> {}
