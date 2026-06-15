//! Sliding-window iteration over array views (ndarray `windows` parity).
//!
//! A *window* of shape `w` is a contiguous-in-logical-index `N`-dimensional
//! subview; [`Windows`] yields every window obtained by sliding `w` one step at
//! a time along each axis. Windows are **zero-copy** — each yielded
//! [`ArrayView`] reuses the parent's strides and only shifts the offset, so no
//! element is read or copied during iteration, and overlapping windows share the
//! same backing storage through immutable borrows.
//!
//! # Theorem (window count)
//! For a parent of shape `s` and window shape `w` with `1 ≤ wᵢ ≤ sᵢ`, the number
//! of distinct window start positions along axis `i` is `cᵢ = sᵢ − wᵢ + 1`, and
//! the total number of windows is `∏ᵢ cᵢ`.
//!
//! *Proof.* A window along axis `i` is fixed by its start coordinate `tᵢ`; it
//! fits iff `tᵢ + wᵢ ≤ sᵢ`, i.e. `tᵢ ∈ {0, …, sᵢ − wᵢ}`, a set of size
//! `sᵢ − wᵢ + 1 = cᵢ`. Window starts range independently over the axes, so by the
//! product rule the total count is `∏ᵢ cᵢ`. The window at start `t` has origin
//! physical offset `o + Σᵢ tᵢ·strideᵢ` (the parent layout's `offset_of(t)`) and
//! inherits the parent strides, so it is a valid view into the same buffer. ∎

use crate::application::index::index_from_flat;
use crate::application::view::ArrayView;
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;

/// Iterator over every sliding window of a fixed shape, in row-major order of
/// window start position.
///
/// Yields `ArrayView<'a, T, N>` of shape `window_shape`. Construct via
/// [`ArrayView::windows`](crate::application::view::ArrayView::windows) or
/// [`Array::windows`](crate::application::array::Array::windows). The iterator is
/// [`ExactSizeIterator`] and [`DoubleEndedIterator`].
pub struct Windows<'a, T, const N: usize> {
    data: &'a [T],
    base_layout: Layout<N>,
    window_shape: [usize; N],
    /// Window-start counts per axis: `counts[i] = shape[i] − window[i] + 1`.
    counts: [usize; N],
    front: usize,
    back: usize,
}

impl<'a, T, const N: usize> Windows<'a, T, N> {
    /// Build a sliding-window iterator over `view` with the given window shape.
    ///
    /// # Errors
    /// [`LetoError`] if any `window_shape[i]` is `0` or exceeds the parent's
    /// extent `shape[i]`.
    pub(crate) fn new(view: &ArrayView<'a, T, N>, window_shape: [usize; N]) -> Result<Self> {
        let shape = view.shape();
        let mut counts = [0usize; N];
        for i in 0..N {
            if window_shape[i] == 0 {
                return Err(LetoError::StorageError {
                    reason: format!("window extent on axis {i} must be non-zero"),
                });
            }
            if window_shape[i] > shape[i] {
                return Err(LetoError::StorageError {
                    reason: format!(
                        "window extent {} on axis {i} exceeds array extent {}",
                        window_shape[i], shape[i]
                    ),
                });
            }
            counts[i] = shape[i] - window_shape[i] + 1;
        }
        let total: usize = counts.iter().product();
        Ok(Self {
            data: view.data(),
            base_layout: view.layout(),
            window_shape,
            counts,
            front: 0,
            back: total,
        })
    }

    /// Materialize the window whose start position has linear rank `flat`.
    #[inline]
    fn window_at(&self, flat: usize) -> ArrayView<'a, T, N> {
        let starts = index_from_flat(flat, &self.counts);
        let offset = self
            .base_layout
            .offset_of(starts)
            .expect("invariant: window start is within parent shape by construction");
        let layout = Layout::new(self.window_shape, self.base_layout.strides, offset);
        ArrayView::new(layout, self.data)
    }
}

impl<'a, T, const N: usize> Iterator for Windows<'a, T, N> {
    type Item = ArrayView<'a, T, N>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        let window = self.window_at(self.front);
        self.front += 1;
        Some(window)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;
        (remaining, Some(remaining))
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for Windows<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        Some(self.window_at(self.back))
    }
}

impl<'a, T, const N: usize> ExactSizeIterator for Windows<'a, T, N> {}
