//! Sliding-window iteration over array views (leto `windows` parity).
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

use crate::application::view::ArrayView;
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;

#[inline]
fn odometer_step<const N: usize>(
    index: &mut [usize; N],
    shape: &[usize; N],
    strides: &[isize; N],
    offset: &mut usize,
) {
    for i in (0..N).rev() {
        index[i] += 1;
        if index[i] < shape[i] {
            *offset = (*offset as isize + strides[i]) as usize;
            break;
        }
        *offset = (*offset as isize - (shape[i] - 1) as isize * strides[i]) as usize;
        index[i] = 0;
    }
}

#[inline]
fn odometer_step_back<const N: usize>(
    index: &mut [usize; N],
    shape: &[usize; N],
    strides: &[isize; N],
    offset: &mut usize,
) {
    for i in (0..N).rev() {
        if index[i] > 0 {
            index[i] -= 1;
            *offset = (*offset as isize - strides[i]) as usize;
            break;
        }
        *offset = (*offset as isize + (shape[i] - 1) as isize * strides[i]) as usize;
        index[i] = shape[i] - 1;
    }
}

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
    front_index: [usize; N],
    front_offset: usize,
    back_index: [usize; N],
    back_offset: usize,
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
        let (back_index, back_offset) = if total > 0 {
            let mut idx = [0usize; N];
            for (i, item) in idx.iter_mut().enumerate() {
                *item = counts[i] - 1;
            }
            let offset = view
                .layout()
                .offset_of(idx)
                .expect("invariant: last window start is valid");
            (idx, offset)
        } else {
            ([0usize; N], view.layout().offset())
        };
        Ok(Self {
            data: view.data(),
            base_layout: view.layout(),
            window_shape,
            counts,
            front: 0,
            back: total,
            front_index: [0usize; N],
            front_offset: view.layout().offset(),
            back_index,
            back_offset,
        })
    }
}

impl<'a, T, const N: usize> Iterator for Windows<'a, T, N> {
    type Item = ArrayView<'a, T, N>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        let layout = Layout::from_parts_unchecked(
            self.window_shape,
            self.base_layout.strides(),
            self.front_offset,
        );
        odometer_step(
            &mut self.front_index,
            &self.counts,
            &self.base_layout.strides(),
            &mut self.front_offset,
        );
        self.front += 1;
        Some(ArrayView::new(layout, self.data))
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
        let layout = Layout::from_parts_unchecked(
            self.window_shape,
            self.base_layout.strides(),
            self.back_offset,
        );
        odometer_step_back(
            &mut self.back_index,
            &self.counts,
            &self.base_layout.strides(),
            &mut self.back_offset,
        );
        Some(ArrayView::new(layout, self.data))
    }
}

impl<'a, T, const N: usize> ExactSizeIterator for Windows<'a, T, N> {}
