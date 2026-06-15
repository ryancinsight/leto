//! Lane iteration: 1-D views *along* one axis (ndarray `lanes` / `lanes_mut`
//! parity).
//!
//! A *lane* along axis `a` is the 1-D slice obtained by fixing every other
//! coordinate and letting the `a`-coordinate range over `0..shape[a]`. Lanes are
//! the dual of [`AxisIter`](crate::application::iter::AxisIter): `AxisIter` walks
//! the `shape[a]` subviews *perpendicular* to `a` (each of rank `N−1`), whereas
//! [`Lanes`] walks the `∏_{i≠a} shape[i]` slices *parallel* to `a` (each of rank
//! `1`, length `shape[a]`).
//!
//! Lanes are **zero-copy**: each yielded view reuses the parent stride along `a`
//! and only shifts its offset to the lane's origin.
//!
//! # Theorem (lanes partition the array)
//! For a layout with an injective index→offset map (no broadcast/aliasing), the
//! set of lanes along axis `a` partitions the array's elements: every element
//! lies in exactly one lane.
//!
//! *Proof.* An element has multi-index `(c, k)` where `c` is its coordinates on
//! the `N−1` complement axes and `k ∈ 0..shape[a]` its `a`-coordinate. The lane
//! with origin `c` contains exactly the elements `(c, 0), …, (c, shape[a]−1)`, so
//! each element belongs to the lane named by its own `c` and to no other (lanes
//! with distinct `c` share no multi-index). By injectivity, distinct multi-
//! indices map to distinct physical offsets, so the physical element sets of
//! distinct lanes are disjoint — which is what makes the mutable iterator sound. ∎

use crate::application::index::index_from_flat;
use crate::application::view::{ArrayView, ArrayViewMut};
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;
use crate::domain::remove_axis::RemoveAxis;

/// Build the reduced (axis-removed) complement layout plus the lane's own
/// `(length, stride)` along `axis`. Shared by [`Lanes`] and [`LanesMut`] (SSOT).
#[inline]
fn lane_geometry<R, const N: usize, const M: usize>(
    shape: [usize; N],
    strides: [isize; N],
    offset: usize,
    axis: usize,
    marker: R,
) -> Result<(Layout<M>, usize, isize)>
where
    R: RemoveAxis<N, SmallerShape = [usize; M], SmallerStrides = [isize; M]>,
{
    let reduced_shape = marker.remove_shape(shape, axis)?;
    let reduced_strides = marker.remove_strides(strides, axis)?;
    let complement = Layout::new(reduced_shape, reduced_strides, offset);
    Ok((complement, shape[axis], strides[axis]))
}

/// Iterator yielding read-only 1-D lane views along a fixed axis, in row-major
/// order of the complement (lane-origin) coordinates.
///
/// Yields `ArrayView<'a, T, 1>`. Construct via
/// [`ArrayView::lanes`](crate::application::view::ArrayView::lanes) or
/// [`Array::lanes`](crate::application::array::Array::lanes). The iterator is
/// [`ExactSizeIterator`] and [`DoubleEndedIterator`].
pub struct Lanes<'a, T, const N: usize, const M: usize> {
    data: &'a [T],
    /// Layout over the `N−1` complement axes; resolves each lane's origin offset.
    complement: Layout<M>,
    axis_len: usize,
    axis_stride: isize,
    front: usize,
    back: usize,
}

impl<'a, T, const N: usize, const M: usize> Lanes<'a, T, N, M> {
    /// Build a lane iterator over `view` along `axis`.
    ///
    /// # Errors
    /// [`LetoError`] if `axis >= N` or the layout does not fit its storage.
    pub(crate) fn new<R>(view: &ArrayView<'a, T, N>, axis: usize, marker: R) -> Result<Self>
    where
        R: RemoveAxis<N, SmallerShape = [usize; M], SmallerStrides = [isize; M]>,
    {
        if axis >= N {
            return Err(LetoError::StorageError {
                reason: format!("Axis {axis} out of bounds for rank {N}"),
            });
        }
        view.layout().validate_storage_len(view.data().len())?;
        let (complement, axis_len, axis_stride) =
            lane_geometry(view.shape(), view.strides(), view.offset(), axis, marker)?;
        let back = complement.size();
        Ok(Self {
            data: view.data(),
            complement,
            axis_len,
            axis_stride,
            front: 0,
            back,
        })
    }

    /// Materialize the lane whose origin has linear rank `flat`.
    #[inline]
    fn lane_at(&self, flat: usize) -> ArrayView<'a, T, 1> {
        let origin = index_from_flat(flat, &self.complement.shape);
        let offset = self
            .complement
            .offset_of(origin)
            .expect("invariant: complement index is within the complement shape");
        let layout = Layout::new([self.axis_len], [self.axis_stride], offset);
        ArrayView::new(layout, self.data)
    }
}

impl<'a, T, const N: usize, const M: usize> Iterator for Lanes<'a, T, N, M> {
    type Item = ArrayView<'a, T, 1>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        let lane = self.lane_at(self.front);
        self.front += 1;
        Some(lane)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;
        (remaining, Some(remaining))
    }
}

impl<'a, T, const N: usize, const M: usize> DoubleEndedIterator for Lanes<'a, T, N, M> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        Some(self.lane_at(self.back))
    }
}

impl<'a, T, const N: usize, const M: usize> ExactSizeIterator for Lanes<'a, T, N, M> {}

/// Iterator yielding mutable 1-D lane views along a fixed axis.
///
/// Yields `ArrayViewMut<'a, T, 1>`. Construct via
/// [`ArrayViewMut::lanes_mut`](crate::application::view::ArrayViewMut::lanes_mut)
/// or [`Array::lanes_mut`](crate::application::array::Array::lanes_mut). Forward
/// and [`ExactSizeIterator`] only.
pub struct LanesMut<'a, T, const N: usize, const M: usize> {
    ptr: *mut T,
    data_len: usize,
    complement: Layout<M>,
    axis_len: usize,
    axis_stride: isize,
    front: usize,
    back: usize,
    _marker: std::marker::PhantomData<&'a mut [T]>,
}

impl<'a, T, const N: usize, const M: usize> LanesMut<'a, T, N, M> {
    /// Build a mutable lane iterator over `view` along `axis`.
    ///
    /// # Errors
    /// [`LetoError`] if `axis >= N`, the layout does not fit its storage, or the
    /// layout aliases (a zero stride), which would make distinct lanes overlap.
    pub(crate) fn new<R>(mut view: ArrayViewMut<'a, T, N>, axis: usize, marker: R) -> Result<Self>
    where
        R: RemoveAxis<N, SmallerShape = [usize; M], SmallerStrides = [isize; M]>,
    {
        if axis >= N {
            return Err(LetoError::StorageError {
                reason: format!("Axis {axis} out of bounds for rank {N}"),
            });
        }
        let data_len = view.data().len();
        view.layout().validate_storage_len(data_len)?;
        if view.layout().has_zero_stride_aliasing() {
            return Err(LetoError::StorageError {
                reason: "lane mutable iterator requires non-aliasing layout".to_string(),
            });
        }
        let (complement, axis_len, axis_stride) =
            lane_geometry(view.shape(), view.strides(), view.offset(), axis, marker)?;
        let back = complement.size();
        let ptr = view.data_mut().as_mut_ptr();
        Ok(Self {
            ptr,
            data_len,
            complement,
            axis_len,
            axis_stride,
            front: 0,
            back,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<'a, T, const N: usize, const M: usize> Iterator for LanesMut<'a, T, N, M> {
    type Item = ArrayViewMut<'a, T, 1>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        let origin = index_from_flat(self.front, &self.complement.shape);
        let offset = self
            .complement
            .offset_of(origin)
            .expect("invariant: complement index is within the complement shape");
        self.front += 1;
        let layout = Layout::new([self.axis_len], [self.axis_stride], offset);

        // SAFETY: distinct lanes have distinct complement origins; under the
        // non-aliasing layout validated in `new`, the index→offset map is
        // injective, so the physical element sets of distinct lanes are disjoint
        // (see the partition theorem). Each yielded mutable view therefore
        // borrows a region no other yielded view touches.
        unsafe {
            let slice = std::slice::from_raw_parts_mut(self.ptr, self.data_len);
            Some(ArrayViewMut::new(layout, slice))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;
        (remaining, Some(remaining))
    }
}

impl<'a, T, const N: usize, const M: usize> ExactSizeIterator for LanesMut<'a, T, N, M> {}
