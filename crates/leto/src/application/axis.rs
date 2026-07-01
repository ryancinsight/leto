use crate::application::array::Array;
use crate::application::iter::{AxisIter, AxisIterMut};
use crate::application::view::{ArrayView, ArrayViewMut};
use crate::domain::error::Result;
use crate::infrastructure::storage::{Storage, StorageMut};

impl<'a, T> ArrayView<'a, T, 2> {
    /// Return zero-copy read-only row views for a rank-2 array view.
    #[inline]
    pub fn rows(&self) -> Result<AxisIter<'a, T, 2, 1>> {
        AxisIter::new(self, 0, crate::domain::remove_axis::RankMarker::<2>)
    }

    /// Return zero-copy read-only column views for a rank-2 array view.
    #[inline]
    pub fn columns(&self) -> Result<AxisIter<'a, T, 2, 1>> {
        AxisIter::new(self, 1, crate::domain::remove_axis::RankMarker::<2>)
    }
}

impl<'a, T> ArrayViewMut<'a, T, 2> {
    /// Return zero-copy mutable row views for a rank-2 array view.
    #[inline]
    pub fn rows_mut(self) -> Result<AxisIterMut<'a, T, 2, 1>> {
        self.axis_iter_mut::<1>(0)
    }

    /// Return zero-copy mutable column views for a rank-2 array view.
    #[inline]
    pub fn columns_mut(self) -> Result<AxisIterMut<'a, T, 2, 1>> {
        self.axis_iter_mut::<1>(1)
    }
}

impl<T, S> Array<T, S, 2>
where
    S: Storage<T>,
{
    /// Return zero-copy read-only row views for a rank-2 owned array.
    #[inline]
    pub fn rows(&self) -> Result<AxisIter<'_, T, 2, 1>> {
        self.view().rows()
    }

    /// Return zero-copy read-only column views for a rank-2 owned array.
    #[inline]
    pub fn columns(&self) -> Result<AxisIter<'_, T, 2, 1>> {
        self.view().columns()
    }
}

impl<T, S> Array<T, S, 2>
where
    S: StorageMut<T>,
{
    /// Return zero-copy mutable row views for a rank-2 owned array.
    #[inline]
    pub fn rows_mut(&mut self) -> Result<AxisIterMut<'_, T, 2, 1>> {
        self.view_mut().rows_mut()
    }

    /// Return zero-copy mutable column views for a rank-2 owned array.
    #[inline]
    pub fn columns_mut(&mut self) -> Result<AxisIterMut<'_, T, 2, 1>> {
        self.view_mut().columns_mut()
    }
}

