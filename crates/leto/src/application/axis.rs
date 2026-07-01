use crate::application::array::Array;
use crate::application::iter::{AxisIter, AxisIterMut};
use crate::application::view::{ArrayView, ArrayViewMut};
use crate::domain::error::Result;
use crate::domain::layout::Layout;
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

    /// Zero-copy read-only view of row `index` (ndarray `row` parity). Shares
    /// the column stride, so it is correct for contiguous and strided layouts.
    ///
    /// # Panics
    /// If `index >= self.shape()[0]`.
    #[inline]
    #[must_use]
    pub fn row(&self, index: usize) -> ArrayView<'_, T, 1> {
        let rows = self.layout.shape[0];
        assert!(index < rows, "row index {index} out of bounds for {rows} rows");
        let cols = self.layout.shape[1];
        let [s0, s1] = self.layout.strides;
        let offset = (self.layout.offset as isize + index as isize * s0) as usize;
        ArrayView::new(Layout::new([cols], [s1], offset), self.storage.as_slice())
    }

    /// Zero-copy read-only view of column `index` (ndarray `column` parity).
    /// Carries the row stride, so column access is strided (no copy).
    ///
    /// # Panics
    /// If `index >= self.shape()[1]`.
    #[inline]
    #[must_use]
    pub fn column(&self, index: usize) -> ArrayView<'_, T, 1> {
        let cols = self.layout.shape[1];
        assert!(index < cols, "column index {index} out of bounds for {cols} columns");
        let rows = self.layout.shape[0];
        let [s0, s1] = self.layout.strides;
        let offset = (self.layout.offset as isize + index as isize * s1) as usize;
        ArrayView::new(Layout::new([rows], [s0], offset), self.storage.as_slice())
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

#[cfg(test)]
mod indexed_axis_tests {
    use crate::application::array::Array;
    use crate::infrastructure::storage::VecStorage;

    #[test]
    fn row_and_column_index_and_mutate() {
        // 2x3 row-major: [[0,1,2],[3,4,5]]
        let mut a = Array::<f64, VecStorage<f64>, 2>::from_shape_vec(
            [2, 3],
            (0..6).map(|i| i as f64).collect(),
        )
        .unwrap();
        // row 1 = [3,4,5] (contiguous)
        assert_eq!(a.row(1).iter().copied().collect::<Vec<_>>(), vec![3.0, 4.0, 5.0]);
        assert_eq!(a.row(1).as_slice(), Some(&[3.0, 4.0, 5.0][..]));
        // column 2 = [2,5] (strided)
        assert_eq!(a.column(2).iter().copied().collect::<Vec<_>>(), vec![2.0, 5.0]);
        // writes go through IndexMut
        for (c, v) in [10.0, 11.0, 12.0].into_iter().enumerate() {
            a[[0, c]] = v;
        }
        assert_eq!(a.row(0).iter().copied().collect::<Vec<_>>(), vec![10.0, 11.0, 12.0]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn row_out_of_bounds_panics() {
        let a = Array::<f64, VecStorage<f64>, 2>::from_shape_vec([2, 2], vec![0.0; 4]).unwrap();
        let _ = a.row(5);
    }
}
