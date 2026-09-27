#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use super::types::Array;
use crate::application::iter::{ElementIterMut, IndexedIterMut, LanesMut, TaskPartitionsMut};
use crate::application::view::ArrayViewMut;
use crate::domain::error::Result;
use crate::infrastructure::storage::StorageMut;

impl<T, S, const N: usize> Array<T, S, N>
where
    S: StorageMut<T>,
{
    /// Returns a mutable view of this array.
    #[inline]
    pub fn view_mut(&mut self) -> ArrayViewMut<'_, T, N> {
        ArrayViewMut::new(self.layout, self.storage.as_mut_slice())
    }

    /// Fill every physical storage slot with `value`.
    #[inline]
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.storage.as_mut_slice().fill(value);
    }

    /// Iterator over the array's elements as mutable references, in logical
    /// row-major order (`leto iter_mut` parity).
    ///
    /// # Panics
    /// Panics if the layout is not C-contiguous.
    #[inline]
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.as_slice_mut()
            .expect("iter_mut: array must be C-contiguous")
            .iter_mut()
    }

    /// Iterator over `(multi-index, &mut element)` pairs in logical row-major
    /// order (leto `indexed_iter_mut` parity).
    ///
    /// # Errors
    /// Returns [`LetoError`](crate::domain::error::LetoError) if the layout is
    /// out of bounds or cannot prove that each logical index addresses a
    /// distinct physical element.
    #[inline]
    pub fn indexed_iter_mut(&mut self) -> Result<IndexedIterMut<'_, T, N>> {
        self.view_mut().indexed_iter_mut()
    }

    /// Iterator over mutable elements in logical row-major order, preserving
    /// arbitrary positive and negative strides without materializing a copy.
    ///
    /// # Errors
    /// Returns [`LetoError`](crate::domain::error::LetoError) if the layout is
    /// out of bounds or cannot prove that each logical index addresses a
    /// distinct physical element. In particular, zero-stride broadcast layouts
    /// are rejected before any mutable reference is yielded.
    #[inline]
    pub fn try_iter_mut(&mut self) -> Result<ElementIterMut<'_, T, N>> {
        self.view_mut().try_iter_mut()
    }

    /// Split the logical row-major domain into disjoint mutable task partitions.
    ///
    /// The partition iterator validates the complete layout once and leaves
    /// scheduling to the caller. Each partition is a move-only range token
    /// suitable for a scoped execution provider.
    ///
    /// # Errors
    /// Returns [`LetoError`](crate::domain::error::LetoError) when
    /// `chunk_size` is zero, storage is invalid, or the layout is not provably
    /// injective.
    #[inline]
    pub fn task_partitions_mut(
        &mut self,
        chunk_size: usize,
    ) -> Result<TaskPartitionsMut<'_, T, N>> {
        self.view_mut().task_partitions_mut(chunk_size)
    }

    /// The elements as one mutable contiguous slice in logical row-major order,
    /// or `None` if the array is not C-contiguous (leto `as_slice_mut`
    /// parity). The safe basis for in-place element iteration: `if let Some(s) =
    /// a.as_slice_mut() { for x in s.iter_mut() { … } }`.
    #[inline]
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        if self.layout.is_c_dense() {
            let start = self.layout.offset();
            let end = start.checked_add(self.layout.checked_size().ok()?)?;
            self.storage.as_mut_slice().get_mut(start..end)
        } else {
            None
        }
    }

    /// Expose the mutable dense physical-memory slice when this array is contiguous.
    #[inline]
    pub fn as_slice_memory_order_mut(&mut self) -> Option<&mut [T]> {
        if self.layout.is_contiguous() {
            let start = self.layout.offset();
            let end = start.checked_add(self.layout.checked_size().ok()?)?;
            self.storage.as_mut_slice().get_mut(start..end)
        } else {
            None
        }
    }

    /// Zero-copy iterator over the mutable 1-D lanes along `axis`
    /// (leto `lanes_mut` parity; `M = N - 1`).
    ///
    /// # Errors
    /// [`LetoError`](crate::domain::error::LetoError) if `axis >= N`, the
    /// layout does not fit its storage, or the layout aliases (a zero
    /// stride).
    #[inline]
    pub fn lanes_mut<const M: usize>(&mut self, axis: usize) -> Result<LanesMut<'_, T, N, M>>
    where
        crate::domain::remove_axis::RankMarker<N>: crate::domain::remove_axis::RemoveAxis<
            N,
            SmallerShape = [usize; M],
            SmallerStrides = [isize; M],
        >,
    {
        self.view_mut().lanes_mut(axis)
    }

    /// Slice the array, returning a mutable view.
    #[inline]
    pub fn slice_mut(
        &mut self,
        ranges: &[(usize, usize, isize); N],
    ) -> Result<ArrayViewMut<'_, T, N>> {
        self.view_mut().slice_mut(ranges)
    }

    /// Slice the array with leto-style arguments, returning a mutable view.
    #[inline]
    pub fn slice_with_mut<const M: usize>(
        &mut self,
        args: &[crate::domain::slice::SliceArg],
    ) -> Result<ArrayViewMut<'_, T, M>> {
        self.view_mut().slice_with_mut(args)
    }

    /// Fix one axis at `index`, reducing the rank by 1 (leto `index_axis_mut` parity).
    ///
    /// `M` must equal `N - 1`; a mismatch returns `LetoError` from `slice_with_mut`.
    #[inline]
    pub fn index_axis_mut<const M: usize>(
        &mut self,
        axis: usize,
        index: usize,
    ) -> Result<ArrayViewMut<'_, T, M>> {
        let args: Vec<crate::domain::slice::SliceArg> = (0..N)
            .map(|i| {
                if i == axis {
                    crate::domain::slice::SliceArg::Index(index as isize)
                } else {
                    crate::domain::slice::SliceArg::All
                }
            })
            .collect();
        self.slice_with_mut::<M>(&args)
    }

    /// Transpose the array, returning a mutable view.
    #[inline]
    pub fn transpose_mut(&mut self, axes: [usize; N]) -> Result<ArrayViewMut<'_, T, N>> {
        self.view_mut().transpose_mut(axes)
    }

    /// Reinterpret this mutable array with a new shape without copying.
    ///
    /// The current layout must be dense row-major and the new shape must have
    /// the same logical element count.
    #[inline]
    pub fn reshape_mut<const M: usize>(
        &mut self,
        shape: [usize; M],
    ) -> Result<ArrayViewMut<'_, T, M>> {
        self.view_mut().reshape_mut(shape)
    }

    /// Named mutable alias for [`transpose_mut`](Self::transpose_mut).
    #[inline]
    pub fn permute_mut(&mut self, axes: [usize; N]) -> Result<ArrayViewMut<'_, T, N>> {
        self.transpose_mut(axes)
    }

    /// Get a mutable reference to the element at the specified index.
    #[inline]
    pub fn get_mut(&mut self, index: [usize; N]) -> Result<&mut T> {
        let offset = self.layout.offset_of(index)?;
        let slice = self.storage.as_mut_slice();
        if offset >= slice.len() {
            return Err(crate::domain::error::LetoError::StorageError {
                reason: format!(
                    "physical offset {offset} exceeds backing slice length {}",
                    slice.len()
                ),
            });
        }
        Ok(&mut slice[offset])
    }

    /// Add `alpha * rhs` to `self` in place.
    ///
    /// # Panics
    /// Panics when the shapes differ.
    #[inline]
    pub fn scaled_add<S2>(&mut self, alpha: T, rhs: &Array<T, S2, N>)
    where
        T: Copy + core::ops::Add<Output = T> + core::ops::Mul<Output = T>,
        S2: crate::infrastructure::storage::Storage<T>,
    {
        let shape = self.shape();
        assert_eq!(
            shape,
            rhs.shape(),
            "scaled_add requires matching shapes: lhs {:?}, rhs {:?}",
            shape,
            rhs.shape()
        );
        // Matching dense memory order (both C or both F with equal shapes
        // implies identical strides): one zipped slice pass replaces the
        // per-element odometer with its three checked offset computations.
        {
            let rhs_view = rhs.view();
            let mut lhs_view = self.view_mut();
            let same_order = (lhs_view.is_c_dense() && rhs_view.is_c_dense())
                || (lhs_view.is_f_dense() && rhs_view.is_f_dense());
            if same_order {
                if let (Some(dst), Some(src)) = (
                    lhs_view.as_mut_slice_memory_order(),
                    rhs_view.as_slice_memory_order(),
                ) {
                    for (target, source) in dst.iter_mut().zip(src) {
                        *target = *target + alpha * *source;
                    }
                    return;
                }
            }
        }
        for linear in 0..self.size() {
            let index = linear_to_index(linear, shape);
            let scaled = alpha
                * *rhs
                    .get(index)
                    .expect("invariant: rhs logical index is in bounds");
            let target = self
                .get_mut(index)
                .expect("invariant: logical index is in bounds");
            *target = *target + scaled;
        }
    }
}

pub(crate) fn linear_to_index<const N: usize>(mut linear: usize, shape: [usize; N]) -> [usize; N] {
    let mut index = [0; N];
    for axis in (0..N).rev() {
        let extent = shape[axis];
        if extent != 0 {
            index[axis] = linear % extent;
            linear /= extent;
        }
    }
    index
}

#[cfg(test)]
mod scaled_add_tests {
    use crate::application::array::Array;
    use crate::infrastructure::storage::VecStorage;

    fn array(data: Vec<f64>) -> Array<f64, VecStorage<f64>, 2> {
        Array::<f64, VecStorage<f64>, 2>::from_shape_vec([2, 2], data).unwrap()
    }

    #[test]
    fn scaled_add_accumulates_scaled_source() {
        let mut dst = array(vec![1.0, 2.0, 3.0, 4.0]);
        let src = array(vec![10.0, 20.0, 30.0, 40.0]);

        dst.scaled_add(0.5, &src);

        assert_eq!(
            dst.iter().copied().collect::<Vec<_>>(),
            vec![6.0, 12.0, 18.0, 24.0]
        );
    }

    #[test]
    #[should_panic(expected = "matching shapes")]
    fn scaled_add_shape_mismatch_panics() {
        let mut dst = array(vec![0.0; 4]);
        let src = Array::<f64, VecStorage<f64>, 2>::from_shape_vec([4, 1], vec![0.0; 4]).unwrap();

        dst.scaled_add(1.0, &src);
    }
}

#[cfg(test)]
mod slice_access_tests {
    use crate::application::array::Array;
    use crate::infrastructure::storage::VecStorage;

    #[test]
    fn as_slice_and_as_slice_mut_on_contiguous() {
        let mut a = Array::<f64, VecStorage<f64>, 2>::from_shape_vec(
            [2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        assert_eq!(a.as_slice(), Some(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0][..]));
        // mutate every element in place through the mutable slice (the iter_mut basis)
        for (i, x) in a.as_slice_mut().unwrap().iter_mut().enumerate() {
            *x = i as f64 * 10.0;
        }
        assert_eq!(a.as_slice(), Some(&[0.0, 10.0, 20.0, 30.0, 40.0, 50.0][..]));
    }
}
