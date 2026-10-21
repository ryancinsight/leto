//! Element-assignment methods on [`Array`] and [`ArrayViewMut`].

use super::assign_into::{assign_into, assign_into_elements};
use crate::application::array::{Array, AssignSource};
use crate::application::view::ArrayViewMut;
use crate::domain::error::Result;
use crate::infrastructure::storage::StorageMut;

impl<T, S, const N: usize> Array<T, S, N>
where
    S: StorageMut<T>,
{
    /// Assign all elements from another array-like source with the same shape.
    ///
    /// Built-in array and view sources validate storage and destination
    /// storage before mutation. Dense and rank-2 transposed layouts use
    /// allocation-free bulk-copy kernels; other injective layouts use
    /// validated logical iterators. Aliased destinations and external
    /// [`AssignSource`] implementations retain the checked logical-index route.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError`](crate::domain::error::LetoError) when shapes differ,
    /// either layout exceeds its storage, or an external source rejects an index.
    #[inline]
    pub fn try_assign<Rhs>(&mut self, rhs: &Rhs) -> Result<()>
    where
        T: Copy,
        Rhs: AssignSource<T, N>,
    {
        assign_into(self.layout, self.storage.as_mut_slice(), rhs)
    }

    /// Assign all elements from another array-like source with the same shape.
    ///
    /// # Panics
    ///
    /// Panics when [`Self::try_assign`] rejects the source or destination.
    #[inline]
    pub fn assign<Rhs>(&mut self, rhs: &Rhs)
    where
        T: Copy,
        Rhs: AssignSource<T, N>,
    {
        self.try_assign(rhs)
            .expect("invariant: assigned arrays have valid compatible layouts");
    }
}

impl<T, const N: usize> ArrayViewMut<'_, T, N> {
    /// Assign all elements from another array-like source with the same shape.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError`](crate::domain::error::LetoError) when shapes differ,
    /// either layout exceeds its storage, or an external source rejects an
    /// index. Validation completes before built-in sources mutate the output.
    #[inline]
    pub fn try_assign<Rhs>(&mut self, rhs: &Rhs) -> Result<()>
    where
        T: Copy,
        Rhs: AssignSource<T, N>,
    {
        if self.window_shared {
            // Iterator-yielded interleaved sub-views must not materialize
            // their window as a slice (it contains sibling views' elements);
            // assign through checked per-element access instead.
            return assign_into_elements(self, rhs);
        }
        let layout = self.layout;
        assign_into(layout, self.data_mut(), rhs)
    }

    /// Assign all elements from another array-like source with the same shape.
    ///
    /// # Panics
    ///
    /// Panics when [`Self::try_assign`] rejects the source or destination.
    #[inline]
    pub fn assign<Rhs>(&mut self, rhs: &Rhs)
    where
        T: Copy,
        Rhs: AssignSource<T, N>,
    {
        self.try_assign(rhs)
            .expect("invariant: assigned views have valid compatible layouts");
    }
}
