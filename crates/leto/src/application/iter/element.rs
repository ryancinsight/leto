//! Logical-order element iteration over array views (leto `iter` /
//! `indexed_iter` / `indexed_iter_mut` parity).
//!
//! Both iterators walk every logical element of a view in row-major order,
//! resolving each element's physical offset through the view's strides — so an
//! arbitrarily strided, transposed, or broadcast view iterates in the same
//! logical order as its contiguous materialization. They are
//! [`ExactSizeIterator`] and [`DoubleEndedIterator`]; the two ends share one
//! `[front, back)` cursor so forward and backward consumption meet exactly once.

use crate::application::array::Array;
use crate::application::index::index_from_flat;
use crate::application::view::{ArrayView, ArrayViewMut};
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;
use crate::infrastructure::storage::Storage;

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

fn layout_may_alias_mutable_offsets<const N: usize>(layout: &Layout<N>) -> Result<bool> {
    Ok(!layout.is_injective()?)
}

/// Private proof token for one bounds-valid, injective mutable layout.
///
/// The token is created once at the mutable partition boundary. It is not
/// exposed as a view or slice, so partition construction cannot duplicate the
/// raw layout proof outside this module.
struct MutableLayoutProof<'a, T, const N: usize> {
    ptr: std::ptr::NonNull<T>,
    layout: Layout<N>,
    storage_len: usize,
    _marker: std::marker::PhantomData<&'a mut [T]>,
}

impl<'a, T, const N: usize> MutableLayoutProof<'a, T, N> {
    fn new(view: ArrayViewMut<'a, T, N>, aliasing_reason: &'static str) -> Result<Self> {
        view.layout.validate_storage_len(view.len)?;
        if layout_may_alias_mutable_offsets(&view.layout)? {
            return Err(LetoError::StorageError {
                reason: aliasing_reason.to_string(),
            });
        }
        Ok(Self {
            ptr: view.ptr,
            layout: view.layout,
            storage_len: view.len,
            _marker: std::marker::PhantomData,
        })
    }
}

/// Iterator over every element of a view in logical row-major order.
///
/// Yields `&T`. Construct via [`ArrayView::iter`](crate::application::view::ArrayView::iter)
/// or [`Array::iter`](crate::application::array::Array::iter).
pub struct ElementIter<'a, T, const N: usize> {
    contiguous_iter: Option<std::slice::Iter<'a, T>>,
    data: &'a [T],
    layout: Layout<N>,
    shape: [usize; N],
    front: usize,
    back: usize,
    front_index: [usize; N],
    front_offset: usize,
    back_index: [usize; N],
    back_offset: usize,
}

impl<'a, T, const N: usize> ElementIter<'a, T, N> {
    /// Build an element iterator over `view`.
    #[inline]
    pub(crate) fn new(view: &ArrayView<'a, T, N>) -> Self {
        let layout = view.layout();
        let contiguous_iter = if layout.is_c_dense() {
            let start = layout.offset;
            let end = start + layout.size();
            Some(view.data()[start..end].iter())
        } else {
            None
        };
        let back = view.size();
        let (back_index, back_offset) = if back > 0 {
            let mut idx = [0usize; N];
            for (i, item) in idx.iter_mut().enumerate() {
                *item = layout.shape[i] - 1;
            }
            let offset = layout
                .offset_of(idx)
                .expect("invariant: last index is valid");
            (idx, offset)
        } else {
            ([0usize; N], layout.offset)
        };
        Self {
            contiguous_iter,
            data: view.data(),
            layout,
            shape: layout.shape,
            front: 0,
            back,
            front_index: [0usize; N],
            front_offset: layout.offset,
            back_index,
            back_offset,
        }
    }
}

impl<'a, T, const N: usize> Iterator for ElementIter<'a, T, N> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        if let Some(ref mut iter) = self.contiguous_iter {
            iter.next()
        } else {
            if self.front >= self.back {
                return None;
            }
            let elem = &self.data[self.front_offset];
            odometer_step(
                &mut self.front_index,
                &self.shape,
                &self.layout.strides,
                &mut self.front_offset,
            );
            self.front += 1;
            Some(elem)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if let Some(ref iter) = self.contiguous_iter {
            iter.size_hint()
        } else {
            let remaining = self.back - self.front;
            (remaining, Some(remaining))
        }
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for ElementIter<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        if let Some(ref mut iter) = self.contiguous_iter {
            iter.next_back()
        } else {
            if self.front >= self.back {
                return None;
            }
            self.back -= 1;
            let elem = &self.data[self.back_offset];
            odometer_step_back(
                &mut self.back_index,
                &self.shape,
                &self.layout.strides,
                &mut self.back_offset,
            );
            Some(elem)
        }
    }
}

impl<'a, T, const N: usize> ExactSizeIterator for ElementIter<'a, T, N> {}

/// Iterator over `(multi-index, element)` pairs in logical row-major order.
///
/// Yields `([usize; N], &T)`. Construct via
/// [`ArrayView::indexed_iter`](crate::application::view::ArrayView::indexed_iter)
/// or [`Array::indexed_iter`](crate::application::array::Array::indexed_iter).
pub struct IndexedIter<'a, T, const N: usize> {
    data: &'a [T],
    layout: Layout<N>,
    shape: [usize; N],
    front: usize,
    back: usize,
    front_index: [usize; N],
    front_offset: usize,
    back_index: [usize; N],
    back_offset: usize,
}

impl<'a, T, const N: usize> IndexedIter<'a, T, N> {
    /// Build an indexed iterator over `view`.
    #[inline]
    pub(crate) fn new(view: &ArrayView<'a, T, N>) -> Self {
        let layout = view.layout();
        let back = view.size();
        let (back_index, back_offset) = if back > 0 {
            let mut idx = [0usize; N];
            for (i, item) in idx.iter_mut().enumerate() {
                *item = layout.shape[i] - 1;
            }
            let offset = layout
                .offset_of(idx)
                .expect("invariant: last index is valid");
            (idx, offset)
        } else {
            ([0usize; N], layout.offset)
        };
        Self {
            data: view.data(),
            layout,
            shape: layout.shape,
            front: 0,
            back,
            front_index: [0usize; N],
            front_offset: layout.offset,
            back_index,
            back_offset,
        }
    }
}

impl<'a, T, const N: usize> Iterator for IndexedIter<'a, T, N> {
    type Item = ([usize; N], &'a T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        let index = self.front_index;
        let elem = &self.data[self.front_offset];
        odometer_step(
            &mut self.front_index,
            &self.shape,
            &self.layout.strides,
            &mut self.front_offset,
        );
        self.front += 1;
        Some((index, elem))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;
        (remaining, Some(remaining))
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for IndexedIter<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        let index = self.back_index;
        let elem = &self.data[self.back_offset];
        odometer_step_back(
            &mut self.back_index,
            &self.shape,
            &self.layout.strides,
            &mut self.back_offset,
        );
        Some((index, elem))
    }
}

impl<'a, T, const N: usize> ExactSizeIterator for IndexedIter<'a, T, N> {}

/// Mutable iterator over `(multi-index, element)` pairs in logical row-major order.
///
/// Yields `([usize; N], &mut T)`. Construct via
/// [`ArrayViewMut::indexed_iter_mut`](crate::application::view::ArrayViewMut::indexed_iter_mut)
/// or [`Array::indexed_iter_mut`](crate::application::array::Array::indexed_iter_mut).
pub struct IndexedIterMut<'a, T, const N: usize> {
    ptr: std::ptr::NonNull<T>,
    layout: Layout<N>,
    shape: [usize; N],
    front: usize,
    back: usize,
    front_index: [usize; N],
    front_offset: usize,
    back_index: [usize; N],
    back_offset: usize,
    _marker: std::marker::PhantomData<&'a mut [T]>,
}

impl<'a, T, const N: usize> IndexedIterMut<'a, T, N> {
    /// Build a mutable indexed iterator over the complete logical view.
    pub(crate) fn new(view: ArrayViewMut<'a, T, N>) -> Result<Self> {
        let end = view.layout.size();
        Self::new_range(view, 0, end)
    }

    /// Build a mutable indexed iterator over a validated logical range.
    ///
    /// The range is expressed in row-major logical positions, not physical
    /// storage offsets. It is the primitive used by [`TaskPartitionsMut`].
    pub(crate) fn new_range(
        view: ArrayViewMut<'a, T, N>,
        start: usize,
        end: usize,
    ) -> Result<Self> {
        Self::from_proof(
            MutableLayoutProof::new(
                view,
                "indexed_iter_mut requires provably disjoint logical offsets",
            )?,
            start,
            end,
        )
    }

    fn from_proof(proof: MutableLayoutProof<'a, T, N>, start: usize, end: usize) -> Result<Self> {
        let MutableLayoutProof {
            ptr,
            layout,
            storage_len: _,
            _marker: _,
        } = proof;
        let size = layout.size();
        if start > end || end > size {
            return Err(LetoError::OutOfBounds {
                index: vec![start, end],
                shape: vec![size],
            });
        }
        let (front_index, front_offset, back_index, back_offset) = if start < end {
            let front_index = index_from_flat(start, &layout.shape);
            let back_index = index_from_flat(end - 1, &layout.shape);
            let front_offset = layout
                .offset_of(front_index)
                .expect("invariant: range start is a valid logical index");
            let back_offset = layout
                .offset_of(back_index)
                .expect("invariant: range end is a valid logical index");
            (front_index, front_offset, back_index, back_offset)
        } else {
            ([0usize; N], layout.offset, [0usize; N], layout.offset)
        };

        Ok(Self {
            ptr,
            layout,
            shape: layout.shape,
            front: 0,
            back: end - start,
            front_index,
            front_offset,
            back_index,
            back_offset,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<'a, T, const N: usize> Iterator for IndexedIterMut<'a, T, N> {
    type Item = ([usize; N], &'a mut T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        let index = self.front_index;
        let offset = self.front_offset;
        odometer_step(
            &mut self.front_index,
            &self.shape,
            &self.layout.strides,
            &mut self.front_offset,
        );
        self.front += 1;
        // SAFETY: construction validates storage bounds and rejects layouts
        // whose logical indices can alias the same physical offset. The shared
        // front/back cursor yields each logical index at most once.
        let elem = unsafe { &mut *self.ptr.as_ptr().add(offset) };
        Some((index, elem))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;
        (remaining, Some(remaining))
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for IndexedIterMut<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        let index = self.back_index;
        let offset = self.back_offset;
        odometer_step_back(
            &mut self.back_index,
            &self.shape,
            &self.layout.strides,
            &mut self.back_offset,
        );
        // SAFETY: construction validates storage bounds and rejects layouts
        // whose logical indices can alias the same physical offset. The shared
        // front/back cursor yields each logical index at most once.
        let elem = unsafe { &mut *self.ptr.as_ptr().add(offset) };
        Some((index, elem))
    }
}

impl<'a, T, const N: usize> ExactSizeIterator for IndexedIterMut<'a, T, N> {}

// SAFETY: construction validates storage bounds and injectivity, and the
// iterator owns the exclusive mutable traversal token for its range.
unsafe impl<T: Send, const N: usize> Send for IndexedIterMut<'_, T, N> {}

/// Fallible mutable iterator over logical row-major elements of a view.
///
/// Construction validates storage reachability and logical-offset injectivity
/// before any mutable reference can escape. This makes arbitrary positive and
/// negative strides safe while rejecting zero-stride or otherwise aliased
/// layouts. Construct through [`ArrayViewMut::try_iter_mut`] or
/// [`Array::try_iter_mut`].
pub struct ElementIterMut<'a, T, const N: usize> {
    inner: IndexedIterMut<'a, T, N>,
}

impl<'a, T, const N: usize> ElementIterMut<'a, T, N> {
    /// Build a plain mutable iterator from an already validated indexed iterator.
    #[inline]
    pub(crate) fn from_indexed(inner: IndexedIterMut<'a, T, N>) -> Self {
        Self { inner }
    }
}

impl<'a, T, const N: usize> Iterator for ElementIterMut<'a, T, N> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, value)| value)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for ElementIterMut<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|(_, value)| value)
    }
}

impl<'a, T, const N: usize> ExactSizeIterator for ElementIterMut<'a, T, N> {}

// SAFETY: ElementIterMut delegates to IndexedIterMut, whose construction and
// range proof prevent duplicate mutable references.
unsafe impl<T: Send, const N: usize> Send for ElementIterMut<'_, T, N> {}

/// A single disjoint logical task partition of a mutable array.
///
/// A partition is restricted to one half-open row-major logical range and
/// exposes only its own mutable element iterator. Consume it with
/// [`IntoIterator`]; a partition cannot be cloned or recreated from its raw
/// storage by safe code.
pub struct TaskPartitionMut<'a, T, const N: usize> {
    start: usize,
    end: usize,
    iter: ElementIterMut<'a, T, N>,
}

impl<'a, T, const N: usize> TaskPartitionMut<'a, T, N> {
    /// Return the zero-based logical range covered by this partition.
    #[must_use]
    pub const fn logical_range(&self) -> core::ops::Range<usize> {
        self.start..self.end
    }

    /// Return the number of logical elements in this partition.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.end - self.start
    }

    /// Return whether this partition contains no logical elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

impl<'a, T, const N: usize> IntoIterator for TaskPartitionMut<'a, T, N> {
    type Item = &'a mut T;
    type IntoIter = ElementIterMut<'a, T, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter
    }
}

// SAFETY: a partition contains only a range-limited mutable iterator. The
// parent proof establishes disjointness from every sibling range, and T: Send
// makes moving the token to a scoped worker sound.
unsafe impl<T: Send, const N: usize> Send for TaskPartitionMut<'_, T, N> {}

/// Allocation-free iterator producing disjoint mutable task partitions.
///
/// Construction validates storage bounds and logical-offset injectivity once.
/// Each yielded partition is a distinct row-major logical range, so the
/// injectivity proof lifts logical disjointness to physical storage
/// disjointness for negative and strided layouts as well.
pub struct TaskPartitionsMut<'a, T, const N: usize> {
    proof: MutableLayoutProof<'a, T, N>,
    chunk_size: usize,
    front: usize,
    back: usize,
}

impl<'a, T, const N: usize> TaskPartitionsMut<'a, T, N> {
    /// Build disjoint logical task partitions from a mutable view.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError`] if `chunk_size` is zero, storage bounds are
    /// invalid, or the layout is not provably injective.
    pub(crate) fn new(view: ArrayViewMut<'a, T, N>, chunk_size: usize) -> Result<Self> {
        if chunk_size == 0 {
            return Err(LetoError::StorageError {
                reason: "mutable task partition chunk size must be non-zero".to_string(),
            });
        }
        let proof = MutableLayoutProof::new(
            view,
            "mutable task partitions require provably disjoint logical offsets",
        )?;
        let size = proof.layout.size();
        Ok(Self {
            proof,
            chunk_size,
            front: 0,
            back: size.div_ceil(chunk_size),
        })
    }

    #[inline]
    fn range_at(&self, partition: usize) -> (usize, usize) {
        let size = self.proof.layout.size();
        let start = partition
            .checked_mul(self.chunk_size)
            .expect("invariant: partition start fits in logical size");
        let end = start.saturating_add(self.chunk_size).min(size);
        (start, end)
    }

    #[inline]
    fn partition_at(&self, partition: usize) -> TaskPartitionMut<'a, T, N> {
        let (start, end) = self.range_at(partition);
        let proof = MutableLayoutProof {
            ptr: self.proof.ptr,
            layout: self.proof.layout,
            storage_len: self.proof.storage_len,
            _marker: std::marker::PhantomData,
        };
        let inner = IndexedIterMut::from_proof(proof, start, end)
            .expect("invariant: partition range is within validated logical domain");
        TaskPartitionMut {
            start,
            end,
            iter: ElementIterMut::from_indexed(inner),
        }
    }
}

impl<'a, T, const N: usize> Iterator for TaskPartitionsMut<'a, T, N> {
    type Item = TaskPartitionMut<'a, T, N>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        let partition = self.partition_at(self.front);
        self.front += 1;
        Some(partition)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;
        (remaining, Some(remaining))
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for TaskPartitionsMut<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        Some(self.partition_at(self.back))
    }
}

impl<'a, T, const N: usize> ExactSizeIterator for TaskPartitionsMut<'a, T, N> {}

/// `for elem in &view` iterates the view's elements in logical row-major order.
impl<'a, T, const N: usize> IntoIterator for &ArrayView<'a, T, N> {
    type Item = &'a T;
    type IntoIter = ElementIter<'a, T, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// `for elem in &array` iterates an owned array in logical row-major order.
///
/// The iterator preserves the array's arbitrary strides and does not require a
/// contiguous copy. Mutable iteration intentionally remains fallible through
/// [`Array::indexed_iter_mut`](crate::application::array::Array::indexed_iter_mut),
/// because an infallible `IntoIterator<Item = &mut T>` implementation could not
/// report rejection of zero-stride aliasing layouts.
impl<'a, T, S, const N: usize> IntoIterator for &'a Array<T, S, N>
where
    S: Storage<T>,
{
    type Item = &'a T;
    type IntoIter = ElementIter<'a, T, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.view().iter()
    }
}

/// `for elem in &mutable_view` provides a read-only traversal without exposing
/// mutable aliases. Use [`ArrayViewMut::indexed_iter_mut`] for validated mutable
/// traversal.
impl<'a, T, const N: usize> IntoIterator for &'a ArrayViewMut<'_, T, N> {
    type Item = &'a T;
    type IntoIter = ElementIter<'a, T, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_view().iter()
    }
}
