//! Disjoint mutable task partitions: [`TaskPartitionMut`] and
//! [`TaskPartitionsMut`].

use super::element_iter::ElementIterMut;
use super::indexed_iter::IndexedIterMut;
use super::odometer::MutableLayoutProof;
use crate::application::view::ArrayViewMut;
use crate::domain::error::{LetoError, Result};

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
