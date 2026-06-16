//! Stack-allocated fixed-capacity array storage (allocation-free backing).

use super::traits::{Storage, StorageMut};

/// Owned array storage backed by an inline `[T; CAP]` — no heap allocation.
///
/// `StackStorage` is the allocation-free counterpart of
/// [`VecStorage`](super::VecStorage): the elements live inline in the value, so a
/// stack-backed [`Array`](crate::application::array::Array) needs no allocator
/// (`no_std`-friendly) and is itself `Copy` when `T: Copy`. Because every array
/// operation is generic over the [`Storage`] trait (DIP), a stack-backed array
/// inherits the **entire** operation surface — reductions, arithmetic,
/// iteration, slicing, transpose — with no duplicated kernels (SSOT).
///
/// `CAP` is the fixed physical element count. Leto encodes const *rank* with
/// runtime *dimensions* (ADR 0002), so the shape is supplied at construction and
/// validated against `CAP` rather than carried in the type; see
/// [`from_stack`](crate::application::array::Array::from_stack).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackStorage<T, const CAP: usize> {
    data: [T; CAP],
}

impl<T, const CAP: usize> StackStorage<T, CAP> {
    /// Wrap an inline array.
    #[inline]
    pub const fn new(data: [T; CAP]) -> Self {
        Self { data }
    }

    /// A storage of `CAP` copies of `value`.
    #[inline]
    pub fn fill(value: T) -> Self
    where
        T: Copy,
    {
        Self { data: [value; CAP] }
    }

    /// Consume the storage and return the inline array.
    #[inline]
    pub fn into_inner(self) -> [T; CAP] {
        self.data
    }
}

impl<T, const CAP: usize> Storage<T> for StackStorage<T, CAP> {
    #[inline]
    fn as_slice(&self) -> &[T] {
        &self.data
    }
}

impl<T, const CAP: usize> StorageMut<T> for StackStorage<T, CAP> {
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}
