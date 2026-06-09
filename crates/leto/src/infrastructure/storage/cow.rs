use super::traits::{Storage, StorageMut};
use std::borrow::Cow;

/// Copy-on-write storage for arrays that can borrow read-only input and detach on mutation.
pub struct CowStorage<'a, T: Clone> {
    data: Cow<'a, [T]>,
}

impl<'a, T: Clone> CowStorage<'a, T> {
    /// Create storage from a borrowed slice without copying.
    #[inline]
    pub const fn borrowed(slice: &'a [T]) -> Self {
        Self {
            data: Cow::Borrowed(slice),
        }
    }

    /// Create storage from an owned vector.
    #[inline]
    pub const fn owned(data: Vec<T>) -> Self {
        Self {
            data: Cow::Owned(data),
        }
    }

    /// Returns true when the storage still borrows caller-owned memory.
    #[inline]
    pub const fn is_borrowed(&self) -> bool {
        matches!(self.data, Cow::Borrowed(_))
    }

    /// Returns true when the storage owns its memory.
    #[inline]
    pub const fn is_owned(&self) -> bool {
        matches!(self.data, Cow::Owned(_))
    }

    /// Returns the borrowed backing slice if this storage has not detached.
    #[inline]
    pub const fn as_borrowed(&self) -> Option<&'a [T]> {
        match &self.data {
            Cow::Borrowed(slice) => Some(slice),
            Cow::Owned(_) => None,
        }
    }

    /// Returns the owned backing vector if this storage has detached or was constructed as owned.
    #[inline]
    pub const fn as_owned(&self) -> Option<&Vec<T>> {
        match &self.data {
            Cow::Borrowed(_) => None,
            Cow::Owned(data) => Some(data),
        }
    }

    /// Consume the storage and return owned data, cloning only when still borrowed.
    #[inline]
    pub fn into_owned(self) -> Vec<T> {
        self.data.into_owned()
    }
}

impl<'a, T: Clone> Storage<T> for CowStorage<'a, T> {
    #[inline]
    fn as_slice(&self) -> &[T] {
        self.data.as_ref()
    }
}

impl<'a, T: Clone> StorageMut<T> for CowStorage<'a, T> {
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.to_mut()
    }
}
