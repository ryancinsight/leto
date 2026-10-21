//! The [`Layout`] type: construction, invariant, and accessors.

/// Represents an N-dimensional strided layout.
///
/// # Invariant
///
/// Every `Layout` reachable from safe code satisfies the *self-contained*
/// layout invariant:
///
/// 1. the shape product (logical element count) fits in `usize`;
/// 2. the physical offsets addressed by the layout — `offset` plus every
///    partial sum of `(shape[i] - 1) * strides[i]` — neither overflow `isize`
///    nor fall below zero.
///
/// Together these guarantee that [`Layout::size`], [`Layout::min_max_offsets`]
/// and [`Layout::offset_of`] are total on in-shape indices, so the infallible
/// accessors cannot panic.
///
/// This invariant is deliberately *buffer-independent*: a `Layout` carries no
/// pointer and no length, so it cannot on its own express "fits in the backing
/// storage". That second invariant is established where a layout meets a
/// buffer, by [`Layout::validate_storage_len`], and is what the `try_new`
/// constructors of the view types check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Layout<const N: usize> {
    pub(super) shape: [usize; N],
    pub(super) strides: [isize; N],
    pub(super) offset: usize,
}

impl<const N: usize> Layout<N> {
    /// Create a layout from explicit shape, strides, and offset, validating the
    /// self-contained layout invariant documented on [`Layout`].
    ///
    /// This is the only construction path available outside this crate.
    ///
    /// # Errors
    ///
    /// Returns [`crate::LetoError::Overflow`] when the shape product or the physical
    /// offset arithmetic exceeds its integer type, and
    /// [`crate::LetoError::StorageError`] when negative strides drive an addressed
    /// physical offset below zero.
    pub fn try_new(
        shape: [usize; N],
        strides: [isize; N],
        offset: usize,
    ) -> crate::domain::error::Result<Self> {
        let candidate = Self::from_parts_unchecked(shape, strides, offset);
        candidate.checked_size()?;
        candidate.checked_min_max_offsets()?;
        Ok(candidate)
    }

    /// Assemble a layout without validating the self-contained invariant.
    ///
    /// Crate-internal only. Every caller derives the parts from an already
    /// validated layout (a slice, transpose, broadcast, or per-step iterator
    /// subview), so the invariant is inherited rather than re-established —
    /// which keeps the check out of hot iterator `next` paths.
    ///
    /// A `Layout` owns no pointer, so a violation here cannot itself cause
    /// undefined behavior; it degrades the infallible accessors to panics.
    #[inline]
    pub(crate) const fn from_parts_unchecked(
        shape: [usize; N],
        strides: [isize; N],
        offset: usize,
    ) -> Self {
        Self {
            shape,
            strides,
            offset,
        }
    }

    /// The shape of the array (size of each dimension).
    #[inline]
    #[must_use]
    pub const fn shape(&self) -> [usize; N] {
        self.shape
    }

    /// The stride of each dimension in elements (not bytes).
    #[inline]
    #[must_use]
    pub const fn strides(&self) -> [isize; N] {
        self.strides
    }

    /// The starting offset in the storage buffer.
    #[inline]
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }
}

impl<const N: usize> TryFrom<([usize; N], [isize; N], usize)> for Layout<N> {
    type Error = crate::domain::error::LetoError;

    /// Validating conversion from `(shape, strides, offset)`; see
    /// [`Layout::try_new`].
    ///
    /// # Errors
    ///
    /// As [`Layout::try_new`].
    #[inline]
    fn try_from(
        (shape, strides, offset): ([usize; N], [isize; N], usize),
    ) -> core::result::Result<Self, Self::Error> {
        Self::try_new(shape, strides, offset)
    }
}
