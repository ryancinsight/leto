//! Runtime-rank strided layout (`LayoutDyn`): the boundary-carrier counterpart
//! of the const-rank [`Layout`](crate::domain::layout::Layout) (ADR 0007).

use crate::domain::error::{LetoError, Result};
use crate::domain::layout::kernels;

/// A strided layout whose **rank is a runtime value**: shape and strides live in
/// `Box<[_]>` rather than `[_; N]`.
///
/// `LayoutDyn` carries arbitrary-rank shape information across boundaries (PyO3,
/// generic I/O); it shares every piece of offset/size/validation arithmetic with
/// `Layout<N>` through the shared layout `kernels` module (SSOT). It is **not** a
/// compute layout — numeric work recovers a typed
/// [`Layout`](crate::domain::layout::Layout) via the array bridge (ADR 0007).
///
/// Invariant: `shape.len() == strides.len()` (the rank), enforced by every
/// constructor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutDyn {
    /// Extent of each axis.
    pub shape: Box<[usize]>,
    /// Stride of each axis, in elements (not bytes).
    pub strides: Box<[isize]>,
    /// Starting offset into the backing storage.
    pub offset: usize,
}

impl LayoutDyn {
    /// Construct from explicit shape, strides, and offset.
    ///
    /// # Errors
    /// [`LetoError::StorageError`] if `shape` and `strides` differ in length.
    pub fn new(shape: Box<[usize]>, strides: Box<[isize]>, offset: usize) -> Result<Self> {
        if shape.len() != strides.len() {
            return Err(LetoError::StorageError {
                reason: format!(
                    "dynamic layout shape rank {} does not match stride rank {}",
                    shape.len(),
                    strides.len()
                ),
            });
        }
        Ok(Self {
            shape,
            strides,
            offset,
        })
    }

    /// Construct a C-contiguous (row-major) layout for `shape`, offset `0`.
    ///
    /// # Errors
    /// [`LetoError::Overflow`] if a stride product does not fit in `isize`.
    pub fn c_contiguous(shape: &[usize]) -> Result<Self> {
        let mut strides = vec![0isize; shape.len()];
        kernels::c_contiguous_strides(shape, &mut strides)?;
        Ok(Self {
            shape: shape.to_vec().into_boxed_slice(),
            strides: strides.into_boxed_slice(),
            offset: 0,
        })
    }

    /// The rank (number of axes).
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Logical element count `∏ shapeᵢ`.
    #[inline]
    pub fn size(&self) -> usize {
        self.checked_size()
            .expect("layout shape product must fit in usize")
    }

    /// Logical element count with overflow validation.
    #[inline]
    pub fn checked_size(&self) -> Result<usize> {
        kernels::shape_size(&self.shape)
    }

    /// Broadcast this layout to a target runtime-rank shape.
    ///
    /// Equal trailing extents retain their strides. A source extent of one
    /// becomes a zero-stride view, and prepended axes are zero-stride views.
    /// The backing storage is never copied.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::IncompatibleBroadcast`]
    /// when a source extent cannot broadcast to the corresponding target
    /// extent, or a layout arithmetic error from the validating constructor.
    pub fn broadcast(&self, target_shape: &[usize]) -> Result<Self> {
        let mut shape = vec![0usize; target_shape.len()];
        let mut strides = vec![0isize; target_shape.len()];
        kernels::broadcast_layout(
            &self.shape,
            &self.strides,
            target_shape,
            &mut shape,
            &mut strides,
        )?;
        Self::new(
            shape.into_boxed_slice(),
            strides.into_boxed_slice(),
            self.offset,
        )
    }

    /// Returns the minimum and maximum physical offsets addressed by this
    /// layout with signed overflow validation.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::Overflow`] when the physical address bounds
    /// overflow, or [`LetoError::StorageError`]
    /// when a negative stride reaches below zero.
    pub fn checked_min_max_offsets(&self) -> Result<(usize, usize)> {
        kernels::min_max_offsets(&self.shape, &self.strides, self.offset)
    }

    /// Returns whether distinct logical indices address distinct elements.
    ///
    /// The proof is shared with fixed-rank layouts, including its exact
    /// bounded search for ambiguous interleaved strides.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::Overflow`] if the
    /// exact difference proof exceeds its integer bounds.
    pub fn is_injective(&self) -> Result<bool> {
        kernels::is_injective(&self.shape, &self.strides)
    }

    /// Physical offset of `index` via the shared `physical_offset` kernel.
    ///
    /// # Errors
    /// [`LetoError::OutOfBounds`] on wrong arity / out-of-range component;
    /// overflow / negative-offset errors otherwise.
    #[inline]
    pub fn offset_of(&self, index: &[usize]) -> Result<usize> {
        kernels::physical_offset(&self.shape, &self.strides, self.offset, index)
    }

    /// Validate that every addressable physical offset lies within `storage_len`.
    ///
    /// # Errors
    /// [`LetoError::StorageError`] if the addressable range exceeds `storage_len`.
    #[inline]
    pub fn validate_storage_len(&self, storage_len: usize) -> Result<()> {
        kernels::validate_storage(&self.shape, &self.strides, self.offset, storage_len)
    }

    /// Returns true when this layout is canonically C-contiguous at offset `0`.
    pub fn is_c_contiguous(&self) -> bool {
        if self.offset != 0 {
            return false;
        }
        let mut expected = vec![0isize; self.ndim()];
        match kernels::c_contiguous_strides(&self.shape, &mut expected) {
            Ok(()) => {}
            Err(_) => return false,
        }
        // A zero-extent axis makes its stride irrelevant to addressing; the
        // canonical comparison still holds because both sides set it to 0.
        self.strides.as_ref() == expected.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::LayoutDyn;

    #[test]
    fn broadcast_preserves_strides_and_zeroes_broadcast_axes() {
        let source = LayoutDyn::new(Box::from([2usize, 1, 3]), Box::from([3isize, 3, 1]), 4)
            .expect("invariant: source layout has matching ranks");
        let broadcasted = source
            .broadcast(&[5, 2, 4, 3])
            .expect("invariant: target is broadcastable");

        assert_eq!(broadcasted.shape.as_ref(), &[5, 2, 4, 3]);
        assert_eq!(broadcasted.strides.as_ref(), &[0, 3, 0, 1]);
        assert_eq!(broadcasted.offset, 4);
    }

    #[test]
    fn dynamic_injectivity_matches_exhaustive_small_rank_two_oracle() {
        for first_dimension in 1..=3 {
            for second_dimension in 1..=3 {
                let shape = [first_dimension, second_dimension];
                for first_stride in -3..=3 {
                    for second_stride in -3..=3 {
                        let strides = [first_stride, second_stride];
                        let mut offsets = BTreeSet::new();
                        for first in 0..first_dimension {
                            for second in 0..second_dimension {
                                offsets.insert(
                                    first as isize * first_stride + second as isize * second_stride,
                                );
                            }
                        }
                        let expected = offsets.len() == first_dimension * second_dimension;
                        let layout = LayoutDyn::new(
                            shape.to_vec().into_boxed_slice(),
                            strides.to_vec().into_boxed_slice(),
                            0,
                        )
                        .expect("invariant: dynamic ranks match");
                        assert_eq!(
                            layout.is_injective().expect("invariant: proof fits"),
                            expected,
                            "shape={shape:?}, strides={strides:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn dynamic_injectivity_accepts_interleaved_and_rejects_broadcast_writes() {
        let interleaved = LayoutDyn::new(Box::from([2usize, 3]), Box::from([3isize, 2]), 0)
            .expect("invariant: valid interleaved layout");
        assert!(interleaved.is_injective().expect("invariant: proof fits"));

        let broadcast = LayoutDyn::new(Box::from([3usize, 4]), Box::from([4isize, 0]), 0)
            .expect("invariant: valid broadcast layout");
        assert!(!broadcast.is_injective().expect("invariant: proof fits"));
    }
}
