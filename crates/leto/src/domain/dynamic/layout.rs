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

    /// Return the minimum and maximum physical offsets addressed by this layout.
    ///
    /// # Errors
    /// [`LetoError::Overflow`] on signed arithmetic overflow or
    /// [`LetoError::StorageError`] when a negative stride reaches below offset
    /// zero.
    #[inline]
    pub fn checked_min_max_offsets(&self) -> Result<(usize, usize)> {
        kernels::min_max_offsets(&self.shape, &self.strides, self.offset)
    }

    /// Return whether distinct logical indices address distinct elements.
    ///
    /// The check shares the allocation-free separated-stride path and exact
    /// fallback with const-rank [`Layout`](crate::Layout). It validates view
    /// aliasing without materializing the addressed offsets.
    ///
    /// # Errors
    /// [`LetoError::Overflow`] when exact difference arithmetic exceeds the
    /// checked integer range.
    #[inline]
    pub fn is_injective(&self) -> Result<bool> {
        kernels::is_injective(&self.shape, &self.strides)
    }

    /// Broadcast this layout to a compatible runtime-rank target shape.
    ///
    /// Only shape and stride metadata are allocated; any element storage
    /// remains untouched and can continue to be shared by the resulting view.
    ///
    /// # Errors
    /// [`LetoError::IncompatibleBroadcast`] when ranks or extents cannot be
    /// aligned.
    pub fn broadcast(&self, target_shape: &[usize]) -> Result<Self> {
        let mut output_strides = vec![0isize; target_shape.len()];
        kernels::broadcast_strides(
            &self.shape,
            &self.strides,
            target_shape,
            &mut output_strides,
        )?;
        Self::new(
            target_shape.to_vec().into_boxed_slice(),
            output_strides.into_boxed_slice(),
            self.offset,
        )
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
