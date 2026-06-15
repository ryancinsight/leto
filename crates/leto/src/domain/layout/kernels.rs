//! Rank-agnostic strided-layout arithmetic, shared by the const-rank
//! [`Layout<N>`](super::Layout) and the runtime-rank
//! [`LayoutDyn`](crate::domain::dynamic::LayoutDyn) (SSOT).
//!
//! Every function operates on `&[usize]` / `&[isize]` slices so a single
//! implementation serves both the `[_; N]`-backed and `Box<[_]>`-backed layouts.
//! The const-rank `Layout<N>` keeps its array storage and zero-cost API; only the
//! arithmetic bodies live here, eliminating the duplication a parallel dynamic
//! layout would otherwise introduce (ADR 0007).

use crate::domain::error::{LetoError, Result};

/// Logical element count `∏ shapeᵢ`, returning `0` if any extent is `0`.
///
/// # Errors
/// [`LetoError::Overflow`] if the product does not fit in `usize`.
#[inline]
pub(crate) fn shape_size(shape: &[usize]) -> Result<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1usize, |size, &dim| {
        size.checked_mul(dim).ok_or(LetoError::Overflow {
            reason: "layout shape product",
        })
    })
}

/// Fill `out` with the C-contiguous (row-major) strides of `shape`.
///
/// `out.len()` must equal `shape.len()`. A zero extent sets the running stride to
/// `0` (matching the const-rank constructor).
///
/// # Errors
/// [`LetoError::Overflow`] if a stride product does not fit in `isize`.
#[inline]
pub(crate) fn c_contiguous_strides(shape: &[usize], out: &mut [isize]) -> Result<()> {
    debug_assert_eq!(shape.len(), out.len());
    let mut stride = 1isize;
    for i in (0..shape.len()).rev() {
        out[i] = stride;
        let dim = shape[i];
        if dim == 0 {
            stride = 0;
        } else {
            let dim = isize::try_from(dim).map_err(|_| LetoError::Overflow {
                reason: "C-contiguous dimension conversion",
            })?;
            stride = stride.checked_mul(dim).ok_or(LetoError::Overflow {
                reason: "C-contiguous stride multiplication",
            })?;
        }
    }
    Ok(())
}

/// Physical offset of `index`: `base + Σ indexᵢ·stridesᵢ`.
///
/// # Errors
/// [`LetoError::OutOfBounds`] if `index` has the wrong arity or any component is
/// out of range; [`LetoError::Overflow`] / [`LetoError::StorageError`] on signed
/// overflow or a negative resulting offset.
#[inline]
pub(crate) fn physical_offset(
    shape: &[usize],
    strides: &[isize],
    base: usize,
    index: &[usize],
) -> Result<usize> {
    if index.len() != shape.len() {
        return Err(LetoError::OutOfBounds {
            index: index.to_vec(),
            shape: shape.to_vec(),
        });
    }
    let mut offset = isize::try_from(base).map_err(|_| LetoError::Overflow {
        reason: "layout base offset conversion",
    })?;
    for i in 0..shape.len() {
        if index[i] >= shape[i] {
            return Err(LetoError::OutOfBounds {
                index: index.to_vec(),
                shape: shape.to_vec(),
            });
        }
        let idx = isize::try_from(index[i]).map_err(|_| LetoError::Overflow {
            reason: "layout index conversion",
        })?;
        let delta = idx.checked_mul(strides[i]).ok_or(LetoError::Overflow {
            reason: "layout offset multiplication",
        })?;
        offset = offset.checked_add(delta).ok_or(LetoError::Overflow {
            reason: "layout offset accumulation",
        })?;
    }
    if offset < 0 {
        return Err(LetoError::StorageError {
            reason: format!("layout index accesses negative physical offset {offset}"),
        });
    }
    Ok(offset as usize)
}

/// Minimum and maximum physical offsets the layout can address.
///
/// Empty shapes and shapes containing a `0` extent collapse to `(base, base)`.
///
/// # Errors
/// [`LetoError::Overflow`] on signed overflow; [`LetoError::StorageError`] if the
/// minimum offset is negative.
#[inline]
pub(crate) fn min_max_offsets(
    shape: &[usize],
    strides: &[isize],
    base: usize,
) -> Result<(usize, usize)> {
    if shape.is_empty() || shape.contains(&0) {
        return Ok((base, base));
    }
    let mut min_offset = isize::try_from(base).map_err(|_| LetoError::Overflow {
        reason: "layout base offset conversion",
    })?;
    let mut max_offset = min_offset;
    for i in 0..shape.len() {
        let s = strides[i];
        let len_minus_one = isize::try_from(shape[i] - 1).map_err(|_| LetoError::Overflow {
            reason: "layout dimension bound conversion",
        })?;
        let bound = len_minus_one.checked_mul(s).ok_or(LetoError::Overflow {
            reason: "layout dimension bound multiplication",
        })?;
        min_offset = min_offset
            .checked_add(0isize.min(bound))
            .ok_or(LetoError::Overflow {
                reason: "layout minimum offset accumulation",
            })?;
        max_offset = max_offset
            .checked_add(0isize.max(bound))
            .ok_or(LetoError::Overflow {
                reason: "layout maximum offset accumulation",
            })?;
    }
    if min_offset < 0 {
        return Err(LetoError::StorageError {
            reason: format!("layout accesses negative physical offset {min_offset}"),
        });
    }
    Ok((min_offset as usize, max_offset as usize))
}

/// Validate that every addressable physical offset lies within `storage_len`.
///
/// # Errors
/// [`LetoError::StorageError`] if the addressable range exceeds `storage_len`;
/// propagates overflow errors from the offset bounds.
#[inline]
pub(crate) fn validate_storage(
    shape: &[usize],
    strides: &[isize],
    base: usize,
    storage_len: usize,
) -> Result<()> {
    if shape_size(shape)? == 0 {
        return Ok(());
    }
    let (min_offset, max_offset) = min_max_offsets(shape, strides, base)?;
    if min_offset >= storage_len || max_offset >= storage_len {
        return Err(LetoError::StorageError {
            reason: format!(
                "storage length {storage_len} does not cover layout physical offsets {min_offset}..={max_offset}"
            ),
        });
    }
    Ok(())
}

/// Decompose a flat row-major index into per-axis coordinates, writing into
/// `out` (`out.len()` must equal `shape.len()`).
#[inline]
pub(crate) fn fill_index_from_flat(flat: usize, shape: &[usize], out: &mut [usize]) {
    debug_assert_eq!(shape.len(), out.len());
    let mut temp = flat;
    for axis in (0..shape.len()).rev() {
        if shape[axis] > 0 {
            out[axis] = temp % shape[axis];
            temp /= shape[axis];
        }
    }
}
