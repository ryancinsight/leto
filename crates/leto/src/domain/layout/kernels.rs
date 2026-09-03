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

/// Broadcast `shape` and `strides` into `target_shape`, writing the resulting
/// metadata into caller-provided buffers.
///
/// The buffers must both have the same length as `target_shape`. Keeping this
/// operation rank-agnostic lets fixed and runtime-rank layouts share the
/// broadcast law without moving either representation to the heap.
pub(crate) fn broadcast_layout(
    shape: &[usize],
    strides: &[isize],
    target_shape: &[usize],
    out_shape: &mut [usize],
    out_strides: &mut [isize],
) -> Result<()> {
    debug_assert_eq!(shape.len(), strides.len());
    debug_assert_eq!(target_shape.len(), out_shape.len());
    debug_assert_eq!(target_shape.len(), out_strides.len());

    if shape.len() > target_shape.len() {
        return Err(LetoError::IncompatibleBroadcast {
            from: shape.to_vec(),
            to: target_shape.to_vec(),
        });
    }

    let shift = target_shape.len() - shape.len();
    out_shape[..shift].copy_from_slice(&target_shape[..shift]);
    out_strides[..shift].fill(0);

    for (axis, (&source_dim, &source_stride)) in shape.iter().zip(strides).enumerate() {
        let target_axis = axis + shift;
        let target_dim = target_shape[target_axis];
        match source_dim {
            dim if dim == target_dim => {
                out_shape[target_axis] = target_dim;
                out_strides[target_axis] = source_stride;
            }
            1 => {
                out_shape[target_axis] = target_dim;
                out_strides[target_axis] = 0;
            }
            _ => {
                return Err(LetoError::IncompatibleBroadcast {
                    from: shape.to_vec(),
                    to: target_shape.to_vec(),
                });
            }
        }
    }
    Ok(())
}

/// Return whether distinct logical indices address distinct physical offsets.
///
/// Separated strides are proved in `O(rank log rank)` time. Ambiguous layouts
/// use the exact bounded difference search below, preserving legal interleaved
/// views without allocating an offset set.
pub(crate) fn is_injective(shape: &[usize], strides: &[isize]) -> Result<bool> {
    debug_assert_eq!(shape.len(), strides.len());
    if shape_size(shape)? <= 1 {
        return Ok(true);
    }

    let mut axes = Vec::with_capacity(shape.len());
    for (&dimension, &stride) in shape.iter().zip(strides) {
        if dimension <= 1 {
            continue;
        }
        let magnitude = stride.unsigned_abs();
        if magnitude == 0 {
            return Ok(false);
        }
        axes.push((magnitude, dimension));
    }
    axes.sort_unstable_by_key(|&(stride, _)| stride);

    let mut covered_span = 0usize;
    for &(stride, dimension) in &axes {
        if stride <= covered_span {
            return exact_injectivity(shape, strides);
        }
        covered_span = covered_span
            .checked_add(
                dimension
                    .checked_sub(1)
                    .and_then(|extent| extent.checked_mul(stride))
                    .ok_or(LetoError::Overflow {
                        reason: "layout injectivity axis span",
                    })?,
            )
            .ok_or(LetoError::Overflow {
                reason: "layout injectivity covered span",
            })?;
    }
    Ok(true)
}

fn exact_injectivity(shape: &[usize], strides: &[isize]) -> Result<bool> {
    let bounds = shape
        .iter()
        .map(|&dimension| {
            i128::try_from(dimension.saturating_sub(1)).map_err(|_| LetoError::Overflow {
                reason: "layout injectivity bound conversion",
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let strides = strides
        .iter()
        .map(|&stride| {
            i128::try_from(stride).map_err(|_| LetoError::Overflow {
                reason: "layout injectivity stride conversion",
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let solve_axis = bounds
        .iter()
        .enumerate()
        .max_by_key(|&(_, bound)| bound)
        .map(|(axis, _)| axis)
        .ok_or(LetoError::Overflow {
            reason: "layout injectivity solve axis",
        })?;
    let search = DifferenceSearch {
        bounds: &bounds,
        strides: &strides,
        solve_axis,
        solve_stride: strides[solve_axis],
    };
    Ok(!search.has_collision(0, 0, false)?)
}

struct DifferenceSearch<'a> {
    bounds: &'a [i128],
    strides: &'a [i128],
    solve_axis: usize,
    solve_stride: i128,
}

impl DifferenceSearch<'_> {
    fn has_collision(
        &self,
        axis: usize,
        residual: i128,
        has_nonzero_difference: bool,
    ) -> Result<bool> {
        if axis == self.bounds.len() {
            if residual % self.solve_stride != 0 {
                return Ok(false);
            }
            let solved = residual.checked_neg().ok_or(LetoError::Overflow {
                reason: "layout injectivity solved difference",
            })? / self.solve_stride;
            return Ok(solved.abs() <= self.bounds[self.solve_axis]
                && (has_nonzero_difference || solved != 0));
        }
        if axis == self.solve_axis {
            return self.has_collision(axis + 1, residual, has_nonzero_difference);
        }
        for difference in -self.bounds[axis]..=self.bounds[axis] {
            let term = difference
                .checked_mul(self.strides[axis])
                .and_then(|term| residual.checked_add(term))
                .ok_or(LetoError::Overflow {
                    reason: "layout injectivity difference sum",
                })?;
            if self.has_collision(axis + 1, term, has_nonzero_difference || difference != 0)? {
                return Ok(true);
            }
        }
        Ok(false)
    }
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
