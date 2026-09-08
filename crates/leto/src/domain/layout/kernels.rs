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
        #[expect(
            clippy::unnecessary_lazy_evaluations,
            reason = "Avoid eager LetoError drop on successful arithmetic; ADR 0027"
        )]
        size.checked_mul(dim).ok_or_else(|| LetoError::Overflow {
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
            #[expect(
                clippy::unnecessary_lazy_evaluations,
                reason = "Avoid eager LetoError drop on successful arithmetic; ADR 0027"
            )]
            {
                stride = stride.checked_mul(dim).ok_or_else(|| LetoError::Overflow {
                    reason: "C-contiguous stride multiplication",
                })?;
            }
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
        #[expect(
            clippy::unnecessary_lazy_evaluations,
            reason = "Avoid eager LetoError drop on successful arithmetic; ADR 0027"
        )]
        let delta = idx
            .checked_mul(strides[i])
            .ok_or_else(|| LetoError::Overflow {
                reason: "layout offset multiplication",
            })?;
        #[expect(
            clippy::unnecessary_lazy_evaluations,
            reason = "Avoid eager LetoError drop on successful arithmetic; ADR 0027"
        )]
        {
            offset = offset
                .checked_add(delta)
                .ok_or_else(|| LetoError::Overflow {
                    reason: "layout offset accumulation",
                })?;
        }
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
        #[expect(
            clippy::unnecessary_lazy_evaluations,
            reason = "Avoid eager LetoError drop on successful arithmetic; ADR 0027"
        )]
        let bound = len_minus_one
            .checked_mul(s)
            .ok_or_else(|| LetoError::Overflow {
                reason: "layout dimension bound multiplication",
            })?;
        #[expect(
            clippy::unnecessary_lazy_evaluations,
            reason = "Avoid eager LetoError drop on successful arithmetic; ADR 0027"
        )]
        {
            min_offset =
                min_offset
                    .checked_add(0isize.min(bound))
                    .ok_or_else(|| LetoError::Overflow {
                        reason: "layout minimum offset accumulation",
                    })?;
        }
        #[expect(
            clippy::unnecessary_lazy_evaluations,
            reason = "Avoid eager LetoError drop on successful arithmetic; ADR 0027"
        )]
        {
            max_offset =
                max_offset
                    .checked_add(0isize.max(bound))
                    .ok_or_else(|| LetoError::Overflow {
                        reason: "layout maximum offset accumulation",
                    })?;
        }
    }
    if min_offset < 0 {
        return Err(LetoError::StorageError {
            reason: format!("layout accesses negative physical offset {min_offset}"),
        });
    }
    Ok((min_offset as usize, max_offset as usize))
}

/// Write the strides produced by broadcasting `source_shape` to `target_shape`.
///
/// The caller owns the output storage so this rank-agnostic kernel performs no
/// allocation. A zero stride is introduced only for a prepended axis or an
/// expanded singleton source axis.
pub(crate) fn broadcast_strides(
    source_shape: &[usize],
    source_strides: &[isize],
    target_shape: &[usize],
    output_strides: &mut [isize],
) -> Result<()> {
    debug_assert_eq!(source_shape.len(), source_strides.len());
    debug_assert_eq!(target_shape.len(), output_strides.len());
    if target_shape.len() < source_shape.len() {
        return Err(LetoError::IncompatibleBroadcast {
            from: source_shape.to_vec(),
            to: target_shape.to_vec(),
        });
    }

    output_strides.fill(0);
    let shift = target_shape.len() - source_shape.len();
    for axis in 0..source_shape.len() {
        let target_axis = axis + shift;
        let source_extent = source_shape[axis];
        let target_extent = target_shape[target_axis];
        if source_extent == target_extent {
            output_strides[target_axis] = source_strides[axis];
        } else if source_extent != 1 {
            return Err(LetoError::IncompatibleBroadcast {
                from: source_shape.to_vec(),
                to: target_shape.to_vec(),
            });
        }
    }
    Ok(())
}

/// Determine whether distinct logical indices address distinct elements.
///
/// Separated strides complete without allocation. Ambiguous layouts use an
/// exact bounded integer-difference search over the same slices, preserving
/// arbitrary injective views for both const and runtime ranks.
///
/// # Errors
/// [`LetoError::Overflow`] when shape, stride, or exact-difference arithmetic
/// cannot be represented by the checked integer types.
pub(crate) fn is_injective(shape: &[usize], strides: &[isize]) -> Result<bool> {
    debug_assert_eq!(shape.len(), strides.len());
    if shape_size(shape)? <= 1 {
        return Ok(true);
    }

    let mut covered_span = 0usize;
    let mut previous_magnitude = 0usize;
    loop {
        let mut next = None;
        for axis in 0..shape.len() {
            if shape[axis] <= 1 {
                continue;
            }
            let magnitude = strides[axis].unsigned_abs();
            if magnitude == 0 {
                return Ok(false);
            }
            if magnitude > previous_magnitude
                && next.is_none_or(|(candidate, _)| magnitude < candidate)
            {
                next = Some((magnitude, axis));
            }
        }
        let Some((magnitude, axis)) = next else {
            break;
        };
        let same_stride_count = shape
            .iter()
            .zip(strides)
            .filter(|&(dimension, stride)| *dimension > 1 && stride.unsigned_abs() == magnitude)
            .count();
        if same_stride_count > 1 || magnitude <= covered_span {
            return exact_injectivity(shape, strides);
        }
        let axis_span = (shape[axis] - 1)
            .checked_mul(magnitude)
            .ok_or(LetoError::Overflow {
                reason: "layout injectivity axis span",
            })?;
        covered_span = covered_span
            .checked_add(axis_span)
            .ok_or(LetoError::Overflow {
                reason: "layout injectivity covered span",
            })?;
        previous_magnitude = magnitude;
    }
    Ok(true)
}

fn exact_injectivity(shape: &[usize], strides: &[isize]) -> Result<bool> {
    let solve_axis = shape
        .iter()
        .enumerate()
        .max_by_key(|&(_, dimension)| dimension.saturating_sub(1))
        .map_or(0, |(axis, _)| axis);
    let search = DifferenceSearch {
        shape,
        strides,
        solve_axis,
        solve_stride: strides[solve_axis],
    };
    Ok(!search.has_collision(0, 0, false)?)
}

struct DifferenceSearch<'a> {
    shape: &'a [usize],
    strides: &'a [isize],
    solve_axis: usize,
    solve_stride: isize,
}

impl DifferenceSearch<'_> {
    fn has_collision(
        &self,
        axis: usize,
        residual: i128,
        has_nonzero_difference: bool,
    ) -> Result<bool> {
        if axis == self.shape.len() {
            let solve_stride = self.solve_stride as i128;
            if solve_stride == 0 || residual % solve_stride != 0 {
                return Ok(false);
            }
            let solved = residual.checked_neg().ok_or(LetoError::Overflow {
                reason: "layout injectivity solved difference",
            })? / solve_stride;
            let absolute = solved.checked_abs().ok_or(LetoError::Overflow {
                reason: "layout injectivity solved difference magnitude",
            })?;
            let bound = i128::try_from(self.shape[self.solve_axis] - 1).map_err(|_| {
                LetoError::Overflow {
                    reason: "layout injectivity solved difference bound",
                }
            })?;
            return Ok(absolute <= bound && (has_nonzero_difference || solved != 0));
        }
        if axis == self.solve_axis {
            return self.has_collision(axis + 1, residual, has_nonzero_difference);
        }

        let bound = i128::try_from(self.shape[axis] - 1).map_err(|_| LetoError::Overflow {
            reason: "layout injectivity difference bound",
        })?;
        let stride = self.strides[axis] as i128;
        for difference in -bound..=bound {
            let term = difference
                .checked_mul(stride)
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
