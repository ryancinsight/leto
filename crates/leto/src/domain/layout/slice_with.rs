use crate::domain::error::{LetoError, Result};
use crate::domain::slice::{normalize_index, normalize_range, SliceArg};

use super::Layout;

impl<const N: usize> Layout<N> {
    /// Compute the physical element offset for a given multi-dimensional index.
    pub fn offset_of(&self, index: [usize; N]) -> Result<usize> {
        crate::domain::layout::kernels::physical_offset(
            &self.shape,
            &self.strides,
            self.offset,
            &index,
        )
    }

    /// Slice the layout on each axis given a slice definition `(start, end, step)`.
    pub fn slice(&self, ranges: &[(usize, usize, isize); N]) -> Result<Self> {
        let mut args = [SliceArg::All; N];
        for (i, &(start, end, step)) in ranges.iter().enumerate() {
            if start > self.shape[i] || end > self.shape[i] {
                return Err(LetoError::IncompatibleSlice {
                    range: (start, end),
                    shape: self.shape.to_vec(),
                });
            }
            let start = isize::try_from(start).map_err(|_| LetoError::Overflow {
                reason: "slice start conversion",
            })?;
            let end = isize::try_from(end).map_err(|_| LetoError::Overflow {
                reason: "slice end conversion",
            })?;
            args[i] = SliceArg::range(Some(start), Some(end), step);
        }
        self.slice_with(&args)
    }

    /// Slice the layout with leto-style arguments.
    ///
    /// This supports full-axis ranges, optional signed bounds, negative indices,
    /// negative strides, integer indexing that removes an axis, inserted new axes,
    /// and one ellipsis expansion. The caller specifies the output rank `M`.
    pub fn slice_with<const M: usize>(&self, args: &[SliceArg]) -> Result<Layout<M>> {
        // Validate the slice specification and expand any ellipsis inline
        // instead of materializing an intermediate `Vec<SliceArg>`. The
        // expanded stream feeds directly into the axis walker below, so a
        // slice/reshape view construction performs no heap allocation.
        let ellipsis_count = args
            .iter()
            .filter(|arg| matches!(arg, SliceArg::Ellipsis))
            .count();
        if ellipsis_count > 1 {
            return Err(LetoError::StorageError {
                reason: "slice specification contains more than one ellipsis".to_string(),
            });
        }

        let consumed_without_ellipsis = args
            .iter()
            .filter(|arg| {
                matches!(
                    arg,
                    SliceArg::All | SliceArg::Range { .. } | SliceArg::Index(_)
                )
            })
            .count();
        if consumed_without_ellipsis > N {
            return Err(slice_rank_error(N, N, args));
        }

        let fill = if ellipsis_count == 0 {
            N.saturating_sub(consumed_without_ellipsis)
        } else {
            N - consumed_without_ellipsis
        };

        let mut shape = [0usize; M];
        let mut strides = [0isize; M];
        let mut input_axis = 0usize;
        let mut output_axis = 0usize;
        let mut offset = isize::try_from(self.offset).map_err(|_| LetoError::Overflow {
            reason: "slice base offset conversion",
        })?;

        // Consume one expanded argument and advance the input/output axis
        // cursors in lockstep. Inlined by the compiler; the `for` loops below
        // drive it directly from `args` plus the implicit `fill` tail.
        let mut emit = |arg: SliceArg| -> Result<()> {
            match arg {
                SliceArg::All => {
                    if input_axis >= N || output_axis >= M {
                        return Err(slice_rank_error(N, M, args));
                    }
                    shape[output_axis] = self.shape[input_axis];
                    strides[output_axis] = self.strides[input_axis];
                    input_axis += 1;
                    output_axis += 1;
                }
                SliceArg::Range { start, end, step } => {
                    if input_axis >= N || output_axis >= M {
                        return Err(slice_rank_error(N, M, args));
                    }
                    let normalized = normalize_range(start, end, step, self.shape[input_axis])?;
                    let axis_offset = normalized
                        .start
                        .checked_mul(self.strides[input_axis])
                        .ok_or(LetoError::Overflow {
                            reason: "slice offset multiplication",
                        })?;
                    offset = offset.checked_add(axis_offset).ok_or(LetoError::Overflow {
                        reason: "slice offset calculation",
                    })?;
                    shape[output_axis] = normalized.len;
                    strides[output_axis] = if normalized.len == 1 {
                        0
                    } else {
                        self.strides[input_axis]
                            .checked_mul(normalized.step)
                            .ok_or(LetoError::Overflow {
                                reason: "slice stride multiplication",
                            })?
                    };
                    input_axis += 1;
                    output_axis += 1;
                }
                SliceArg::Index(index) => {
                    if input_axis >= N {
                        return Err(slice_rank_error(N, M, args));
                    }
                    let normalized = normalize_index(index, self.shape[input_axis])?;
                    let normalized =
                        isize::try_from(normalized).map_err(|_| LetoError::Overflow {
                            reason: "slice index conversion",
                        })?;
                    let axis_offset = normalized.checked_mul(self.strides[input_axis]).ok_or(
                        LetoError::Overflow {
                            reason: "slice index offset multiplication",
                        },
                    )?;
                    offset = offset.checked_add(axis_offset).ok_or(LetoError::Overflow {
                        reason: "slice index offset calculation",
                    })?;
                    input_axis += 1;
                }
                SliceArg::NewAxis => {
                    if output_axis >= M {
                        return Err(slice_rank_error(N, M, args));
                    }
                    shape[output_axis] = 1;
                    strides[output_axis] = 0;
                    output_axis += 1;
                }
                SliceArg::Ellipsis => {
                    return Err(LetoError::StorageError {
                        reason: "ellipsis must be expanded before slicing".to_string(),
                    });
                }
            }
            Ok(())
        };

        for &arg in args {
            if matches!(arg, SliceArg::Ellipsis) {
                for _ in 0..fill {
                    emit(SliceArg::All)?;
                }
            } else {
                emit(arg)?;
            }
        }
        // With no ellipsis, the trailing `fill` axes are implicit full-axis
        // selections appended at the end.
        if ellipsis_count == 0 {
            for _ in 0..fill {
                emit(SliceArg::All)?;
            }
        }

        if input_axis != N || output_axis != M {
            return Err(slice_rank_error(N, M, args));
        }
        if offset < 0 {
            return Err(LetoError::StorageError {
                reason: format!("slice accesses negative physical offset {offset}"),
            });
        }

        Layout::try_new(shape, strides, offset as usize)
    }

    /// Transpose the layout by permuting the axes.
    pub fn transpose(&self, axes: [usize; N]) -> Result<Self> {
        // Validate permutation
        let mut checked = [false; N];
        for &ax in &axes {
            if ax >= N {
                return Err(LetoError::StorageError {
                    reason: format!("Axis {} out of range for transposition", ax),
                });
            }
            if checked[ax] {
                return Err(LetoError::StorageError {
                    reason: format!("Duplicate axis {} in transposition", ax),
                });
            }
            checked[ax] = true;
        }

        let mut new_shape = [0usize; N];
        let mut new_strides = [0isize; N];
        for i in 0..N {
            new_shape[i] = self.shape[axes[i]];
            new_strides[i] = self.strides[axes[i]];
        }

        Self::try_new(new_shape, new_strides, self.offset)
    }
}

pub(super) fn slice_rank_error(
    input_rank: usize,
    output_rank: usize,
    args: &[SliceArg],
) -> LetoError {
    LetoError::StorageError {
        reason: format!(
            "slice rank mismatch: input rank {input_rank}, output rank {output_rank}, args {args:?}"
        ),
    }
}
