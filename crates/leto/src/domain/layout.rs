use crate::domain::error::{LetoError, Result};
use crate::domain::slice::{normalize_index, normalize_range, SliceArg};

/// Represents an N-dimensional strided layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout<const N: usize> {
    /// The shape of the array (size of each dimension).
    pub shape: [usize; N],
    /// The stride of each dimension in elements (not bytes).
    pub strides: [isize; N],
    /// The starting offset in the storage buffer.
    pub offset: usize,
}

impl<const N: usize> Layout<N> {
    /// Create a new layout with explicit shape, strides, and offset.
    pub const fn new(shape: [usize; N], strides: [isize; N], offset: usize) -> Self {
        Self {
            shape,
            strides,
            offset,
        }
    }

    /// Create a C-contiguous (row-major) layout for a given shape.
    pub fn c_contiguous(shape: [usize; N]) -> Result<Self> {
        let mut strides = [0isize; N];
        let mut stride = 1isize;
        for i in (0..N).rev() {
            strides[i] = stride;
            let dim = shape[i];
            if dim == 0 {
                // If any dimension is zero, strides are set but size is zero
                stride = 0;
            } else {
                stride = match stride.checked_mul(dim as isize) {
                    Some(s) => s,
                    None => {
                        return Err(LetoError::Overflow {
                            reason: "C-contiguous stride multiplication",
                        })
                    }
                };
            }
        }
        Ok(Self {
            shape,
            strides,
            offset: 0,
        })
    }

    /// Create a Fortran-contiguous (column-major) layout for a given shape.
    pub fn f_contiguous(shape: [usize; N]) -> Result<Self> {
        let mut strides = [0isize; N];
        let mut stride = 1isize;
        for i in 0..N {
            strides[i] = stride;
            let dim = shape[i];
            if dim == 0 {
                stride = 0;
            } else {
                stride = match stride.checked_mul(dim as isize) {
                    Some(s) => s,
                    None => {
                        return Err(LetoError::Overflow {
                            reason: "F-contiguous stride multiplication",
                        })
                    }
                };
            }
        }
        Ok(Self {
            shape,
            strides,
            offset: 0,
        })
    }

    /// Returns the logical number of elements represented by this layout.
    pub fn size(&self) -> usize {
        if self.shape.contains(&0) {
            0
        } else {
            self.shape.iter().product()
        }
    }

    /// Returns the minimum and maximum physical offsets spanned by this layout.
    pub fn min_max_offsets(&self) -> (usize, usize) {
        if N == 0 {
            return (self.offset, self.offset);
        }
        if self.shape.contains(&0) {
            return (self.offset, self.offset);
        }

        let mut min_offset = self.offset as isize;
        let mut max_offset = self.offset as isize;

        for i in 0..N {
            let s = self.strides[i];
            let len = self.shape[i];
            let bound1 = 0isize;
            let bound2 = (len - 1) as isize * s;
            min_offset += bound1.min(bound2);
            max_offset += bound1.max(bound2);
        }

        (min_offset as usize, max_offset as usize)
    }

    /// Check if the layout is C-contiguous (row-major).
    pub fn is_c_contiguous(&self) -> bool {
        if self.offset != 0 {
            return false;
        }
        let mut expected_stride = 1isize;
        for i in (0..N).rev() {
            if self.shape[i] == 1 {
                // Stride of 1-sized dimension does not affect contiguity
                continue;
            }
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i] as isize;
        }
        true
    }

    /// Check if the layout is Fortran-contiguous (column-major).
    pub fn is_f_contiguous(&self) -> bool {
        if self.offset != 0 {
            return false;
        }
        let mut expected_stride = 1isize;
        for i in 0..N {
            if self.shape[i] == 1 {
                continue;
            }
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i] as isize;
        }
        true
    }

    /// Compute the physical element offset for a given multi-dimensional index.
    pub fn offset_of(&self, index: [usize; N]) -> Result<usize> {
        let mut offset = self.offset as isize;
        for i in 0..N {
            if index[i] >= self.shape[i] {
                return Err(LetoError::OutOfBounds {
                    index: index.to_vec(),
                    shape: self.shape.to_vec(),
                });
            }
            offset += index[i] as isize * self.strides[i];
        }
        Ok(offset as usize)
    }

    /// Slice the layout on each axis given a slice definition `(start, end, step)`.
    pub fn slice(&self, ranges: &[(usize, usize, isize); N]) -> Result<Self> {
        let mut new_shape = [0usize; N];
        let mut new_strides = [0isize; N];
        let mut new_offset = self.offset as isize;

        for i in 0..N {
            let (start, end, step) = ranges[i];
            let dim_len = self.shape[i];

            if step == 0 {
                return Err(LetoError::IncompatibleSlice {
                    range: (start, end),
                    shape: self.shape.to_vec(),
                });
            }

            if start > dim_len || end > dim_len {
                return Err(LetoError::IncompatibleSlice {
                    range: (start, end),
                    shape: self.shape.to_vec(),
                });
            }

            if step > 0 {
                if start > end {
                    new_shape[i] = 0;
                } else {
                    new_shape[i] = (end - start - 1) / step as usize + 1;
                }
                new_offset += start as isize * self.strides[i];
                new_strides[i] = self.strides[i] * step;
            } else {
                // negative step: start is higher index, end is lower index (exclusive)
                if start < end {
                    new_shape[i] = 0;
                } else {
                    new_shape[i] = (start - end - 1) / (-step) as usize + 1;
                }
                new_offset += start as isize * self.strides[i];
                new_strides[i] = self.strides[i] * step;
            }
        }

        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: new_offset as usize,
        })
    }

    /// Slice the layout with ndarray-style arguments.
    ///
    /// This supports full-axis ranges, optional signed bounds, negative indices,
    /// negative strides, integer indexing that removes an axis, inserted new axes,
    /// and one ellipsis expansion. The caller specifies the output rank `M`.
    pub fn slice_with<const M: usize>(&self, args: &[SliceArg]) -> Result<Layout<M>> {
        let expanded = self.expand_slice_args(args)?;
        let mut shape = [0usize; M];
        let mut strides = [0isize; M];
        let mut input_axis = 0usize;
        let mut output_axis = 0usize;
        let mut offset = self.offset as isize;

        for arg in expanded {
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
                    strides[output_axis] = self.strides[input_axis]
                        .checked_mul(normalized.step)
                        .ok_or(LetoError::Overflow {
                            reason: "slice stride multiplication",
                        })?;
                    input_axis += 1;
                    output_axis += 1;
                }
                SliceArg::Index(index) => {
                    if input_axis >= N {
                        return Err(slice_rank_error(N, M, args));
                    }
                    let normalized = normalize_index(index, self.shape[input_axis])?;
                    let axis_offset = (normalized as isize)
                        .checked_mul(self.strides[input_axis])
                        .ok_or(LetoError::Overflow {
                            reason: "slice index offset multiplication",
                        })?;
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
        }

        if input_axis != N || output_axis != M {
            return Err(slice_rank_error(N, M, args));
        }
        if offset < 0 {
            return Err(LetoError::StorageError {
                reason: format!("slice accesses negative physical offset {offset}"),
            });
        }

        Ok(Layout {
            shape,
            strides,
            offset: offset as usize,
        })
    }

    fn expand_slice_args(&self, args: &[SliceArg]) -> Result<Vec<SliceArg>> {
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
        let mut expanded = Vec::with_capacity(args.len() + fill);
        let mut inserted_implicit_tail = false;

        for &arg in args {
            if matches!(arg, SliceArg::Ellipsis) {
                for _ in 0..fill {
                    expanded.push(SliceArg::All);
                }
                inserted_implicit_tail = true;
            } else {
                expanded.push(arg);
            }
        }

        if ellipsis_count == 0 && !inserted_implicit_tail {
            for _ in 0..fill {
                expanded.push(SliceArg::All);
            }
        }

        Ok(expanded)
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

        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Broadcast the current layout to a target shape of length `M` where `M >= N`.
    pub fn broadcast<const M: usize>(&self, target_shape: [usize; M]) -> Result<Layout<M>> {
        if M < N {
            return Err(LetoError::IncompatibleBroadcast {
                from: self.shape.to_vec(),
                to: target_shape.to_vec(),
            });
        }

        let mut new_shape = [0usize; M];
        let mut new_strides = [0isize; M];
        let shift = M - N;

        // Populate prepended dimensions
        for i in 0..shift {
            new_shape[i] = target_shape[i];
            new_strides[i] = 0; // Stride is 0 for broadcasted dimensions
        }

        // Populate matching dimensions
        for i in 0..N {
            let target_idx = i + shift;
            let target_dim = target_shape[target_idx];
            let source_dim = self.shape[i];

            if source_dim == target_dim {
                new_shape[target_idx] = target_dim;
                new_strides[target_idx] = self.strides[i];
            } else if source_dim == 1 {
                new_shape[target_idx] = target_dim;
                new_strides[target_idx] = 0; // Stride is 0 when broadcasting a 1-sized dim
            } else {
                return Err(LetoError::IncompatibleBroadcast {
                    from: self.shape.to_vec(),
                    to: target_shape.to_vec(),
                });
            }
        }

        Ok(Layout {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }
}

fn slice_rank_error(input_rank: usize, output_rank: usize, args: &[SliceArg]) -> LetoError {
    LetoError::StorageError {
        reason: format!(
            "slice rank mismatch: input rank {input_rank}, output rank {output_rank}, args {args:?}"
        ),
    }
}
