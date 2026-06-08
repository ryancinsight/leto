use crate::domain::error::{LetoError, Result};

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
        Self { shape, strides, offset }
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
                    None => return Err(LetoError::Overflow { reason: "C-contiguous stride multiplication" }),
                };
            }
        }
        Ok(Self { shape, strides, offset: 0 })
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
                    None => return Err(LetoError::Overflow { reason: "F-contiguous stride multiplication" }),
                };
            }
        }
        Ok(Self { shape, strides, offset: 0 })
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
