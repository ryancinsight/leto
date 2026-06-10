use crate::domain::error::{LetoError, Result};

use super::Layout;

impl<const N: usize> Layout<N> {
    /// Returns the logical number of elements represented by this layout.
    pub fn size(&self) -> usize {
        self.checked_size()
            .expect("layout shape product must fit in usize")
    }

    /// Returns the logical number of elements with overflow validation.
    pub fn checked_size(&self) -> Result<usize> {
        if self.shape.contains(&0) {
            Ok(0)
        } else {
            self.shape.iter().try_fold(1usize, |size, &dim| {
                size.checked_mul(dim).ok_or(LetoError::Overflow {
                    reason: "layout shape product",
                })
            })
        }
    }

    /// Returns the minimum and maximum physical offsets spanned by this layout.
    pub fn min_max_offsets(&self) -> (usize, usize) {
        self.checked_min_max_offsets()
            .expect("layout physical offsets must be non-negative and fit in usize")
    }

    /// Returns the minimum and maximum physical offsets with signed overflow validation.
    pub fn checked_min_max_offsets(&self) -> Result<(usize, usize)> {
        if N == 0 {
            return Ok((self.offset, self.offset));
        }
        if self.shape.contains(&0) {
            return Ok((self.offset, self.offset));
        }

        let mut min_offset = isize::try_from(self.offset).map_err(|_| LetoError::Overflow {
            reason: "layout base offset conversion",
        })?;
        let mut max_offset = min_offset;

        for i in 0..N {
            let s = self.strides[i];
            let len = self.shape[i];
            let bound1 = 0isize;
            let len_minus_one = isize::try_from(len - 1).map_err(|_| LetoError::Overflow {
                reason: "layout dimension bound conversion",
            })?;
            let bound2 = len_minus_one.checked_mul(s).ok_or(LetoError::Overflow {
                reason: "layout dimension bound multiplication",
            })?;
            min_offset = min_offset
                .checked_add(bound1.min(bound2))
                .ok_or(LetoError::Overflow {
                    reason: "layout minimum offset accumulation",
                })?;
            max_offset = max_offset
                .checked_add(bound1.max(bound2))
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

    /// Validates that every addressable physical element is inside `storage_len`.
    pub fn validate_storage_len(&self, storage_len: usize) -> Result<()> {
        if self.checked_size()? == 0 {
            return Ok(());
        }

        let (min_offset, max_offset) = self.checked_min_max_offsets()?;
        if min_offset >= storage_len || max_offset >= storage_len {
            return Err(LetoError::StorageError {
                reason: format!(
                    "storage length {storage_len} does not cover layout physical offsets {min_offset}..={max_offset}"
                ),
            });
        }

        Ok(())
    }

    /// Returns true when multiple logical mutable indices can address one element.
    pub fn has_zero_stride_aliasing(&self) -> bool {
        self.shape
            .iter()
            .zip(self.strides.iter())
            .any(|(&dim, &stride)| dim > 1 && stride == 0)
    }

    /// Reinterpret a dense row-major layout with a new shape.
    ///
    /// This preserves logical row-major element order and performs no
    /// materialization. Strided, Fortran-order, or broadcasted layouts must be
    /// materialized with `to_contiguous` before reshaping.
    pub fn reshape<const M: usize>(&self, shape: [usize; M]) -> Result<Layout<M>> {
        let target = Layout::<M>::c_contiguous(shape)?;
        if self.checked_size()? != target.checked_size()? {
            return Err(LetoError::ShapeMismatch {
                lhs: self.shape.to_vec(),
                rhs: shape.to_vec(),
            });
        }
        if !self.is_c_dense() {
            return Err(LetoError::StorageError {
                reason: "reshape requires a dense row-major layout".to_string(),
            });
        }

        Ok(Layout {
            shape,
            strides: target.strides,
            offset: self.offset,
        })
    }
}
