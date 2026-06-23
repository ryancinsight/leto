use crate::domain::error::{LetoError, Result};
use crate::domain::layout::kernels;

use super::Layout;

impl<const N: usize> Layout<N> {
    /// Returns the logical number of elements represented by this layout.
    pub fn size(&self) -> usize {
        self.checked_size()
            .expect("layout shape product must fit in usize")
    }

    /// Returns the logical number of elements with overflow validation.
    pub fn checked_size(&self) -> Result<usize> {
        kernels::shape_size(&self.shape)
    }

    /// Returns the minimum and maximum physical offsets spanned by this layout.
    pub fn min_max_offsets(&self) -> (usize, usize) {
        if N == 1 {
            let dim = self.shape[0];
            if dim == 0 {
                return (self.offset, self.offset);
            }
            let s = self.strides[0];
            let bound = (dim - 1) as isize * s;
            if bound >= 0 {
                (self.offset, self.offset + bound as usize)
            } else {
                let min_offset = self.offset as isize + bound;
                assert!(min_offset >= 0, "layout accesses negative physical offset");
                (min_offset as usize, self.offset)
            }
        } else {
            self.checked_min_max_offsets()
                .expect("layout physical offsets must be non-negative and fit in usize")
        }
    }

    /// Returns the minimum and maximum physical offsets with signed overflow validation.
    pub fn checked_min_max_offsets(&self) -> Result<(usize, usize)> {
        kernels::min_max_offsets(&self.shape, &self.strides, self.offset)
    }

    /// Validates that every addressable physical element is inside `storage_len`.
    pub fn validate_storage_len(&self, storage_len: usize) -> Result<()> {
        kernels::validate_storage(&self.shape, &self.strides, self.offset, storage_len)
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
