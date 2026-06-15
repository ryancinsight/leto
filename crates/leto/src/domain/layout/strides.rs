use crate::domain::error::{LetoError, Result};

use super::Layout;

impl<const N: usize> Layout<N> {
    /// Create a C-contiguous (row-major) layout for a given shape.
    pub fn c_contiguous(shape: [usize; N]) -> Result<Self> {
        let mut strides = [0isize; N];
        crate::domain::layout::kernels::c_contiguous_strides(&shape, &mut strides)?;
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
                let dim = isize::try_from(dim).map_err(|_| LetoError::Overflow {
                    reason: "F-contiguous dimension conversion",
                })?;
                stride = match stride.checked_mul(dim) {
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
}
