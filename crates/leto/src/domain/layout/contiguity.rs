use super::Layout;

impl<const N: usize> Layout<N> {
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
            let dim = match isize::try_from(self.shape[i]) {
                Ok(dim) => dim,
                Err(_) => return false,
            };
            expected_stride = match expected_stride.checked_mul(dim) {
                Some(stride) => stride,
                None => return false,
            };
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
            let dim = match isize::try_from(self.shape[i]) {
                Ok(dim) => dim,
                Err(_) => return false,
            };
            expected_stride = match expected_stride.checked_mul(dim) {
                Some(stride) => stride,
                None => return false,
            };
        }
        true
    }
}
