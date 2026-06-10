use super::Layout;

impl<const N: usize> Layout<N> {
    /// Returns true when the strides match the C (row-major) pattern for this
    /// shape, independent of the base offset.
    ///
    /// Size-1 axes are ignored because their stride cannot affect element
    /// addressing. This is the offset-independent half of [`is_c_contiguous`];
    /// it is the predicate used to decide whether the logical elements occupy a
    /// dense `[offset, offset + size)` block in row-major order.
    ///
    /// [`is_c_contiguous`]: Layout::is_c_contiguous
    fn matches_c_strides(&self) -> bool {
        let mut expected_stride = 1isize;
        for i in (0..N).rev() {
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

    /// Returns true when the strides match the Fortran (column-major) pattern
    /// for this shape, independent of the base offset.
    fn matches_f_strides(&self) -> bool {
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

    /// Check if the layout is canonically C-contiguous (row-major) at offset 0.
    pub fn is_c_contiguous(&self) -> bool {
        self.offset == 0 && self.matches_c_strides()
    }

    /// Check if the layout is canonically Fortran-contiguous (column-major) at
    /// offset 0.
    pub fn is_f_contiguous(&self) -> bool {
        self.offset == 0 && self.matches_f_strides()
    }

    /// Returns true when the logical elements occupy a single dense physical
    /// block in some memory order (C or F), independent of the base offset.
    ///
    /// This is the predicate behind memory-order slice exposure: a view sliced
    /// or iterated to a non-zero offset can still be a contiguous block, which
    /// the canonical [`is_c_contiguous`]/[`is_f_contiguous`] predicates reject
    /// because they pin the offset to 0.
    ///
    /// [`is_c_contiguous`]: Layout::is_c_contiguous
    /// [`is_f_contiguous`]: Layout::is_f_contiguous
    pub fn is_contiguous(&self) -> bool {
        self.matches_c_strides() || self.matches_f_strides()
    }

    /// Returns true when the strides match the C pattern for this shape,
    /// independent of the base offset. Row-major dense block predicate.
    pub fn is_c_dense(&self) -> bool {
        self.matches_c_strides()
    }
}
