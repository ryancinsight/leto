/// Convert a flat row-major logical index into an N-dimensional index.
#[inline(always)]
pub(crate) fn index_from_flat<const N: usize>(flat: usize, shape: &[usize; N]) -> [usize; N] {
    let mut index = [0usize; N];
    let mut temp = flat;
    for axis in (0..N).rev() {
        if shape[axis] > 0 {
            index[axis] = temp % shape[axis];
            temp /= shape[axis];
        }
    }
    index
}

/// Row-major traversal descriptor for strided logical iteration.
///
/// The descriptor groups the logical element space into innermost rows. Each
/// operation computes offsets once per row and then walks the last axis by
/// stride increments, avoiding per-element div/mod decomposition.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowMajorTraversal<const N: usize> {
    shape: [usize; N],
    inner: usize,
    rows: usize,
}

impl<const N: usize> RowMajorTraversal<N> {
    #[inline]
    pub(crate) fn new(size: usize, shape: [usize; N]) -> Option<Self> {
        let inner = if N == 0 { 1 } else { shape[N - 1] };
        if inner == 0 || size == 0 {
            return None;
        }
        Some(Self {
            shape,
            inner,
            rows: size / inner,
        })
    }

    #[inline]
    pub(crate) const fn inner(self) -> usize {
        self.inner
    }

    #[inline]
    pub(crate) const fn rows(self) -> usize {
        self.rows
    }

    #[inline]
    pub(crate) fn base_index(self, row: usize) -> [usize; N] {
        index_from_flat(row * self.inner, &self.shape)
    }

    #[inline]
    pub(crate) fn chunk_rows_for(self, target_elements: usize) -> usize {
        (target_elements / self.inner.max(1)).max(1)
    }

    #[inline]
    pub(crate) const fn last_axis_stride(self, layout: leto::Layout<N>) -> isize {
        if N == 0 {
            0
        } else {
            layout.strides[N - 1]
        }
    }
}
