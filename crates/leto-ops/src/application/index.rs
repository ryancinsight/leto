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

    #[cfg(feature = "parallel")]
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

/// Cache-line micro-tile geometry over the last two axes (rank ≥ 2).
///
/// A column-strided operand (|last-axis stride| ≥ elements-per-line) touches
/// a fresh cache line per element under plain row-walk, wasting
/// `1 - 1/lane` of every line. Tiling the last two axes at
/// `lane = 64 / size_of::<T>()` elements per side makes each touched line
/// fully consumed before eviction: within a tile the strided operand revisits
/// the same `tile` lines across `tile` rows. The tile size is derived from
/// the 64-byte line, not tuned.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TileGeometry<const N: usize> {
    shape: [usize; N],
    slabs: usize,
    height: usize,
    width: usize,
    tile: usize,
}

impl<const N: usize> TileGeometry<N> {
    /// `None` when rank < 2 or the space is empty (callers keep row-walk).
    #[inline]
    pub(crate) fn new(size: usize, shape: [usize; N], tile: usize) -> Option<Self> {
        if N < 2 || size == 0 {
            return None;
        }
        let height = shape[N - 2];
        let width = shape[N - 1];
        if height == 0 || width == 0 || tile == 0 {
            return None;
        }
        Some(Self {
            shape,
            slabs: size / (height * width),
            height,
            width,
            tile,
        })
    }

    #[inline]
    pub(crate) const fn slabs(self) -> usize {
        self.slabs
    }

    #[inline]
    pub(crate) const fn height(self) -> usize {
        self.height
    }

    #[inline]
    pub(crate) const fn width(self) -> usize {
        self.width
    }

    #[inline]
    pub(crate) const fn tile(self) -> usize {
        self.tile
    }

    /// Row blocks per slab: `ceil(height / tile)`.
    #[inline]
    pub(crate) const fn row_blocks(self) -> usize {
        self.height.div_ceil(self.tile)
    }

    /// Logical index of a slab's `[.., 0, 0]` corner.
    #[inline]
    pub(crate) fn slab_base_index(self, slab: usize) -> [usize; N] {
        index_from_flat(slab * self.height * self.width, &self.shape)
    }
}

/// Elements of `T` per 64-byte cache line (≥ 1): the analytic micro-tile side.
#[inline]
pub(crate) const fn line_elements<T>() -> usize {
    let lane = 64 / core::mem::size_of::<T>();
    if lane == 0 {
        1
    } else {
        lane
    }
}
