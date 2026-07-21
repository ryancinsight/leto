//! GAT-driven lending iterators for zero-copy array subview access.
//!
//! ## Motivation
//!
//! The standard [`Iterator`] trait requires `Item` to have a lifetime independent
//! of `&self`. This prevents returning views *borrowed from the iterator itself*
//! (the "streaming iterator" problem). GATs (`type Item<'this>`) solve this by
//! tying the yielded lifetime to `&mut self`.
//!
//! ## `LendingIterator`
//!
//! ```text
//! trait LendingIterator {
//!     type Item<'this> where Self: 'this;
//!     fn next(&mut self) -> Option<Self::Item<'_>>;
//! }
//! ```
//!
//! ## `Tiles` — non-overlapping rectangular tile views
//!
//! Partitions an array into `N`-D tiles of shape `tile_shape`. Each call to
//! [`Tiles::next`] yields a *borrowed* [`ArrayView`] whose lifetime is tied to
//! the `Tiles` object itself. For SIMD hot-paths this enables processing one
//! tile at a time without loading all windows into memory simultaneously.
//!
//! # Theorem (tile cover)
//!
//! For a parent of shape `s` and tile shape `t`, the tile grid has
//! `⌈sᵢ/tᵢ⌉` tiles along axis `i`. The last tile along each axis is clipped to
//! `sᵢ − ⌊sᵢ/tᵢ⌋·tᵢ` elements when `sᵢ mod tᵢ ≠ 0`.

use crate::application::view::ArrayView;
use crate::domain::layout::Layout;

// ── LendingIterator trait ──────────────────────────────────────────────────

/// A GAT-based streaming iterator that lends each item *from itself*, allowing
/// zero-copy views without requiring the data to outlive `self`.
///
/// Unlike [`Iterator`], `Item` may borrow from `&mut self`, so the same
/// backing store can be reused across calls (e.g. a single scratch buffer).
///
/// # Note on standard library compatibility
///
/// Once `LendingIterator` stabilizes in the standard library (RFC 3301) this
/// trait will become redundant. Until then, it is the correct GAT-native
/// pattern for building zero-copy window/tile iterators over owned data.
pub trait LendingIterator {
    /// The type yielded on each call to [`next`](Self::next), borrowing `self`
    /// for lifetime `'this`.
    type Item<'this>
    where
        Self: 'this;

    /// Advance the iterator and return the next item, or `None` if exhausted.
    fn next(&mut self) -> Option<Self::Item<'_>>;

    /// Count the remaining items (exhausts the iterator).
    #[inline]
    fn count_remaining(&mut self) -> usize {
        let mut n = 0;
        while self.next().is_some() {
            n += 1;
        }
        n
    }
}

// ── Tiles ──────────────────────────────────────────────────────────────────

/// GAT-driven non-overlapping tile iterator.
///
/// Partitions the backing slice (in row-major order) into rectangular tiles of
/// `tile_shape`. The last tile along each axis is automatically clipped to the
/// remaining extent when the array shape is not divisible by `tile_shape`.
///
/// Each [`LendingIterator::next`] call returns a zero-copy [`ArrayView`] into
/// the backing data; no element is copied. The yielded view's lifetime is tied
/// to `&'this self` (the GAT bound).
///
/// Construct via [`Array::tiles`](crate::application::array::Array::tiles) or
/// [`ArrayView::tiles`](crate::application::view::ArrayView::tiles).
pub struct Tiles<'a, T, const N: usize> {
    data: &'a [T],
    parent_layout: Layout<N>,
    tile_shape: [usize; N],
    /// Tile-grid shape: `tile_grid[i] = ceil(parent[i] / tile[i])`.
    tile_grid: [usize; N],
    /// Total number of tiles.
    total: usize,
    /// Index of the next tile to yield (row-major).
    cursor: usize,
}

impl<'a, T, const N: usize> Tiles<'a, T, N> {
    /// Construct a tile iterator from a data slice and its layout.
    ///
    /// # Errors
    ///
    /// Returns `None` if any `tile_shape[i] == 0` (zero-size tiles are
    /// not meaningful) or if `N == 0`.
    #[must_use]
    pub fn new(data: &'a [T], parent_layout: Layout<N>, tile_shape: [usize; N]) -> Option<Self> {
        if N == 0 {
            return None;
        }
        if tile_shape.iter().any(|&t| t == 0) {
            return None;
        }
        let parent_shape = parent_layout.shape;
        let mut tile_grid = [0usize; N];
        let mut total = 1usize;
        for i in 0..N {
            tile_grid[i] = (parent_shape[i] + tile_shape[i] - 1) / tile_shape[i];
            total = total.checked_mul(tile_grid[i])?;
        }
        Some(Self {
            data,
            parent_layout,
            tile_shape,
            tile_grid,
            total,
            cursor: 0,
        })
    }

    /// Total number of tiles (including partial boundary tiles).
    #[must_use]
    #[inline]
    pub fn total_tiles(&self) -> usize {
        self.total
    }

    /// Convert a flat tile index to tile-grid multi-index (row-major).
    fn flat_to_tile_index(&self, flat: usize) -> [usize; N] {
        let mut idx = [0usize; N];
        let mut rem = flat;
        for i in (0..N).rev() {
            idx[i] = rem % self.tile_grid[i];
            rem /= self.tile_grid[i];
        }
        idx
    }

    /// Compute the actual (possibly clipped) shape for tile `tile_idx`.
    fn tile_extent(&self, tile_idx: &[usize; N]) -> [usize; N] {
        let parent_shape = self.parent_layout.shape;
        let mut extent = [0usize; N];
        for i in 0..N {
            let start = tile_idx[i] * self.tile_shape[i];
            let end = (start + self.tile_shape[i]).min(parent_shape[i]);
            extent[i] = end - start;
        }
        extent
    }
}

impl<'a, T: Copy, const N: usize> LendingIterator for Tiles<'a, T, N> {
    type Item<'this>
        = ArrayView<'this, T, N>
    where
        Self: 'this;

    fn next(&mut self) -> Option<Self::Item<'_>> {
        if self.cursor >= self.total {
            return None;
        }

        let tile_idx = self.flat_to_tile_index(self.cursor);
        let extent = self.tile_extent(&tile_idx);

        // Starting element index in row-major: Σᵢ tile_idx[i]·tile_shape[i]·stride[i]
        let origin: [usize; N] = std::array::from_fn(|i| tile_idx[i] * self.tile_shape[i]);
        let offset = self.parent_layout.offset_of(origin).ok()?;

        let view_layout = Layout::new(
            extent,
            self.parent_layout.strides,
            offset,
        );

        self.cursor += 1;

        Some(ArrayView::new(view_layout, self.data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::array::Array;
    use crate::infrastructure::storage::{Storage, VecStorage};

    fn array2x3() -> Array<f64, VecStorage<f64>, 2> {
        // [[0,1,2],[3,4,5]]
        Array::from_shape_vec([2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("shape matches data")
    }

    #[test]
    fn tile_count_exact_divisible() {
        let a = array2x3();
        let tiles = Tiles::new(a.storage().as_slice(), a.layout(), [1, 3])
            .expect("valid tile shape");
        // 2x3 / 1x3 = 2 tiles
        assert_eq!(tiles.total_tiles(), 2);
    }

    #[test]
    fn tile_count_partial_boundary() {
        let a = array2x3();
        let tiles = Tiles::new(a.storage().as_slice(), a.layout(), [1, 2])
            .expect("valid tile shape");
        // 2 rows x ceil(3/2) = 2x2 = 4 tiles (last column tile has width 1)
        assert_eq!(tiles.total_tiles(), 4);
    }

    #[test]
    fn tiles_cover_all_elements() {
        let a = array2x3();
        let mut tiles = Tiles::new(a.storage().as_slice(), a.layout(), [1, 2])
            .expect("valid tile shape");
        let mut collected = Vec::new();
        while let Some(tile) = tiles.next() {
            for val in tile.iter() {
                collected.push(*val);
            }
        }
        // All 6 elements covered (row 0 tiles [0,1], [2]; row 1 tiles [3,4], [5])
        let mut sorted = collected.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(sorted, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn tile_yields_correct_values_exact() {
        let a = array2x3();
        let mut tiles = Tiles::new(a.storage().as_slice(), a.layout(), [1, 3])
            .expect("valid tile shape");
        let row0 = tiles.next().expect("first tile");
        assert_eq!(
            row0.iter().copied().collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0]
        );
        let row1 = tiles.next().expect("second tile");
        assert_eq!(
            row1.iter().copied().collect::<Vec<_>>(),
            vec![3.0, 4.0, 5.0]
        );
        assert!(tiles.next().is_none());
    }

    #[test]
    fn lending_iterator_manual_loop() {
        let a = array2x3();
        let mut tiles = Tiles::new(a.storage().as_slice(), a.layout(), [1, 3])
            .expect("valid tile shape");
        let mut sums = Vec::new();
        while let Some(tile) = tiles.next() {
            sums.push(tile.iter().copied().sum::<f64>());
        }
        assert_eq!(sums, vec![3.0, 12.0]); // 0+1+2 = 3, 3+4+5 = 12
    }

    #[test]
    fn reject_zero_tile_shape() {
        let a = array2x3();
        assert!(Tiles::new(a.storage().as_slice(), a.layout(), [0, 3]).is_none());
    }

    #[test]
    fn count_remaining_exhausts_iterator() {
        let a = array2x3();
        let mut tiles = Tiles::new(a.storage().as_slice(), a.layout(), [1, 3])
            .expect("valid tile shape");
        assert_eq!(tiles.count_remaining(), 2);
        // Iterator is now exhausted
        assert!(tiles.next().is_none());
    }
}
