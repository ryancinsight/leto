//! Non-overlapping rectangular tile views.
//!
//! Partitions an array into `N`-D tiles of shape `tile_shape`. Each tile is a
//! zero-copy [`ArrayView`] into the *parent* slice — it borrows the parent for
//! `'a` rather than borrowing the iterator — so `Tiles` is a plain [`Iterator`]
//! and composes with the whole adaptor ecosystem. A streaming-iterator
//! signature (`type Item<'this>` tied to `&mut self`) would forfeit
//! [`IntoIterator`], `zip`, `enumerate`, `rev`, [`ExactSizeIterator`] and every
//! parallel bridge in exchange for no capability, so it is not used here.
//!
//! # Theorem (tile cover)
//!
//! For a parent of shape `s` and tile shape `t`, the tile grid has
//! `⌈sᵢ/tᵢ⌉` tiles along axis `i`. The last tile along each axis is clipped to
//! `sᵢ − ⌊sᵢ/tᵢ⌋·tᵢ` elements when `sᵢ mod tᵢ ≠ 0`.

use crate::application::view::ArrayView;
use crate::domain::layout::Layout;

/// Non-overlapping tile iterator over a strided parent layout.
///
/// Partitions the backing slice (in row-major tile order) into rectangular tiles
/// of `tile_shape`. The last tile along each axis is automatically clipped to the
/// remaining extent when the array shape is not divisible by `tile_shape`.
///
/// Each item is a zero-copy [`ArrayView`] into the backing data; no element is
/// copied. Items borrow the parent slice for `'a`, not the iterator, so `Tiles`
/// is a plain [`Iterator`] and additionally satisfies [`DoubleEndedIterator`]
/// and [`ExactSizeIterator`].
///
/// Construct with [`Tiles::new`].
///
/// # Examples
///
/// ```
/// use leto::{Array2, Storage, Tiles};
///
/// // 3x5 row-major, tiled 2x2 — a tiling that does not divide it evenly.
/// let a = Array2::from_shape_vec([3, 5], (0..15).map(f64::from).collect::<Vec<_>>())?;
/// let tiles = Tiles::new(a.storage().as_slice(), a.layout(), [2, 2]).expect("valid tile shape");
///
/// assert_eq!(tiles.len(), 6);
/// let shapes: Vec<[usize; 2]> = tiles.map(|tile| tile.shape()).collect();
/// // The right column, bottom row, and bottom-right corner are clipped.
/// assert_eq!(shapes, [[2, 2], [2, 2], [2, 1], [1, 2], [1, 2], [1, 1]]);
/// # Ok::<(), leto::LetoError>(())
/// ```
pub struct Tiles<'a, T, const N: usize> {
    data: &'a [T],
    parent_layout: Layout<N>,
    tile_shape: [usize; N],
    /// Tile-grid shape: `tile_grid[i] = ceil(parent[i] / tile[i])`.
    tile_grid: [usize; N],
    /// Total number of tiles in the tiling; fixed at construction.
    total: usize,
    /// Flat index of the next tile yielded from the front.
    front: usize,
    /// One past the flat index of the next tile yielded from the back.
    back: usize,
}

impl<'a, T, const N: usize> Tiles<'a, T, N> {
    /// Construct a tile iterator from a data slice and its layout.
    ///
    /// # Errors
    ///
    /// Returns `None` if `N == 0`, if any `tile_shape[i] == 0` (zero-size tiles
    /// are not meaningful), if the tile grid overflows `usize`, or if
    /// `parent_layout` addresses physical offsets outside `data`.
    ///
    /// Rejecting an out-of-range layout here is what makes the
    /// [`ExactSizeIterator`] contract sound: every tile origin is then a valid
    /// parent index, so iteration can never terminate early.
    #[must_use]
    pub fn new(data: &'a [T], parent_layout: Layout<N>, tile_shape: [usize; N]) -> Option<Self> {
        if N == 0 {
            return None;
        }
        if tile_shape.contains(&0) {
            return None;
        }
        parent_layout.validate_storage_len(data.len()).ok()?;
        let parent_shape = parent_layout.shape();
        let mut tile_grid = [0usize; N];
        let mut total = 1usize;
        for i in 0..N {
            tile_grid[i] = parent_shape[i].div_ceil(tile_shape[i]);
            total = total.checked_mul(tile_grid[i])?;
        }
        Some(Self {
            data,
            parent_layout,
            tile_shape,
            tile_grid,
            total,
            front: 0,
            back: total,
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
        let parent_shape = self.parent_layout.shape();
        let mut extent = [0usize; N];
        for i in 0..N {
            let start = tile_idx[i] * self.tile_shape[i];
            let end = (start + self.tile_shape[i]).min(parent_shape[i]);
            extent[i] = end - start;
        }
        extent
    }

    /// Build the view for flat tile index `flat`, which must be `< self.total`.
    ///
    /// Borrows `self` only to read `Copy` state; the returned view borrows the
    /// parent slice for `'a`, which is what lets `Tiles` be a plain `Iterator`.
    #[inline]
    fn view_at(&self, flat: usize) -> ArrayView<'a, T, N> {
        let tile_idx = self.flat_to_tile_index(flat);
        let extent = self.tile_extent(&tile_idx);

        // Starting element index in row-major: Σᵢ tile_idx[i]·tile_shape[i]·stride[i]
        let origin: [usize; N] = std::array::from_fn(|i| tile_idx[i] * self.tile_shape[i]);
        // `tile_idx[i] < ceil(sᵢ/tᵢ)` implies `origin[i] ≤ (ceil(sᵢ/tᵢ) − 1)·tᵢ < sᵢ`,
        // so the origin is in bounds; `Tiles::new` validated that every in-bounds
        // parent index resolves to a physical offset inside `data`.
        let offset = self
            .parent_layout
            .offset_of(origin)
            .expect("invariant: tile origin is an in-bounds index of a validated parent layout");

        let view_layout =
            Layout::from_parts_unchecked(extent, self.parent_layout.strides(), offset);
        ArrayView::new(view_layout, self.data)
    }
}

impl<'a, T, const N: usize> Iterator for Tiles<'a, T, N> {
    type Item = ArrayView<'a, T, N>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        let view = self.view_at(self.front);
        self.front += 1;
        Some(view)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;
        (remaining, Some(remaining))
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for Tiles<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        Some(self.view_at(self.back))
    }
}

impl<T, const N: usize> ExactSizeIterator for Tiles<'_, T, N> {}

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

    /// 3x5 row-major holding `0..15`. Tiled `[2, 2]` it clips on the right
    /// column, the bottom row, and the bottom-right corner simultaneously.
    fn array3x5() -> Array<f64, VecStorage<f64>, 2> {
        Array::from_shape_vec([3, 5], (0..15).map(f64::from).collect::<Vec<_>>())
            .expect("shape matches data")
    }

    /// Tiles of `array3x5()` under a `[2, 2]` tiling, in row-major tile order.
    const RAGGED_3X5_TILES: [(&[usize; 2], &[f64]); 6] = [
        (&[2, 2], &[0.0, 1.0, 5.0, 6.0]),
        (&[2, 2], &[2.0, 3.0, 7.0, 8.0]),
        (&[2, 1], &[4.0, 9.0]),
        (&[1, 2], &[10.0, 11.0]),
        (&[1, 2], &[12.0, 13.0]),
        (&[1, 1], &[14.0]),
    ];

    fn values(tile: &ArrayView<'_, f64, 2>) -> Vec<f64> {
        tile.iter().copied().collect()
    }

    #[test]
    fn tile_count_exact_divisible() {
        let a = array2x3();
        let tiles =
            Tiles::new(a.storage().as_slice(), a.layout(), [1, 3]).expect("valid tile shape");
        // 2x3 / 1x3 = 2 tiles
        assert_eq!(tiles.total_tiles(), 2);
    }

    #[test]
    fn tile_count_partial_boundary() {
        let a = array2x3();
        let tiles =
            Tiles::new(a.storage().as_slice(), a.layout(), [1, 2]).expect("valid tile shape");
        // 2 rows x ceil(3/2) = 2x2 = 4 tiles (last column tile has width 1)
        assert_eq!(tiles.total_tiles(), 4);
    }

    #[test]
    fn tiles_cover_all_elements() {
        let a = array2x3();
        let tiles =
            Tiles::new(a.storage().as_slice(), a.layout(), [1, 2]).expect("valid tile shape");
        let mut collected = Vec::new();
        for tile in tiles {
            collected.extend(tile.iter().copied());
        }
        // All 6 elements covered (row 0 tiles [0,1], [2]; row 1 tiles [3,4], [5])
        collected.sort_by(f64::total_cmp);
        assert_eq!(collected, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn tile_yields_correct_values_exact() {
        let a = array2x3();
        let mut tiles =
            Tiles::new(a.storage().as_slice(), a.layout(), [1, 3]).expect("valid tile shape");
        let row0 = tiles.next().expect("first tile");
        assert_eq!(values(&row0), vec![0.0, 1.0, 2.0]);
        let row1 = tiles.next().expect("second tile");
        assert_eq!(values(&row1), vec![3.0, 4.0, 5.0]);
        assert!(tiles.next().is_none());
    }

    #[test]
    fn reject_zero_tile_shape() {
        let a = array2x3();
        assert!(Tiles::new(a.storage().as_slice(), a.layout(), [0, 3]).is_none());
    }

    #[test]
    fn reject_layout_exceeding_data() {
        let a = array2x3();
        // A 4x3 layout over a 6-element slice addresses offsets 0..=11.
        let oversized = Layout::from_parts_unchecked([4, 3], [3, 1], 0);
        assert!(Tiles::new(a.storage().as_slice(), oversized, [1, 3]).is_none());
    }

    // ── Iterator composition (what the GAT previously precluded) ───────────

    #[test]
    fn for_loop_over_tiles_uses_into_iterator() {
        let a = array2x3();
        let tiles =
            Tiles::new(a.storage().as_slice(), a.layout(), [1, 2]).expect("valid tile shape");
        let mut seen = Vec::new();
        for tile in tiles {
            seen.push(values(&tile));
        }
        // Columns clip at width 1 because 3 mod 2 == 1.
        assert_eq!(
            seen,
            vec![vec![0.0, 1.0], vec![2.0], vec![3.0, 4.0], vec![5.0]]
        );
    }

    #[test]
    fn zip_and_enumerate_over_ragged_tiling() {
        let a = array3x5();
        let tiles =
            Tiles::new(a.storage().as_slice(), a.layout(), [2, 2]).expect("valid tile shape");

        let mut checked = 0usize;
        for (index, (tile, (expected_shape, expected_values))) in
            tiles.zip(RAGGED_3X5_TILES.iter()).enumerate()
        {
            assert_eq!(tile.shape(), **expected_shape, "shape of tile {index}");
            assert_eq!(values(&tile), *expected_values, "values of tile {index}");
            checked += 1;
        }
        assert_eq!(checked, RAGGED_3X5_TILES.len());
    }

    #[test]
    fn collect_ragged_tiling_preserves_clipped_contents() {
        let a = array3x5();
        let tiles =
            Tiles::new(a.storage().as_slice(), a.layout(), [2, 2]).expect("valid tile shape");
        let collected: Vec<Vec<f64>> = tiles.map(|tile| values(&tile)).collect();

        let expected: Vec<Vec<f64>> = RAGGED_3X5_TILES
            .iter()
            .map(|(_, vals)| vals.to_vec())
            .collect();
        assert_eq!(collected, expected);

        // The clipped tiles partition the parent exactly: 4+4+2+2+2+1 == 15.
        let mut flat: Vec<f64> = collected.into_iter().flatten().collect();
        flat.sort_by(f64::total_cmp);
        assert_eq!(flat, (0..15).map(f64::from).collect::<Vec<_>>());
    }

    // ── ExactSizeIterator / DoubleEndedIterator contracts ──────────────────

    #[test]
    fn exact_size_len_matches_yielded_count() {
        let a = array3x5();
        let mut tiles =
            Tiles::new(a.storage().as_slice(), a.layout(), [2, 2]).expect("valid tile shape");

        let mut remaining = tiles.len();
        assert_eq!(remaining, 6);
        while let Some(tile) = tiles.next() {
            // A non-trivial read proves the tile is real, not a placeholder.
            assert!(!values(&tile).is_empty());
            remaining -= 1;
            assert_eq!(tiles.len(), remaining, "len must be exact after each step");
            assert_eq!(tiles.size_hint(), (remaining, Some(remaining)));
        }
        assert_eq!(remaining, 0);
        assert_eq!(tiles.len(), 0);
    }

    #[test]
    fn double_ended_reverses_the_same_sequence() {
        let a = array3x5();
        let forward: Vec<Vec<f64>> = Tiles::new(a.storage().as_slice(), a.layout(), [2, 2])
            .expect("valid tile shape")
            .map(|tile| values(&tile))
            .collect();
        let mut backward: Vec<Vec<f64>> = Tiles::new(a.storage().as_slice(), a.layout(), [2, 2])
            .expect("valid tile shape")
            .rev()
            .map(|tile| values(&tile))
            .collect();
        backward.reverse();
        assert_eq!(backward, forward);
    }

    #[test]
    fn interleaved_ends_yield_each_tile_exactly_once() {
        let a = array3x5();
        let mut tiles =
            Tiles::new(a.storage().as_slice(), a.layout(), [2, 2]).expect("valid tile shape");

        let mut head = Vec::new();
        let mut tail = Vec::new();
        while let Some(front) = tiles.next() {
            head.push(values(&front));
            if let Some(back) = tiles.next_back() {
                tail.push(values(&back));
            }
        }
        tail.reverse();
        head.extend(tail);

        let expected: Vec<Vec<f64>> = RAGGED_3X5_TILES
            .iter()
            .map(|(_, vals)| vals.to_vec())
            .collect();
        assert_eq!(head, expected);
    }
}
