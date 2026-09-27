//! Shared strided elementwise traversal for lane operations.
//!
//! `binary_map` (two inputs, one output) and `map_into` (one input, one output)
//! walk the same logical element space with the same cache-aware strategy:
//! compute one physical offset per innermost row, then a stride-increment walk
//! along the last axis, switching to a cache-line micro-tile when some
//! operand's last-axis walk skips whole cache lines. The two wrappers carried
//! that assembly separately for their serial and parallel arms; this module
//! owns it once, generic over the operand fan-in and fan-out.
//!
//! The per-lane operation is a closure over the physical offsets of every
//! operand, so the traversal never names the element type: the caller keeps the
//! typed read/apply/write, and this module keeps the offset arithmetic, the
//! tiling decision and the row/block decomposition. `element_bytes` is the
//! per-element traffic across all operands, which sizes the parallel tasks.

use crate::application::index::{RowMajorTraversal, TileGeometry};
use leto::{Layout, Result};

/// The affine and cache-tile inputs shared by every strided lane traversal.
#[derive(Clone, Copy)]
pub(crate) struct StridedLayout<const N: usize, const IN: usize, const OUT: usize> {
    size: usize,
    shape: [usize; N],
    in_layouts: [Layout<N>; IN],
    out_layouts: [Layout<N>; OUT],
    in_tile: usize,
    out_tile: usize,
}

impl<const N: usize, const IN: usize, const OUT: usize> StridedLayout<N, IN, OUT> {
    /// Bundles the shape, the per-operand affine maps and the element tiles.
    ///
    /// `in_tile` and `out_tile` are the analytic cache-line widths in elements
    /// for the input and output element types ([`crate::application::index::line_elements_for`]).
    pub(crate) const fn new(
        size: usize,
        shape: [usize; N],
        in_layouts: [Layout<N>; IN],
        out_layouts: [Layout<N>; OUT],
        in_tile: usize,
        out_tile: usize,
    ) -> Self {
        Self {
            size,
            shape,
            in_layouts,
            out_layouts,
            in_tile,
            out_tile,
        }
    }
}

/// One physical offset per operand for a shared logical index.
#[inline]
fn offsets<const M: usize, const N: usize>(
    layouts: [Layout<N>; M],
    index: [usize; N],
) -> Result<[isize; M]> {
    let mut offsets = [0isize; M];
    for (offset, layout) in offsets.iter_mut().zip(layouts) {
        *offset = layout.offset_of(index)? as isize;
    }
    Ok(offsets)
}

/// [`offsets`] for the parallel arms, where a failed offset is a violated
/// validated-view invariant rather than caller input.
#[cfg(feature = "parallel")]
#[inline]
fn offsets_expect<const M: usize, const N: usize>(
    layouts: [Layout<N>; M],
    index: [usize; N],
) -> [isize; M] {
    let mut offsets = [0isize; M];
    for (offset, layout) in offsets.iter_mut().zip(layouts) {
        *offset = layout
            .offset_of(index)
            .expect("validated layout must map every logical index") as isize;
    }
    offsets
}

/// Advance each operand offset by its last-axis stride.
#[inline(always)]
fn advance<const M: usize>(offsets: &mut [isize; M], strides: &[isize; M]) {
    for (offset, stride) in offsets.iter_mut().zip(strides) {
        *offset += *stride;
    }
}

/// Whether some operand's last-axis walk skips whole cache lines — the
/// condition that selects the cache-line micro-tile over the plain row walk.
#[inline]
fn is_column_walk<const IN: usize, const OUT: usize>(
    in_step: &[isize; IN],
    out_step: &[isize; OUT],
    in_tile: usize,
    out_tile: usize,
) -> bool {
    in_step
        .iter()
        .any(|&stride| stride.unsigned_abs() >= in_tile)
        || out_step
            .iter()
            .any(|&stride| stride.unsigned_abs() >= out_tile)
}

/// Serial strided traversal over `IN` inputs and `OUT` outputs.
///
/// `lane` is invoked once per logical element with the current physical offset
/// of every operand.
///
/// # Errors
/// Returns the layout error of an out-of-span offset (a malformed view).
#[inline]
pub(crate) fn strided_serial<const N: usize, const IN: usize, const OUT: usize, F>(
    geometry: StridedLayout<N, IN, OUT>,
    mut lane: F,
) -> Result<()>
where
    F: FnMut([isize; IN], [isize; OUT]),
{
    let StridedLayout {
        size,
        shape,
        in_layouts,
        out_layouts,
        in_tile,
        out_tile,
    } = geometry;
    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(());
    };
    let in_step: [isize; IN] = core::array::from_fn(|i| traversal.last_axis_stride(in_layouts[i]));
    let out_step: [isize; OUT] =
        core::array::from_fn(|j| traversal.last_axis_stride(out_layouts[j]));

    if is_column_walk(&in_step, &out_step, in_tile, out_tile) {
        if let Some(tile_geometry) = TileGeometry::new(size, shape, in_tile.min(out_tile)) {
            let in_row_step: [isize; IN] = core::array::from_fn(|i| in_layouts[i].strides()[N - 2]);
            let out_row_step: [isize; OUT] =
                core::array::from_fn(|j| out_layouts[j].strides()[N - 2]);
            for slab in 0..tile_geometry.slabs() {
                let base_index = tile_geometry.slab_base_index(slab);
                let in_base = offsets(in_layouts, base_index)?;
                let out_base = offsets(out_layouts, base_index)?;
                let mut row_block = 0;
                while row_block < tile_geometry.height() {
                    let row_end = (row_block + tile_geometry.tile()).min(tile_geometry.height());
                    let mut col_block = 0;
                    while col_block < tile_geometry.width() {
                        let col_end = (col_block + tile_geometry.tile()).min(tile_geometry.width());
                        for row in row_block..row_end {
                            let r = row as isize;
                            let c0 = col_block as isize;
                            let mut in_off: [isize; IN] = core::array::from_fn(|i| {
                                in_base[i] + r * in_row_step[i] + c0 * in_step[i]
                            });
                            let mut out_off: [isize; OUT] = core::array::from_fn(|j| {
                                out_base[j] + r * out_row_step[j] + c0 * out_step[j]
                            });
                            for _ in col_block..col_end {
                                lane(in_off, out_off);
                                advance(&mut in_off, &in_step);
                                advance(&mut out_off, &out_step);
                            }
                        }
                        col_block = col_end;
                    }
                    row_block = row_end;
                }
            }
            return Ok(());
        }
    }

    for row in 0..traversal.rows() {
        let base_index = traversal.base_index(row);
        let mut in_off = offsets(in_layouts, base_index)?;
        let mut out_off = offsets(out_layouts, base_index)?;
        for _ in 0..traversal.inner() {
            lane(in_off, out_off);
            advance(&mut in_off, &in_step);
            advance(&mut out_off, &out_step);
        }
    }
    Ok(())
}

/// Parallel strided traversal over `IN` inputs and `OUT` outputs.
///
/// The offset arithmetic, tiling decision and row/block decomposition match
/// [`strided_serial`]; workers own disjoint (slab, row-block) pairs — or
/// disjoint innermost rows on the row walk — so no two workers write one
/// output element, and the aliasing-rejection guarantee of the caller's output
/// validation carries over unchanged. `element_bytes` is the traffic per
/// logical element across all operands, which sizes the unit tasks (moirai ADR
/// 0059); `lane` must be callable from several workers at once.
#[cfg(feature = "parallel")]
#[inline]
pub(crate) fn strided_parallel<const N: usize, const IN: usize, const OUT: usize, F>(
    geometry: StridedLayout<N, IN, OUT>,
    element_bytes: usize,
    lane: F,
) where
    F: Fn([isize; IN], [isize; OUT]) + Send + Sync,
{
    use crate::infrastructure::parallel::for_each_unit_range;

    let StridedLayout {
        size,
        shape,
        in_layouts,
        out_layouts,
        in_tile,
        out_tile,
    } = geometry;
    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return;
    };
    let in_step: [isize; IN] = core::array::from_fn(|i| traversal.last_axis_stride(in_layouts[i]));
    let out_step: [isize; OUT] =
        core::array::from_fn(|j| traversal.last_axis_stride(out_layouts[j]));
    let lane = &lane;

    if is_column_walk(&in_step, &out_step, in_tile, out_tile) {
        if let Some(tile_geometry) = TileGeometry::new(size, shape, in_tile.min(out_tile)) {
            let in_row_step: [isize; IN] = core::array::from_fn(|i| in_layouts[i].strides()[N - 2]);
            let out_row_step: [isize; OUT] =
                core::array::from_fn(|j| out_layouts[j].strides()[N - 2]);
            let blocks = tile_geometry.slabs() * tile_geometry.row_blocks();
            // One unit is a tile of rows across the block's width.
            let block_bytes = tile_geometry
                .tile()
                .saturating_mul(tile_geometry.width())
                .saturating_mul(element_bytes);
            for_each_unit_range(blocks, block_bytes, move |first, count| {
                for block in first..first + count {
                    let slab = block / tile_geometry.row_blocks();
                    let row_block = (block % tile_geometry.row_blocks()) * tile_geometry.tile();
                    let row_end = (row_block + tile_geometry.tile()).min(tile_geometry.height());
                    let base_index = tile_geometry.slab_base_index(slab);
                    let in_base = offsets_expect(in_layouts, base_index);
                    let out_base = offsets_expect(out_layouts, base_index);
                    let mut col_block = 0;
                    while col_block < tile_geometry.width() {
                        let col_end = (col_block + tile_geometry.tile()).min(tile_geometry.width());
                        for row in row_block..row_end {
                            let r = row as isize;
                            let c0 = col_block as isize;
                            let mut in_off: [isize; IN] = core::array::from_fn(|i| {
                                in_base[i] + r * in_row_step[i] + c0 * in_step[i]
                            });
                            let mut out_off: [isize; OUT] = core::array::from_fn(|j| {
                                out_base[j] + r * out_row_step[j] + c0 * out_step[j]
                            });
                            for _ in col_block..col_end {
                                lane(in_off, out_off);
                                advance(&mut in_off, &in_step);
                                advance(&mut out_off, &out_step);
                            }
                        }
                        col_block = col_end;
                    }
                }
            });
            return;
        }
    }

    // One unit is an innermost row.
    let row_bytes = traversal.inner().saturating_mul(element_bytes);
    for_each_unit_range(traversal.rows(), row_bytes, move |first, count| {
        for row in first..first + count {
            let base_index = traversal.base_index(row);
            let mut in_off = offsets_expect(in_layouts, base_index);
            let mut out_off = offsets_expect(out_layouts, base_index);
            for _ in 0..traversal.inner() {
                lane(in_off, out_off);
                advance(&mut in_off, &in_step);
                advance(&mut out_off, &out_step);
            }
        }
    });
}

/// Gated parallel slice kernel for the contiguous fast path.
///
/// Runs `lane` over consecutive runs of a dense `output` the runtime hands out
/// as real slices, each about one moirai unit task of work at `element_bytes`
/// per element. `lane` receives the matching input sub-slices beside each run.
#[cfg(feature = "parallel")]
#[inline]
pub(crate) fn parallel_slice_into<T, U, const IN: usize, F>(
    inputs: [&[T]; IN],
    output: &mut [U],
    element_bytes: usize,
    lane: F,
) where
    T: Send + Sync,
    U: Send,
    F: Fn([&[T]; IN], &mut [U]) + Send + Sync,
{
    crate::infrastructure::parallel::for_each_unit_run_mut(output, element_bytes, |first, run| {
        let end = first + run.len();
        lane(core::array::from_fn(|i| &inputs[i][first..end]), run);
    });
}
