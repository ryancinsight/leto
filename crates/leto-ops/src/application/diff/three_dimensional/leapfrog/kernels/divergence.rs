//! Transposed staggered-divergence sweeps — the transpose of the gradient
//! family: contiguous interiors share the gradient's window sums shifted one
//! cell, strided axes gather the scattered terms in the scatter's order, and
//! non-contiguous arrays fall back to three-index addressing.

use super::super::{Axis, StaggeredLeapfrog3D};
use super::{reflect, window_sum};
use crate::infrastructure::parallel::for_each_plane_mut;
use eunomia::{FloatElement, NumericElement, RealField};
use leto::{ArrayView3, ArrayViewMut3};

pub(crate) fn divergence<T: RealField + FloatElement + Copy>(
    op: &StaggeredLeapfrog3D<T>,
    axis: Axis,
    field: ArrayView3<'_, T>,
    dst: &mut ArrayViewMut3<'_, T>,
    shape: [usize; 3],
) {
    let (index, extent, scale) = op.axis_geometry(axis, shape);
    // The transposes accumulate into a zeroed target. The indexed fallback
    // zeroes the whole target on the calling thread; every contiguous arm
    // zeroes each plane inside its own task, so the zeroing spreads with the
    // sweep instead of staying a serial pass over the volume.
    let (Some(source), Some(target)) = (field.as_slice(), dst.as_mut_slice()) else {
        dst.fill(<T as NumericElement>::ZERO);
        divergence_indexed(op, index, extent, scale, field, dst, shape);
        return;
    };
    if source.is_empty() {
        return;
    }
    let extent = extent as usize;
    let plane = shape[1] * shape[2];
    let element_bytes = size_of::<T>();
    match index {
        // A source plane scatters into reflected target planes, so each target
        // plane gathers its terms instead, in the scatter's order.
        0 => {
            // A target plane reads up to 2·halo source planes.
            let reads = 1 + 2 * op.halo_width();
            for_each_plane_mut(target, plane, reads * element_bytes, |here, out| {
                divergence_block_gather(op, source, out, here, extent, scale);
            });
        }
        1 => for_each_plane_mut(target, plane, 2 * element_bytes, |x, out| {
            out.fill(<T as NumericElement>::ZERO);
            let start = x * plane;
            divergence_blocks(
                op,
                &source[start..start + plane],
                out,
                extent,
                shape[2],
                scale,
            );
        }),
        _ => for_each_plane_mut(target, plane, 2 * element_bytes, |x, out| {
            out.fill(<T as NumericElement>::ZERO);
            let start = x * plane;
            for (source, target) in source[start..start + plane]
                .chunks_exact(extent)
                .zip(out.chunks_exact_mut(extent))
            {
                divergence_line(op, source, target, scale);
            }
        }),
    }
}

/// Transpose along a non-contiguous axis: each source block scatters into the
/// two reflected target blocks per tap, in the reference's order, so every cell
/// accumulates the same terms in the same sequence.
fn divergence_blocks<T: RealField + FloatElement + Copy>(
    op: &StaggeredLeapfrog3D<T>,
    source: &[T],
    target: &mut [T],
    extent: usize,
    block: usize,
    scale: T,
) {
    let reach = extent as isize;
    for (here, value) in source.chunks_exact(block).enumerate() {
        for (offset, &c) in op.coefficients().taps().iter().enumerate() {
            let n = offset as isize + 1;
            let hi = reflect(here as isize + n, reach) * block;
            for (out, &value) in target[hi..hi + block].iter_mut().zip(value) {
                *out -= c * (value * scale);
            }
            let lo = reflect(here as isize - n + 1, reach) * block;
            for (out, &value) in target[lo..lo + block].iter_mut().zip(value) {
                *out += c * (value * scale);
            }
        }
    }
}

/// One target block of [`divergence_blocks`] over a whole axis, gathered: block
/// `here` of an axis of `extent` blocks of `out.len()` cells.
///
/// The scatter visits sources in ascending order, taps in ascending order, and
/// for each writes its high reflection before its low one. This gather walks the
/// same sequence and applies only the writes that land on `here`, so every cell
/// accumulates the same terms in the same order and matches the scatter to the
/// bit; target blocks no longer share writes and can run on separate tasks.
fn divergence_block_gather<T: RealField + FloatElement + Copy>(
    op: &StaggeredLeapfrog3D<T>,
    source: &[T],
    out: &mut [T],
    here: usize,
    extent: usize,
    scale: T,
) {
    let block = out.len();
    divergence_gather(op, out, here, extent, scale, |at| {
        &source[at * block..(at + 1) * block]
    });
}

/// Lane `here` of an axis of `extent` lanes, each as long as `out`, gathered
/// in the scatter's order from `lane(i)`, lane `i` of the source.
fn divergence_gather<'a, T: RealField + FloatElement + Copy + 'a>(
    op: &StaggeredLeapfrog3D<T>,
    out: &mut [T],
    here: usize,
    extent: usize,
    scale: T,
    lane: impl Fn(usize) -> &'a [T],
) {
    let reach = extent as isize;
    out.fill(<T as NumericElement>::ZERO);
    // A lane `2·halo` or more from both walls is reached only by unreflected
    // taps, from sources within `halo` of it; nearer a wall a reflected tap
    // can arrive from anywhere the reflection maps. Either way the sources
    // are walked in ascending order, so the terms keep the scatter's order.
    let halo = op.halo_width();
    let sources = if here >= 2 * halo && here + 2 * halo < extent {
        here - halo..here + halo + 1
    } else {
        0..extent
    };
    for from in sources {
        for (offset, &c) in op.coefficients().taps().iter().enumerate() {
            let n = offset as isize + 1;
            if reflect(from as isize + n, reach) == here {
                for (out, &value) in out.iter_mut().zip(lane(from)) {
                    *out -= c * (value * scale);
                }
            }
            if reflect(from as isize - n + 1, reach) == here {
                for (out, &value) in out.iter_mut().zip(lane(from)) {
                    *out += c * (value * scale);
                }
            }
        }
    }
}

/// The divergence along `axis` at row `y` of x-plane `x` of a C-contiguous
/// `source` of `shape`, written into the row `out`: the value [`divergence`]
/// writes there, by the same terms in the same order.
pub(crate) fn divergence_row<T: RealField + FloatElement + Copy>(
    op: &StaggeredLeapfrog3D<T>,
    axis: Axis,
    source: &[T],
    out: &mut [T],
    [x, y]: [usize; 2],
    shape: [usize; 3],
) {
    let (index, extent, scale) = op.axis_geometry(axis, shape);
    let extent = extent as usize;
    let [_, ny, nz] = shape;
    let row_at = |plane: usize, row: usize| {
        let start = (plane * ny + row) * nz;
        &source[start..start + nz]
    };
    match index {
        0 => divergence_gather(op, out, x, extent, scale, |plane| row_at(plane, y)),
        1 => divergence_gather(op, out, y, extent, scale, |row| row_at(x, row)),
        _ => {
            out.fill(<T as NumericElement>::ZERO);
            divergence_line(op, row_at(x, y), out, scale);
        }
    }
}

/// Transpose along the contiguous axis. A cell `halo` or more from either wall
/// receives only unreflected taps (a reflected tap `reflect(k+n)` or
/// `reflect(k+1−n)` lands within `halo − 1` of the wall it crossed), so the
/// interior is the transpose in gather form — the same window sum as the
/// gradient, shifted by one cell. Wall cells keep the scatter: only sources
/// within `2·halo` of a wall reach them, and the interior guard stops those
/// sources from double-counting into gathered cells.
fn divergence_line<T: RealField + FloatElement + Copy>(
    op: &StaggeredLeapfrog3D<T>,
    source: &[T],
    target: &mut [T],
    scale: T,
) {
    let halo = op.halo_width();
    let extent = source.len();
    let interior = if extent >= 2 * halo {
        halo..(extent - halo)
    } else {
        0..0
    };
    for (out, window) in target[interior.clone()]
        .iter_mut()
        .zip(source.windows(2 * halo))
    {
        *out = window_sum(op, window) * scale;
    }
    let reach = extent as isize;
    let near_a_wall = |here: &usize| *here < 2 * halo || *here + 2 * halo >= extent;
    for here in (0..extent).filter(near_a_wall) {
        let value = source[here] * scale;
        for (offset, &c) in op.coefficients().taps().iter().enumerate() {
            let n = offset as isize + 1;
            let hi = reflect(here as isize + n, reach);
            if !interior.contains(&hi) {
                target[hi] -= c * value;
            }
            let lo = reflect(here as isize - n + 1, reach);
            if !interior.contains(&lo) {
                target[lo] += c * value;
            }
        }
    }
}

/// Divergence via three-index addressing, for arrays that are not contiguous.
fn divergence_indexed<T: RealField + FloatElement + Copy>(
    op: &StaggeredLeapfrog3D<T>,
    index: usize,
    extent: isize,
    scale: T,
    field: ArrayView3<'_, T>,
    dst: &mut ArrayViewMut3<'_, T>,
    shape: [usize; 3],
) {
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let base = [i, j, k];
                let here = base[index] as isize;
                let value = field[base] * scale;
                for (offset, &c) in op.coefficients().taps().iter().enumerate() {
                    let n = offset as isize + 1;
                    let mut hi = base;
                    hi[index] = reflect(here + n, extent);
                    let mut lo = base;
                    lo[index] = reflect(here - n + 1, extent);
                    dst[hi] -= c * value;
                    dst[lo] += c * value;
                }
            }
        }
    }
}
