//! Sweep kernels behind [`StaggeredLeapfrog3D`].
//!
//! Each kernel picks its traversal from where the differentiated axis sits in
//! row-major storage: the innermost axis is a contiguous line, the outer two
//! are strided blocks whose taps zip across whole blocks rather than cells.
//! A non-contiguous array falls back to three-index addressing, which computes
//! the same sums in the same order.

use eunomia::{FloatElement, NumericElement, RealField};
use leto::{ArrayView3, ArrayViewMut3};

use super::{Axis, StaggeredLeapfrog3D};

pub(super) fn gradient<T: RealField + FloatElement + Copy>(
    op: &StaggeredLeapfrog3D<T>,
    axis: Axis,
    field: ArrayView3<'_, T>,
    dst: &mut ArrayViewMut3<'_, T>,
    shape: [usize; 3],
) {
    let (index, extent, scale) = op.axis_geometry(axis, shape);
    let (Some(source), Some(target)) = (field.as_slice(), dst.as_mut_slice()) else {
        gradient_indexed(op, index, extent, scale, field, dst, shape);
        return;
    };
    if source.is_empty() {
        return;
    }
    // Row-major: the array is `shape[0]` planes of `shape[1] * shape[2]`
    // cells, each plane `shape[1]` rows of `shape[2]`; the chunk sizes divide
    // the length exactly, so no remainder exists to handle.
    let extent = extent as usize;
    let plane = shape[1] * shape[2];
    match index {
        0 => gradient_blocks(op, source, target, extent, plane, scale),
        1 => {
            for (source, target) in source
                .chunks_exact(plane)
                .zip(target.chunks_exact_mut(plane))
            {
                gradient_blocks(op, source, target, extent, shape[2], scale);
            }
        }
        _ => {
            for (source, target) in source
                .chunks_exact(extent)
                .zip(target.chunks_exact_mut(extent))
            {
                gradient_line(op, source, target, scale);
            }
        }
    }
}

pub(super) fn divergence<T: RealField + FloatElement + Copy>(
    op: &StaggeredLeapfrog3D<T>,
    axis: Axis,
    field: ArrayView3<'_, T>,
    dst: &mut ArrayViewMut3<'_, T>,
    shape: [usize; 3],
) {
    let (index, extent, scale) = op.axis_geometry(axis, shape);
    dst.fill(<T as NumericElement>::ZERO);

    let (Some(source), Some(target)) = (field.as_slice(), dst.as_mut_slice()) else {
        divergence_indexed(op, index, extent, scale, field, dst, shape);
        return;
    };
    if source.is_empty() {
        return;
    }
    let extent = extent as usize;
    let plane = shape[1] * shape[2];
    match index {
        0 => divergence_blocks(op, source, target, extent, plane, scale),
        1 => {
            for (source, target) in source
                .chunks_exact(plane)
                .zip(target.chunks_exact_mut(plane))
            {
                divergence_blocks(op, source, target, extent, shape[2], scale);
            }
        }
        _ => {
            for (source, target) in source
                .chunks_exact(extent)
                .zip(target.chunks_exact_mut(extent))
            {
                divergence_line(op, source, target, scale);
            }
        }
    }
}

/// One window of `2·halo` source cells split at `halo` feeds one output cell of
/// either operator: `Σ_n c_n (hi[n−1] − lo[halo−n])`. The gradient's window for
/// face `i+½` starts at `i+1−halo`; the divergence's window for cell `j` starts
/// at `j−halo` — the transpose shifts the output by one cell and changes
/// nothing else. Taps accumulate in ascending `n`, the order the indexed
/// reference uses, so the two agree bit for bit.
fn window_sum<T: RealField + FloatElement + Copy>(op: &StaggeredLeapfrog3D<T>, window: &[T]) -> T {
    let taps = op.coefficients().taps();
    let (lo, hi) = window.split_at(taps.len());
    taps.iter()
        .zip(hi)
        .zip(lo.iter().rev())
        .fold(<T as NumericElement>::ZERO, |sum, ((&c, &hi), &lo)| {
            sum + c * (hi - lo)
        })
}

/// Gradient along a non-contiguous axis: every output block of `block`
/// contiguous cells reads whole source blocks, so the taps zip across the
/// faster axes and reflection selects blocks, never cells.
fn gradient_blocks<T: RealField + FloatElement + Copy>(
    op: &StaggeredLeapfrog3D<T>,
    source: &[T],
    target: &mut [T],
    extent: usize,
    block: usize,
    scale: T,
) {
    let reach = extent as isize;
    for (here, out) in target.chunks_exact_mut(block).enumerate() {
        out.fill(<T as NumericElement>::ZERO);
        for (offset, &c) in op.coefficients().taps().iter().enumerate() {
            let n = offset as isize + 1;
            let hi = reflect(here as isize + n, reach) * block;
            let lo = reflect(here as isize - n + 1, reach) * block;
            for ((out, &hi), &lo) in out
                .iter_mut()
                .zip(&source[hi..hi + block])
                .zip(&source[lo..lo + block])
            {
                *out += c * (hi - lo);
            }
        }
        for out in out.iter_mut() {
            *out *= scale;
        }
    }
}

/// Gradient along the contiguous axis: interior cells read sliding windows, the
/// `halo − 1` leading and `halo` trailing cells reflect.
fn gradient_line<T: RealField + FloatElement + Copy>(
    op: &StaggeredLeapfrog3D<T>,
    source: &[T],
    target: &mut [T],
    scale: T,
) {
    let halo = op.halo_width();
    let extent = source.len();
    let interior = if extent >= 2 * halo {
        (halo - 1)..(extent - halo)
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
    for here in (0..extent).filter(|here| !interior.contains(here)) {
        let mut sum = <T as NumericElement>::ZERO;
        for (offset, &c) in op.coefficients().taps().iter().enumerate() {
            let n = offset as isize + 1;
            let hi = source[reflect(here as isize + n, reach)];
            let lo = source[reflect(here as isize - n + 1, reach)];
            sum += c * (hi - lo);
        }
        target[here] = sum * scale;
    }
}

/// Gradient via three-index addressing, for arrays that are not contiguous.
fn gradient_indexed<T: RealField + FloatElement + Copy>(
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
                let mut sum = <T as NumericElement>::ZERO;
                for (offset, &c) in op.coefficients().taps().iter().enumerate() {
                    let n = offset as isize + 1;
                    let mut hi = base;
                    hi[index] = reflect(here + n, extent);
                    let mut lo = base;
                    lo[index] = reflect(here - n + 1, extent);
                    sum += c * (field[hi] - field[lo]);
                }
                dst[base] = sum * scale;
            }
        }
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

/// Mirror an index about the nearest wall until it lands inside `0..extent`.
///
/// Cell centres sit at `(i+½)Δ`, so the walls fall *between* cells and the
/// mirror is `−1−m` at the low end and `2·extent−1−m` at the high end — no cell
/// is its own reflection. The loop repeats for stencils deeper than the grid,
/// which only arises for extents below the halo width; it terminates for any
/// `extent ≥ 1`.
fn reflect(mut m: isize, extent: isize) -> usize {
    debug_assert!(extent >= 1, "reflection needs a non-empty axis");
    loop {
        if m < 0 {
            m = -1 - m;
        } else if m >= extent {
            m = 2 * extent - 1 - m;
        } else {
            return usize::try_from(m).expect("invariant: m is non-negative and below extent");
        }
    }
}
