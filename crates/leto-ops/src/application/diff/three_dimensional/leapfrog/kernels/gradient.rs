//! Forward staggered-gradient sweeps: contiguous lines read sliding windows,
//! the two strided axes zip their taps across whole blocks, and non-contiguous
//! arrays fall back to three-index addressing.

use super::super::{Axis, StaggeredLeapfrog3D};
use super::{reflect, window_sum};
use crate::infrastructure::parallel::for_each_plane_mut;
use eunomia::{FloatElement, NumericElement, RealField};
use leto::{ArrayView3, ArrayViewMut3};

pub(crate) fn gradient<T: RealField + FloatElement + Copy>(
    op: &StaggeredLeapfrog3D<T>,
    axis: Axis,
    field: ArrayView3<'_, T>,
    dst: &mut ArrayViewMut3<'_, T>,
    shape: [usize; 3],
) {
    let (Some(source), Some(target)) = (field.as_slice(), dst.as_mut_slice()) else {
        let (index, extent, scale) = op.axis_geometry(axis, shape);
        gradient_indexed(op, index, extent, scale, field, dst, shape);
        return;
    };
    if source.is_empty() {
        return;
    }
    let index = axis.index();
    let plane = shape[1] * shape[2];
    // Every output plane is written from shared source reads alone, so the
    // planes spread over tasks and each still runs the serial body.
    for_each_plane_mut(
        target,
        plane,
        plane_reads(op, index) * size_of::<T>(),
        |x, out| {
            gradient_plane(op, axis, source, out, x, shape);
        },
    );
}

/// Source elements a gradient plane along axis `index` reads per output
/// element: along the outer axis each face reads 2·halo whole source planes
/// and itself; along the inner two, its own plane twice.
pub(crate) fn plane_reads<T: RealField + FloatElement + Copy>(
    op: &StaggeredLeapfrog3D<T>,
    index: usize,
) -> usize {
    if index == 0 {
        1 + 2 * op.halo_width()
    } else {
        2
    }
}

/// The gradient along `axis` at x-plane `x` of a C-contiguous `source` of
/// `shape`, written into the plane `out`.
///
/// Row-major: the array is `shape[0]` planes of `shape[1] * shape[2]` cells,
/// each plane `shape[1]` rows of `shape[2]`; the chunk sizes divide the length
/// exactly, so no remainder exists to handle.
fn gradient_plane<T: RealField + FloatElement + Copy>(
    op: &StaggeredLeapfrog3D<T>,
    axis: Axis,
    source: &[T],
    out: &mut [T],
    x: usize,
    shape: [usize; 3],
) {
    let (index, extent, scale) = op.axis_geometry(axis, shape);
    let extent = extent as usize;
    let plane = shape[1] * shape[2];
    let own = &source[x * plane..(x + 1) * plane];
    match index {
        0 => gradient_block(op, source, out, x, extent, scale),
        1 => gradient_blocks(op, own, out, extent, shape[2], scale),
        _ => {
            for (source, target) in own.chunks_exact(extent).zip(out.chunks_exact_mut(extent)) {
                gradient_line(op, source, target, scale);
            }
        }
    }
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
    for (here, out) in target.chunks_exact_mut(block).enumerate() {
        gradient_block(op, source, out, here, extent, scale);
    }
}

/// One output block of [`gradient_blocks`]: face `here` of an axis of `extent`
/// blocks of `out.len()` cells, read from whole source blocks.
fn gradient_block<T: RealField + FloatElement + Copy>(
    op: &StaggeredLeapfrog3D<T>,
    source: &[T],
    out: &mut [T],
    here: usize,
    extent: usize,
    scale: T,
) {
    let block = out.len();
    gradient_taps(op, out, here, extent, scale, |at| {
        &source[at * block..(at + 1) * block]
    });
}

/// Face `here` of an axis of `extent` lanes, each as long as `out`, where
/// `lane(i)` is lane `i` of the source: the taps summed across whole lanes
/// in coefficient order, then scaled.
fn gradient_taps<'a, T: RealField + FloatElement + Copy + 'a>(
    op: &StaggeredLeapfrog3D<T>,
    out: &mut [T],
    here: usize,
    extent: usize,
    scale: T,
    lane: impl Fn(usize) -> &'a [T],
) {
    let reach = extent as isize;
    out.fill(<T as NumericElement>::ZERO);
    for (offset, &c) in op.coefficients().taps().iter().enumerate() {
        let n = offset as isize + 1;
        let hi = lane(reflect(here as isize + n, reach));
        let lo = lane(reflect(here as isize - n + 1, reach));
        for ((out, &hi), &lo) in out.iter_mut().zip(hi).zip(lo) {
            *out += c * (hi - lo);
        }
    }
    for out in out.iter_mut() {
        *out *= scale;
    }
}

/// The gradient along `axis` at row `y` of x-plane `x` of a C-contiguous
/// `source` of `shape`, written into the row `out`: the value
/// [`gradient_plane`] writes there, by the same sums in the same order.
pub(crate) fn gradient_row<T: RealField + FloatElement + Copy>(
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
        0 => gradient_taps(op, out, x, extent, scale, |plane| row_at(plane, y)),
        1 => gradient_taps(op, out, y, extent, scale, |row| row_at(x, row)),
        _ => gradient_line(op, row_at(x, y), out, scale),
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
