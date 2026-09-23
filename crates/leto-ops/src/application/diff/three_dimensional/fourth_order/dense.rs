//! Dense (C-contiguous) traversal of the fourth-order stencils: field data is
//! read as flat slices and swept lane-by-lane, task-parallel across planes.

use core::ops::Range;

use eunomia::{FloatElement, NumericElement, RealField};

use super::super::leapfrog::Axis;
use super::stencil::{Scales, Stencil, ELEMENTS_PER_UNIT};
use crate::infrastructure::parallel::{for_each_plane_mut, for_each_plane_mut_many_with};

pub(super) fn sweep_dense<T>(
    source: &[T],
    out: &mut [T],
    shape: [usize; 3],
    axis: Axis,
    scales: Scales<T>,
) where
    T: RealField + FloatElement + Copy,
{
    let [nx, ny, nz] = shape;
    let plane_len = ny * nz;
    let element_bytes = ELEMENTS_PER_UNIT * size_of::<T>();
    for_each_plane_mut(out, plane_len, element_bytes, |x, out_plane| {
        let plane_at = |i: usize| &source[i * plane_len..(i + 1) * plane_len];
        match axis {
            Axis::X => Stencil::at(x, nx)
                .apply_lane(scales, out_plane, |o| plane_at(x.wrapping_add_signed(o))),
            Axis::Y => {
                let plane = plane_at(x);
                for (y, out_row) in out_plane.chunks_exact_mut(nz).enumerate() {
                    Stencil::at(y, ny).apply_lane(scales, out_row, |o| {
                        let row = y.wrapping_add_signed(o);
                        &plane[row * nz..(row + 1) * nz]
                    });
                }
            }
            Axis::Z => {
                let plane = plane_at(x);
                for (out_row, row) in out_plane.chunks_exact_mut(nz).zip(plane.chunks_exact(nz)) {
                    sweep_row(scales, out_row, row);
                }
            }
        }
    });
}

/// `dst = ∂fields[0]/∂x + ∂fields[1]/∂y + ∂fields[2]/∂z`, summed in that order.
///
/// Each field is read once per output lane, so the sum never round-trips
/// through per-axis buffers: at 64 cubed that is 8 MB of traffic where the
/// composed form moves 20 MB. The per-coordinate stencils and the arithmetic
/// are [`super::dispatch::central4_into`]'s, so the result is bit-identical
/// to sweeping each axis into its own buffer and adding the three in this
/// order.
pub(super) fn divergence_dense<T>(
    fields: [&[T]; 3],
    out: &mut [T],
    shape: [usize; 3],
    scales: [Scales<T>; 3],
) where
    T: RealField + FloatElement + Copy,
{
    let [nx, ny, nz] = shape;
    let plane_len = ny * nz;
    // One output element, and for each of the three fields the four
    // neighbours its axis stencil reaches.
    let element_bytes = (1 + 3 * (ELEMENTS_PER_UNIT - 1)) * size_of::<T>();
    for_each_plane_mut(out, plane_len, element_bytes, |x, out_plane| {
        let x_stencil = Stencil::at(x, nx);
        for (y, out_row) in out_plane.chunks_exact_mut(nz).enumerate() {
            x_stencil.apply_lane(scales[0], out_row, |o| {
                let base = x.wrapping_add_signed(o) * plane_len + y * nz;
                &fields[0][base..base + nz]
            });
            Stencil::at(y, ny).add_lane(scales[1], out_row, |o| {
                let base = x * plane_len + y.wrapping_add_signed(o) * nz;
                &fields[1][base..base + nz]
            });
            let base = x * plane_len + y * nz;
            add_row(scales[2], out_row, &fields[2][base..base + nz]);
        }
    });
}

/// One axis derivative a fused dense pass computes: the field it reads, the
/// grid plane that field's storage starts at, the axis it differentiates
/// along, and that axis's stencil scales.
#[derive(Clone, Copy)]
pub(super) struct DenseTerm<'a, T> {
    pub(super) axis: Axis,
    pub(super) field: &'a [T],
    pub(super) origin: usize,
    pub(super) scales: Scales<T>,
}

/// One fused pass writing `K` destinations: each output lane receives the `N`
/// axis derivatives named by `terms` and the `M` pointwise values at that
/// lane and the `K` values the destinations hold there, and `combine`
/// returns the `K` values written in their place.
///
/// Fusing is where the traffic goes. A caller that scales a sum of two axis
/// derivatives by a field -- an elastic shear stress, `K = 1` -- reads its
/// fields once per lane here instead of sweeping each axis into a buffer and
/// reading both back beside the scale: 8 MB at 64 cubed where the composed
/// form moves 16 MB. Where several results read the same derivatives -- the
/// three diagonal stresses read the same three normal strains, `K = 3` --
/// one pass sweeps the stencils once and writes all of them, where a call
/// per destination would sweep them again. And any `K` runs as one parallel
/// region: moirai splits all `K` outputs in lockstep (its ADR 0059
/// K-buffer form), so the destination count never becomes a region count.
///
/// The per-coordinate stencils and the arithmetic are
/// [`super::dispatch::central4_into`]'s, so each derivative handed to
/// `combine` is the value that sweep would have written.
///
/// Terms name distinct axes in the cases this serves. A repeated axis is not
/// rejected; it reads its field twice and hands `combine` both values, which
/// is what it asked for.
pub(super) fn map_dense<T, const N: usize, const M: usize, const K: usize, F>(
    terms: [DenseTerm<'_, T>; N],
    pointwise: [(&[T], usize); M],
    (out, out_origin): ([&mut [T]; K], usize),
    shape: [usize; 3],
    planes: Range<usize>,
    combine: &F,
) where
    T: RealField + FloatElement + Copy,
    F: Fn([T; N], [T; M], [T; K]) -> [T; K] + Send + Sync,
{
    let [nx, ny, nz] = shape;
    let plane_len = ny * nz;
    let first_plane = planes.start;
    let written = (planes.start - out_origin) * plane_len..(planes.end - out_origin) * plane_len;
    // `K` output elements, the four neighbours each term's stencil reaches,
    // and one element of each pointwise input.
    let element_bytes = (K + N * (ELEMENTS_PER_UNIT - 1) + M) * size_of::<T>();
    for_each_plane_mut_many_with(
        out.map(|destination| &mut destination[written.clone()]),
        plane_len,
        element_bytes,
        || vec![<T as NumericElement>::ZERO; N * nz],
        |scratch, offset, out_planes| {
            let x = first_plane + offset;
            let mut out_rows = out_planes.map(|plane| plane.chunks_exact_mut(nz));
            for y in 0..ny {
                // Each term writes a whole lane through the same kernel a
                // separate sweep would use, into a row of scratch that stays
                // in L1: the fields are still read once per output lane, and
                // the lane writers keep vectorizing. Dispatching a stencil
                // per element instead costs more than the traffic it saves --
                // kwavers' shear assembly measured 134 -> 170 us at 64 cubed
                // on the per-element form.
                for (term, values) in terms.iter().zip(scratch.chunks_exact_mut(nz)) {
                    let DenseTerm {
                        axis,
                        field,
                        origin,
                        scales,
                    } = *term;
                    // `origin` is the grid plane the field's storage starts
                    // at; the stencil is chosen by the grid plane `x`.
                    match axis {
                        Axis::X => Stencil::at(x, nx).apply_lane(scales, values, |o| {
                            let plane = (x.wrapping_add_signed(o) - origin) * plane_len + y * nz;
                            &field[plane..plane + nz]
                        }),
                        Axis::Y => Stencil::at(y, ny).apply_lane(scales, values, |o| {
                            let lane = (x - origin) * plane_len + y.wrapping_add_signed(o) * nz;
                            &field[lane..lane + nz]
                        }),
                        Axis::Z => {
                            let lane = (x - origin) * plane_len + y * nz;
                            sweep_row(scales, values, &field[lane..lane + nz]);
                        }
                    }
                }
                // Every input lane and destination row is sliced to exactly
                // `nz`, so each access below is provably in bounds.
                let lanes: [&[T]; N] = core::array::from_fn(|j| &scratch[j * nz..(j + 1) * nz]);
                let points: [&[T]; M] = core::array::from_fn(|p| {
                    let start = (x - pointwise[p].1) * plane_len + y * nz;
                    &pointwise[p].0[start..start + nz]
                });
                let mut rows: [&mut [T]; K] = core::array::from_fn(|d| {
                    let row = out_rows[d]
                        .next()
                        .expect("invariant: every destination plane holds ny rows");
                    &mut row[..nz]
                });
                for k in 0..nz {
                    let values = combine(
                        core::array::from_fn(|j| lanes[j][k]),
                        core::array::from_fn(|p| points[p][k]),
                        core::array::from_fn(|d| rows[d][k]),
                    );
                    for (row, value) in rows.iter_mut().zip(values) {
                        row[k] = value;
                    }
                }
            }
        },
    );
}

/// The z-axis derivative of `row`, added to what `out` already holds.
pub(super) fn add_row<T: RealField + FloatElement + Copy>(
    scales: Scales<T>,
    out: &mut [T],
    row: &[T],
) {
    let n = row.len();
    if n >= 5 {
        let interior = row[..n - 4]
            .iter()
            .zip(&row[1..n - 3])
            .zip(&row[3..n - 1])
            .zip(&row[4..]);
        for (value, (((&m2, &m1), &p1), &p2)) in out[2..n - 2].iter_mut().zip(interior) {
            *value += scales.fourth(m2, m1, p1, p2);
        }
    }
    // Accumulation must visit each coordinate once: on a short axis the four
    // edge positions collapse onto each other (n = 2 gives 0, 1, 0, 1), and
    // adding that coordinate's derivative twice is a wrong value, where the
    // assigning twin merely writes it twice. From n = 4 up the four are
    // already distinct and ordered, so the dedupe is confined to the short
    // case rather than paid on every row of every sweep; below it every
    // coordinate is an edge, and 0..n is exactly the collapsed set.
    if n >= 4 {
        for c in [0, 1, n - 2, n - 1] {
            out[c] += Stencil::at(c, n).apply(scales, |o| row[c.wrapping_add_signed(o)]);
        }
    } else {
        for c in 0..n {
            out[c] += Stencil::at(c, n).apply(scales, |o| row[c.wrapping_add_signed(o)]);
        }
    }
}

/// The derivative along one lane of the differentiated axis.
pub(super) fn sweep_row<T: RealField + FloatElement + Copy>(
    scales: Scales<T>,
    out: &mut [T],
    row: &[T],
) {
    let n = row.len();
    if n >= 5 {
        let interior = row[..n - 4]
            .iter()
            .zip(&row[1..n - 3])
            .zip(&row[3..n - 1])
            .zip(&row[4..]);
        for (value, (((&m2, &m1), &p1), &p2)) in out[2..n - 2].iter_mut().zip(interior) {
            *value = scales.fourth(m2, m1, p1, p2);
        }
    }
    let edges = [0, 1, n.saturating_sub(2), n.saturating_sub(1)];
    for c in edges.into_iter().filter(|&c| c < n) {
        out[c] = Stencil::at(c, n).apply(scales, |o| row[c.wrapping_add_signed(o)]);
    }
}
