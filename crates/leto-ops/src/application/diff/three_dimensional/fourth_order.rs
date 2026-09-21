//! Fourth-order central first derivative behind
//! [`FiniteDifference3DScheme::CentralFourthOrder`](super::FiniteDifference3DScheme).
//!
//! A coordinate `c` on an axis of `n` points takes one stencil:
//!
//! | Position | Stencil | Order |
//! |---|---|---|
//! | `2 ≤ c ≤ n − 3` | `(−f[c+2] + 8f[c+1] − 8f[c−1] + f[c−2]) / 12Δ` | 4 |
//! | `c ∈ {1, n − 2}` | `(f[c+1] − f[c−1]) / 2Δ` | 2 |
//! | `c = 0` | `(f[1] − f[0]) / Δ` | 1 |
//! | `c = n − 1` | `(f[n−1] − f[n−2]) / Δ` | 1 |
//! | `n = 1` | `0` | — |
//!
//! The first matching row wins, so an axis shorter than five points takes only
//! the lower-order closures (a two-point axis is one-sided at both ends), and a
//! singleton axis has no variation along it. Each stencil is written as one
//! arithmetic chain shared by both traversals, so they agree bit for bit.
//!
//! C-contiguous fields sweep whole lanes, a task per run of x-planes sized by
//! the bytes a plane moves; any other layout walks the logical indices on the
//! calling thread.

use eunomia::{FloatElement, NumericElement, RealField};
use leto::{ArrayView3, ArrayViewMut3, Result};

use super::f;
use super::leapfrog::Axis;
use crate::infrastructure::parallel::{
    for_each_plane_mut, for_each_plane_mut_triple_with, for_each_plane_mut_with,
};

/// Bytes one output element moves: itself and the four neighbours the
/// interior stencil reads.
const ELEMENTS_PER_UNIT: usize = 5;

/// Scale factors of the three stencil orders.
#[derive(Clone, Copy)]
struct Scales<T> {
    inv_12h: T,
    inv_2h: T,
    inv_h: T,
    eight: T,
}

impl<T: RealField + FloatElement + Copy> Scales<T> {
    fn new(h: T) -> Self {
        Self {
            inv_12h: <T as NumericElement>::ONE / (f::<T>(12.0) * h),
            inv_2h: <T as NumericElement>::ONE / (f::<T>(2.0) * h),
            inv_h: <T as NumericElement>::ONE / h,
            eight: f::<T>(8.0),
        }
    }

    #[inline]
    fn fourth(self, m2: T, m1: T, p1: T, p2: T) -> T {
        ((-self.eight * m1) + (self.eight * p1) + (-p2) + m2) * self.inv_12h
    }

    #[inline]
    fn second(self, m1: T, p1: T) -> T {
        (p1 - m1) * self.inv_2h
    }

    #[inline]
    fn first(self, lower: T, upper: T) -> T {
        (upper - lower) * self.inv_h
    }
}

/// The stencil coordinate `c` takes on an axis of `n` points.
#[derive(Clone, Copy)]
enum Stencil {
    Flat,
    Forward,
    Backward,
    Second,
    Fourth,
}

impl Stencil {
    #[inline]
    fn at(c: usize, n: usize) -> Self {
        if n == 1 {
            Self::Flat
        } else if c == 0 {
            Self::Forward
        } else if c == n - 1 {
            Self::Backward
        } else if c < 2 || c >= n - 2 {
            Self::Second
        } else {
            Self::Fourth
        }
    }

    /// The derivative at a point whose neighbour at signed offset `o` along
    /// the axis is `at(o)`; only the offsets the stencil reads are asked for.
    #[inline]
    fn apply<T: RealField + FloatElement + Copy>(
        self,
        scales: Scales<T>,
        at: impl Fn(isize) -> T,
    ) -> T {
        match self {
            Self::Flat => <T as NumericElement>::ZERO,
            Self::Forward => scales.first(at(0), at(1)),
            Self::Backward => scales.first(at(-1), at(0)),
            Self::Second => scales.second(at(-1), at(1)),
            Self::Fourth => scales.fourth(at(-2), at(-1), at(1), at(2)),
        }
    }

    /// The derivative along a whole lane, added to what `out` already holds.
    #[inline]
    fn add_lane<'a, T: RealField + FloatElement + Copy + 'a>(
        self,
        scales: Scales<T>,
        out: &mut [T],
        lane: impl Fn(isize) -> &'a [T],
    ) {
        match self {
            Self::Flat => (),
            Self::Forward => {
                for (value, (&lower, &upper)) in out.iter_mut().zip(lane(0).iter().zip(lane(1))) {
                    *value += scales.first(lower, upper);
                }
            }
            Self::Backward => {
                for (value, (&lower, &upper)) in out.iter_mut().zip(lane(-1).iter().zip(lane(0))) {
                    *value += scales.first(lower, upper);
                }
            }
            Self::Second => {
                for (value, (&m1, &p1)) in out.iter_mut().zip(lane(-1).iter().zip(lane(1))) {
                    *value += scales.second(m1, p1);
                }
            }
            Self::Fourth => {
                let neighbours = lane(-2).iter().zip(lane(-1)).zip(lane(1)).zip(lane(2));
                for (value, (((&m2, &m1), &p1), &p2)) in out.iter_mut().zip(neighbours) {
                    *value += scales.fourth(m2, m1, p1, p2);
                }
            }
        }
    }

    /// The derivative along a whole lane whose neighbour lane at signed
    /// offset `o` is `lane(o)`, each as long as `out`.
    #[inline]
    fn apply_lane<'a, T: RealField + FloatElement + Copy + 'a>(
        self,
        scales: Scales<T>,
        out: &mut [T],
        lane: impl Fn(isize) -> &'a [T],
    ) {
        match self {
            Self::Flat => out.fill(<T as NumericElement>::ZERO),
            Self::Forward => {
                for (value, (&lower, &upper)) in out.iter_mut().zip(lane(0).iter().zip(lane(1))) {
                    *value = scales.first(lower, upper);
                }
            }
            Self::Backward => {
                for (value, (&lower, &upper)) in out.iter_mut().zip(lane(-1).iter().zip(lane(0))) {
                    *value = scales.first(lower, upper);
                }
            }
            Self::Second => {
                for (value, (&m1, &p1)) in out.iter_mut().zip(lane(-1).iter().zip(lane(1))) {
                    *value = scales.second(m1, p1);
                }
            }
            Self::Fourth => {
                let neighbours = lane(-2).iter().zip(lane(-1)).zip(lane(1)).zip(lane(2));
                for (value, (((&m2, &m1), &p1), &p2)) in out.iter_mut().zip(neighbours) {
                    *value = scales.fourth(m2, m1, p1, p2);
                }
            }
        }
    }
}

/// `dst = ∂field/∂axis` with the closure in the module documentation.
///
/// The caller has checked that `dst` has `field`'s shape.
pub(super) fn central4_into<T>(
    field: ArrayView3<T>,
    dst: &mut ArrayViewMut3<'_, T>,
    axis: Axis,
    h: T,
) -> Result<()>
where
    T: RealField + FloatElement + Copy,
{
    let shape = field.shape();
    let scales = Scales::new(h);
    if let (Some(source), Some(out)) = (field.as_slice(), dst.as_mut_slice()) {
        sweep_dense(source, out, shape, axis, scales);
    } else {
        sweep_strided(field, dst, shape, axis, scales);
    }
    Ok(())
}

fn sweep_dense<T>(source: &[T], out: &mut [T], shape: [usize; 3], axis: Axis, scales: Scales<T>)
where
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
/// are [`central4_into`]'s, so the result is bit-identical to sweeping each
/// axis into its own buffer and adding the three in this order.
///
/// The caller has checked that every field and `dst` share one shape.
pub(super) fn central4_divergence_into<T>(
    fields: [ArrayView3<'_, T>; 3],
    dst: &mut ArrayViewMut3<'_, T>,
    spacing: [T; 3],
) -> Result<()>
where
    T: RealField + FloatElement + Copy,
{
    let shape = fields[0].shape();
    let scales = spacing.map(Scales::new);
    let slices = (
        fields[0].as_slice(),
        fields[1].as_slice(),
        fields[2].as_slice(),
        dst.as_mut_slice(),
    );
    if let (Some(fx), Some(fy), Some(fz), Some(out)) = slices {
        divergence_dense([fx, fy, fz], out, shape, scales);
    } else {
        divergence_strided(fields, dst, shape, scales);
    }
    Ok(())
}

fn divergence_dense<T>(fields: [&[T]; 3], out: &mut [T], shape: [usize; 3], scales: [Scales<T>; 3])
where
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

/// One fused pass over `dst`: each output lane receives the `N` axis
/// derivatives named by `terms` and the `M` pointwise values at that lane,
/// and `combine` decides what to write.
///
/// A caller that scales a sum of two axis derivatives by a field -- an
/// elastic shear stress is one -- reads its fields once per lane here,
/// instead of sweeping each axis into a buffer and reading both back
/// alongside the scale: 8 MB of traffic at 64 cubed where the composed form
/// moves 16 MB. The per-coordinate stencils and the arithmetic are
/// [`central4_into`]'s, so each derivative handed to `combine` is the value
/// that sweep would have written.
///
/// Terms name distinct axes. A repeated axis is not rejected; it reads its
/// field twice and hands `combine` both values, which is what it asked for.
///
/// The caller has checked that every field, every pointwise input and `dst`
/// share one shape.
pub(super) fn central4_map_into<T, const N: usize, const M: usize, F>(
    terms: [(Axis, ArrayView3<'_, T>); N],
    pointwise: [ArrayView3<'_, T>; M],
    dst: &mut ArrayViewMut3<'_, T>,
    spacing: [T; 3],
    combine: F,
) -> Result<()>
where
    T: RealField + FloatElement + Copy,
    F: Fn([T; N], [T; M]) -> T + Send + Sync,
{
    let shape = dst.shape();
    let scales: [Scales<T>; N] = core::array::from_fn(|j| Scales::new(spacing[terms[j].0.index()]));
    let axes: [Axis; N] = core::array::from_fn(|j| terms[j].0);
    let mut dense = true;
    let fields: [&[T]; N] = core::array::from_fn(|j| {
        terms[j].1.as_slice().unwrap_or_else(|| {
            dense = false;
            &[]
        })
    });
    let scalars: [&[T]; M] = core::array::from_fn(|k| {
        pointwise[k].as_slice().unwrap_or_else(|| {
            dense = false;
            &[]
        })
    });
    if dense {
        if let Some(out) = dst.as_mut_slice() {
            map_dense(fields, scalars, axes, out, shape, scales, &combine);
            return Ok(());
        }
    }
    map_strided(terms, pointwise, dst, shape, scales, &combine);
    Ok(())
}

fn map_dense<T, const N: usize, const M: usize, F>(
    fields: [&[T]; N],
    pointwise: [&[T]; M],
    axes: [Axis; N],
    out: &mut [T],
    shape: [usize; 3],
    scales: [Scales<T>; N],
    combine: &F,
) where
    T: RealField + FloatElement + Copy,
    F: Fn([T; N], [T; M]) -> T + Send + Sync,
{
    let [nx, ny, nz] = shape;
    let plane_len = ny * nz;
    // One output element, the four neighbours each term's stencil reaches,
    // and one element of each pointwise input.
    let element_bytes = (1 + N * (ELEMENTS_PER_UNIT - 1) + M) * size_of::<T>();
    for_each_plane_mut_with(
        out,
        plane_len,
        element_bytes,
        || vec![<T as NumericElement>::ZERO; N * nz],
        |scratch, x, out_plane| {
            for (y, out_row) in out_plane.chunks_exact_mut(nz).enumerate() {
                let base = x * plane_len + y * nz;
                // Each term writes a whole lane through the same kernel a
                // separate sweep would use, into a row of scratch that stays
                // in L1: the fields are still read once per output lane, and
                // the lane writers keep vectorizing. Dispatching a stencil
                // per element instead costs more than the traffic it saves --
                // kwavers' shear assembly measured 134 -> 170 us at 64 cubed
                // on the per-element form.
                for (j, values) in scratch.chunks_exact_mut(nz).enumerate() {
                    match axes[j] {
                        Axis::X => Stencil::at(x, nx).apply_lane(scales[j], values, |o| {
                            let plane = x.wrapping_add_signed(o) * plane_len + y * nz;
                            &fields[j][plane..plane + nz]
                        }),
                        Axis::Y => Stencil::at(y, ny).apply_lane(scales[j], values, |o| {
                            let row = x * plane_len + y.wrapping_add_signed(o) * nz;
                            &fields[j][row..row + nz]
                        }),
                        Axis::Z => sweep_row(scales[j], values, &fields[j][base..base + nz]),
                    }
                }
                for (k, value) in out_row.iter_mut().enumerate() {
                    *value = combine(
                        core::array::from_fn(|j| scratch[j * nz + k]),
                        core::array::from_fn(|p| pointwise[p][base + k]),
                    );
                }
            }
        },
    );
}

/// [`central4_map_into`] writing three destinations from one derivative pass.
///
/// The elastic diagonal stresses are the case: all three read the same three
/// normal strains, so computing the strains once and writing all three
/// stresses in that pass moves 16 MB at 64 cubed where sweeping the strains
/// into buffers and combining them afterwards moves 28 MB. Splitting the
/// combination into three single-destination calls would instead sweep the
/// stencils three times over.
///
/// The caller has checked that every field, every pointwise input and every
/// destination share one shape.
pub(super) fn central4_map_triple_into<T, const N: usize, const M: usize, F>(
    terms: [(Axis, ArrayView3<'_, T>); N],
    pointwise: [ArrayView3<'_, T>; M],
    dst: [&mut ArrayViewMut3<'_, T>; 3],
    spacing: [T; 3],
    combine: F,
) -> Result<()>
where
    T: RealField + FloatElement + Copy,
    F: Fn([T; N], [T; M]) -> [T; 3] + Send + Sync,
{
    let shape = dst[0].shape();
    let scales: [Scales<T>; N] = core::array::from_fn(|j| Scales::new(spacing[terms[j].0.index()]));
    let axes: [Axis; N] = core::array::from_fn(|j| terms[j].0);
    let mut dense = true;
    let fields: [&[T]; N] = core::array::from_fn(|j| {
        terms[j].1.as_slice().unwrap_or_else(|| {
            dense = false;
            &[]
        })
    });
    let scalars: [&[T]; M] = core::array::from_fn(|k| {
        pointwise[k].as_slice().unwrap_or_else(|| {
            dense = false;
            &[]
        })
    });
    let [first, second, third] = dst;
    if dense {
        if let (Some(a), Some(b), Some(c)) = (
            first.as_mut_slice(),
            second.as_mut_slice(),
            third.as_mut_slice(),
        ) {
            map_triple_dense(fields, scalars, axes, [a, b, c], shape, scales, &combine);
            return Ok(());
        }
    }
    map_triple_strided(
        terms,
        pointwise,
        [first, second, third],
        shape,
        scales,
        &combine,
    );
    Ok(())
}

fn map_triple_dense<T, const N: usize, const M: usize, F>(
    fields: [&[T]; N],
    pointwise: [&[T]; M],
    axes: [Axis; N],
    out: [&mut [T]; 3],
    shape: [usize; 3],
    scales: [Scales<T>; N],
    combine: &F,
) where
    T: RealField + FloatElement + Copy,
    F: Fn([T; N], [T; M]) -> [T; 3] + Send + Sync,
{
    let [nx, ny, nz] = shape;
    let plane_len = ny * nz;
    // Three output elements, the four neighbours each term's stencil reaches,
    // and one element of each pointwise input.
    let element_bytes = (3 + N * (ELEMENTS_PER_UNIT - 1) + M) * size_of::<T>();
    let [first, second, third] = out;
    for_each_plane_mut_triple_with(
        first,
        second,
        third,
        plane_len,
        element_bytes,
        || vec![<T as NumericElement>::ZERO; N * nz],
        |scratch, x, planes| {
            let [a, b, c] = planes;
            let rows = a
                .chunks_exact_mut(nz)
                .zip(b.chunks_exact_mut(nz))
                .zip(c.chunks_exact_mut(nz));
            for (y, ((first_row, second_row), third_row)) in rows.enumerate() {
                let base = x * plane_len + y * nz;
                for (j, values) in scratch.chunks_exact_mut(nz).enumerate() {
                    match axes[j] {
                        Axis::X => Stencil::at(x, nx).apply_lane(scales[j], values, |o| {
                            let plane = x.wrapping_add_signed(o) * plane_len + y * nz;
                            &fields[j][plane..plane + nz]
                        }),
                        Axis::Y => Stencil::at(y, ny).apply_lane(scales[j], values, |o| {
                            let row = x * plane_len + y.wrapping_add_signed(o) * nz;
                            &fields[j][row..row + nz]
                        }),
                        Axis::Z => sweep_row(scales[j], values, &fields[j][base..base + nz]),
                    }
                }
                let lanes = first_row
                    .iter_mut()
                    .zip(second_row.iter_mut())
                    .zip(third_row.iter_mut());
                for (k, ((one, two), three)) in lanes.enumerate() {
                    let [x_value, y_value, z_value] = combine(
                        core::array::from_fn(|j| scratch[j * nz + k]),
                        core::array::from_fn(|p| pointwise[p][base + k]),
                    );
                    *one = x_value;
                    *two = y_value;
                    *three = z_value;
                }
            }
        },
    );
}

fn map_triple_strided<T, const N: usize, const M: usize, F>(
    terms: [(Axis, ArrayView3<'_, T>); N],
    pointwise: [ArrayView3<'_, T>; M],
    dst: [&mut ArrayViewMut3<'_, T>; 3],
    shape: [usize; 3],
    scales: [Scales<T>; N],
    combine: &F,
) where
    T: RealField + FloatElement + Copy,
    F: Fn([T; N], [T; M]) -> [T; 3] + Send + Sync,
{
    let [first, second, third] = dst;
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let index = [i, j, k];
                let derivatives: [T; N] = core::array::from_fn(|t| {
                    let axis = terms[t].0.index();
                    let c = index[axis];
                    Stencil::at(c, shape[axis]).apply(scales[t], |o| {
                        let mut neighbour = index;
                        neighbour[axis] = c.wrapping_add_signed(o);
                        terms[t].1[neighbour]
                    })
                });
                let values = combine(derivatives, core::array::from_fn(|p| pointwise[p][index]));
                first[index] = values[0];
                second[index] = values[1];
                third[index] = values[2];
            }
        }
    }
}

fn map_strided<T, const N: usize, const M: usize, F>(
    terms: [(Axis, ArrayView3<'_, T>); N],
    pointwise: [ArrayView3<'_, T>; M],
    dst: &mut ArrayViewMut3<'_, T>,
    shape: [usize; 3],
    scales: [Scales<T>; N],
    combine: &F,
) where
    T: RealField + FloatElement + Copy,
    F: Fn([T; N], [T; M]) -> T + Send + Sync,
{
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let index = [i, j, k];
                let derivatives: [T; N] = core::array::from_fn(|t| {
                    let axis = terms[t].0.index();
                    let c = index[axis];
                    Stencil::at(c, shape[axis]).apply(scales[t], |o| {
                        let mut neighbour = index;
                        neighbour[axis] = c.wrapping_add_signed(o);
                        terms[t].1[neighbour]
                    })
                });
                dst[index] = combine(derivatives, core::array::from_fn(|p| pointwise[p][index]));
            }
        }
    }
}

fn divergence_strided<T>(
    fields: [ArrayView3<'_, T>; 3],
    dst: &mut ArrayViewMut3<'_, T>,
    shape: [usize; 3],
    scales: [Scales<T>; 3],
) where
    T: RealField + FloatElement + Copy,
{
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let index = [i, j, k];
                let mut total = <T as NumericElement>::ZERO;
                for (axis, field) in fields.iter().enumerate() {
                    let c = index[axis];
                    let value = Stencil::at(c, shape[axis]).apply(scales[axis], |o| {
                        let mut neighbour = index;
                        neighbour[axis] = c.wrapping_add_signed(o);
                        field[neighbour]
                    });
                    if axis == 0 {
                        total = value;
                    } else {
                        total += value;
                    }
                }
                dst[index] = total;
            }
        }
    }
}

/// The z-axis derivative of `row`, added to what `out` already holds.
fn add_row<T: RealField + FloatElement + Copy>(scales: Scales<T>, out: &mut [T], row: &[T]) {
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
fn sweep_row<T: RealField + FloatElement + Copy>(scales: Scales<T>, out: &mut [T], row: &[T]) {
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

fn sweep_strided<T>(
    field: ArrayView3<T>,
    dst: &mut ArrayViewMut3<'_, T>,
    shape: [usize; 3],
    axis: Axis,
    scales: Scales<T>,
) where
    T: RealField + FloatElement + Copy,
{
    let d = axis.index();
    let n = shape[d];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let index = [i, j, k];
                let c = index[d];
                dst[index] = Stencil::at(c, n).apply(scales, |o| {
                    let mut neighbour = index;
                    neighbour[d] = c.wrapping_add_signed(o);
                    field[neighbour]
                });
            }
        }
    }
}

#[cfg(test)]
mod tests;
