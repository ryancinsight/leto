//! Strided (logically-indexed) fallback traversal of the fourth-order
//! stencils, taken when a field is not C-contiguous along the differentiated
//! axis.

use core::ops::Range;

use eunomia::{FloatElement, NumericElement, RealField};
use leto::{ArrayView3, ArrayViewMut3};

use super::super::window::{PlaneWindow, PlaneWindowMut};

use super::super::leapfrog::Axis;
use super::stencil::{Scales, Stencil};

pub(super) fn map_strided<T, const N: usize, const M: usize, const K: usize, F>(
    terms: [(Axis, PlaneWindow<'_, T>); N],
    pointwise: [PlaneWindow<'_, T>; M],
    mut dst: [PlaneWindowMut<'_, '_, T>; K],
    shape: [usize; 3],
    planes: Range<usize>,
    scales: [Scales<T>; N],
    combine: &F,
) where
    T: RealField + FloatElement + Copy,
    F: Fn([T; N], [T; M]) -> [T; K] + Send + Sync,
{
    for i in planes {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let index = [i, j, k];
                let derivatives: [T; N] = core::array::from_fn(|t| {
                    let axis = terms[t].0.index();
                    let c = index[axis];
                    Stencil::at(c, shape[axis]).apply(scales[t], |o| {
                        let mut neighbour = index;
                        neighbour[axis] = c.wrapping_add_signed(o);
                        neighbour[0] -= terms[t].1.first();
                        terms[t].1.view()[neighbour]
                    })
                });
                let at = |window: &PlaneWindow<'_, T>| window.view()[[i - window.first(), j, k]];
                let values = combine(derivatives, core::array::from_fn(|p| at(&pointwise[p])));
                for (destination, value) in dst.iter_mut().zip(values) {
                    let local = [i - destination.first(), j, k];
                    destination.view_mut()[local] = value;
                }
            }
        }
    }
}

pub(super) fn divergence_strided<T>(
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

pub(super) fn sweep_strided<T>(
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
