//! Strided (logically-indexed) fallback traversal of the fourth-order
//! stencils, taken when a field is not C-contiguous along the differentiated
//! axis.

use eunomia::{FloatElement, NumericElement, RealField};
use leto::{ArrayView3, ArrayViewMut3};

use super::super::leapfrog::Axis;
use super::stencil::{Scales, Stencil};

pub(super) fn map_triple_strided<T, const N: usize, const M: usize, F>(
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

pub(super) fn map_strided<T, const N: usize, const M: usize, F>(
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
