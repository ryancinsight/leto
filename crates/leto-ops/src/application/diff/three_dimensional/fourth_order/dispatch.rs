//! Entry points dispatching each fourth-order operation to its
//! [`dense`](super::dense) or [`strided`](super::strided) implementation by
//! field contiguity.

use core::ops::Range;

use eunomia::{FloatElement, RealField};
use leto::{ArrayView3, ArrayViewMut3, Result};

use super::super::window::{PlaneWindow, PlaneWindowMut};

use super::super::leapfrog::Axis;
use super::dense::{divergence_dense, map_dense, sweep_dense, DenseTerm};
use super::stencil::Scales;
use super::strided::{divergence_strided, map_strided, sweep_strided};

/// `dst = ∂field/∂axis` with the closure in the [module documentation](super).
///
/// The caller has checked that `dst` has `field`'s shape.
pub(in super::super) fn central4_into<T>(
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

/// `dst = ∂fields[0]/∂x + ∂fields[1]/∂y + ∂fields[2]/∂z`, summed in that order.
///
/// The caller has checked that every field and `dst` share one shape.
pub(in super::super) fn central4_divergence_into<T>(
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

/// One fused pass over the grid planes `planes`, writing `K` destinations:
/// each output lane receives the `N` axis derivatives named by `terms` and
/// the `M` pointwise values at that lane, and the `K` values the
/// destinations hold there, and `combine` returns the `K`
/// values written there. Planes outside the range are not touched.
///
/// Each field and each destination hold a window of a grid of `grid_planes`
/// x-planes; the caller has checked that every window covers what the pass
/// reads or writes in it, that all share the destinations' lanes, and that
/// the destinations hold the same planes.
pub(in super::super) fn central4_map_into<T, const N: usize, const M: usize, const K: usize, F>(
    terms: [(Axis, PlaneWindow<'_, T>); N],
    pointwise: [PlaneWindow<'_, T>; M],
    mut dst: [PlaneWindowMut<'_, '_, T>; K],
    grid_planes: usize,
    planes: Range<usize>,
    spacing: [T; 3],
    combine: F,
) -> Result<()>
where
    T: RealField + FloatElement + Copy,
    F: Fn([T; N], [T; M], [T; K]) -> [T; K] + Send + Sync,
{
    let Some(first) = dst.first() else {
        return Ok(());
    };
    let [_, ny, nz] = first.shape();
    let origin = first.first();
    let shape = [grid_planes, ny, nz];
    let scales: [Scales<T>; N] = core::array::from_fn(|j| Scales::new(spacing[terms[j].0.index()]));
    if let (Some(fields), Some(scalars)) =
        (dense_terms(&terms, scales), dense_pointwise(&pointwise))
    {
        let out = dst
            .each_mut()
            .map(|window| window.view_mut().as_mut_slice());
        if out.iter().all(Option::is_some) {
            let out = out.map(|slice| slice.expect("invariant: every slice was checked present"));
            map_dense(fields, scalars, (out, origin), shape, planes, &combine);
            return Ok(());
        }
    }
    map_strided(terms, pointwise, dst, shape, planes, scales, &combine);
    Ok(())
}

/// The dense form of a windowed pass's derivative terms, or `None` unless
/// every field is C-contiguous.
fn dense_terms<'a, T: Copy, const N: usize>(
    terms: &[(Axis, PlaneWindow<'a, T>); N],
    scales: [Scales<T>; N],
) -> Option<[DenseTerm<'a, T>; N]> {
    let fields = terms.each_ref().map(|(_, window)| window.view().as_slice());
    fields.iter().all(Option::is_some).then(|| {
        core::array::from_fn(|j| DenseTerm {
            axis: terms[j].0,
            field: fields[j].expect("invariant: every field was checked present"),
            origin: terms[j].1.first(),
            scales: scales[j],
        })
    })
}

/// The dense form of a windowed pass's pointwise inputs, each with the grid
/// plane its storage starts at, or `None` unless every one is C-contiguous.
fn dense_pointwise<'a, T: Copy, const M: usize>(
    pointwise: &[PlaneWindow<'a, T>; M],
) -> Option<[(&'a [T], usize); M]> {
    let values = pointwise.each_ref().map(|window| window.view().as_slice());
    values.iter().all(Option::is_some).then(|| {
        core::array::from_fn(|k| {
            (
                values[k].expect("invariant: every input was checked present"),
                pointwise[k].first(),
            )
        })
    })
}
