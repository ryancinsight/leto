//! Entry points dispatching each fourth-order operation to its
//! [`dense`](super::dense) or [`strided`](super::strided) implementation by
//! field contiguity.

use eunomia::{FloatElement, RealField};
use leto::{ArrayView3, ArrayViewMut3, Result};

use super::super::leapfrog::Axis;
use super::dense::{divergence_dense, map_dense, map_triple_dense, sweep_dense};
use super::stencil::Scales;
use super::strided::{divergence_strided, map_strided, map_triple_strided, sweep_strided};

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

/// One fused pass over `dst`: each output lane receives the `N` axis
/// derivatives named by `terms` and the `M` pointwise values at that lane,
/// and `combine` decides what to write.
///
/// The caller has checked that every field, every pointwise input and `dst`
/// share one shape.
pub(in super::super) fn central4_map_into<T, const N: usize, const M: usize, F>(
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

/// [`central4_map_into`] writing three destinations from one derivative pass.
///
/// The caller has checked that every field, every pointwise input and every
/// destination share one shape.
pub(in super::super) fn central4_map_triple_into<T, const N: usize, const M: usize, F>(
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
