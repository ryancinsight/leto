//! The staggered gradient and divergence fused with the pointwise updates
//! that consume them.
//!
//! A leapfrog velocity update is `v ← v − (Δt/ρ) G p`: one gradient sweep and
//! one pointwise pass per component. Swept into a grid-sized buffer, the
//! gradient is written out and read back before the update can use it. Here
//! each row's gradient is swept into a row-sized buffer its task keeps, so the
//! gradient never leaves the first-level cache, and the update reads it from
//! there in the same task. A plane-sized buffer measured slower than the
//! separate passes at 64 cubed: a fresh 32 KB allocation per task per call.

use eunomia::{FloatElement, NumericElement, RealField};
use leto::{Array3, ArrayView3, ArrayViewMut3, Result};

use super::kernels::{divergence, divergence_row, gradient, gradient_row, plane_reads};
use super::{assert_grid_shaped, Axis, StaggeredLeapfrog3D};
use crate::infrastructure::parallel::for_each_plane_mut_many_with;

impl<T: RealField + FloatElement + Copy> StaggeredLeapfrog3D<T> {
    /// The gradient along `axis` combined, face by face, into `dst`:
    /// `dst[i] = combine(g[i], values, dst[i])`, where `g` is what
    /// [`Self::gradient_into`] would write, `values[k]` is `pointwise[k]` at
    /// that face, and the last argument is the value `dst` held before.
    ///
    /// `g` is the same sweep [`Self::gradient_into`] runs, so a `combine`
    /// doing the arithmetic a separate pass over the swept gradient would do
    /// gives that pass's values to the bit. What it saves is the gradient's
    /// round trip: swept a plane at a time into a buffer its task keeps, it is
    /// read back from cache instead of written to and read from a grid.
    ///
    /// # Errors
    ///
    /// Returns [`leto::LetoError::InvalidInput`] when `dst` or a pointwise
    /// input does not have `field`'s shape.
    pub fn map_gradient_into<const M: usize, F>(
        &self,
        axis: Axis,
        field: ArrayView3<'_, T>,
        pointwise: [ArrayView3<'_, T>; M],
        dst: &mut ArrayViewMut3<'_, T>,
        combine: F,
    ) -> Result<()>
    where
        F: Fn(T, [T; M], T) -> T + Send + Sync,
    {
        let shape = field.shape();
        assert_grid_shaped(dst.shape(), shape, "map_gradient")?;
        for values in &pointwise {
            assert_grid_shaped(values.shape(), shape, "map_gradient pointwise input")?;
        }
        let values = pointwise.each_ref().map(|values| values.as_slice());
        if let (Some(source), Some(target)) = (field.as_slice(), dst.as_mut_slice()) {
            if values.iter().all(Option::is_some) {
                let values = values
                    .map(|values| values.expect("invariant: every input was checked present"));
                map_dense(self, axis, source, values, target, shape, &combine);
                return Ok(());
            }
        }
        map_logical(self, axis, field, pointwise, dst, shape, &combine);
        Ok(())
    }

    /// The divergence of the face-centred components `fields` -- `fields[0]`
    /// along x, `fields[1]` along y, `fields[2]` along z -- combined, cell by
    /// cell, into `dst`: `dst[i] = combine(d, values, dst[i])`, where `d[a]`
    /// is what [`Self::divergence_into`] writes for `fields[a]` along axis
    /// `a`, `values[k]` is `pointwise[k]` at that cell, and the last argument
    /// is the value `dst` held before.
    ///
    /// Each `d[a]` is the same sum [`Self::divergence_into`] forms, term for
    /// term in the same order, so a `combine` doing the arithmetic separate
    /// passes over the three swept divergences would do gives their values to
    /// the bit. The three divergences are swept a row at a time into buffers
    /// the task keeps, so none of them is written to a grid.
    ///
    /// # Errors
    ///
    /// Returns [`leto::LetoError::InvalidInput`] when a field, a pointwise
    /// input or `dst` does not have `fields[0]`'s shape.
    pub fn map_divergence_into<const M: usize, F>(
        &self,
        fields: [ArrayView3<'_, T>; 3],
        pointwise: [ArrayView3<'_, T>; M],
        dst: &mut ArrayViewMut3<'_, T>,
        combine: F,
    ) -> Result<()>
    where
        F: Fn([T; 3], [T; M], T) -> T + Send + Sync,
    {
        let shape = fields[0].shape();
        assert_grid_shaped(dst.shape(), shape, "map_divergence")?;
        for field in &fields[1..] {
            assert_grid_shaped(field.shape(), shape, "map_divergence field")?;
        }
        for values in &pointwise {
            assert_grid_shaped(values.shape(), shape, "map_divergence pointwise input")?;
        }
        let sources = fields.each_ref().map(|field| field.as_slice());
        let values = pointwise.each_ref().map(|values| values.as_slice());
        if let Some(target) = dst.as_mut_slice() {
            if sources.iter().chain(&values).all(Option::is_some) {
                let sources = sources
                    .map(|source| source.expect("invariant: every field was checked present"));
                let values = values
                    .map(|values| values.expect("invariant: every input was checked present"));
                map_divergence_dense(self, sources, values, target, shape, &combine);
                return Ok(());
            }
        }
        map_divergence_logical(self, fields, pointwise, dst, shape, &combine);
        Ok(())
    }
}

/// The three axes a divergence's components are differentiated along.
const COMPONENT_AXES: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

/// [`StaggeredLeapfrog3D::map_divergence_into`] over C-contiguous storage: a
/// plane task sweeps each row's three divergences into the row buffers it
/// keeps, then combines them into the destination row.
fn map_divergence_dense<T, const M: usize, F>(
    op: &StaggeredLeapfrog3D<T>,
    sources: [&[T]; 3],
    values: [&[T]; M],
    target: &mut [T],
    shape: [usize; 3],
    combine: &F,
) where
    T: RealField + FloatElement + Copy,
    F: Fn([T; 3], [T; M], T) -> T + Send + Sync,
{
    let [_, ny, nz] = shape;
    let plane = ny * nz;
    if plane == 0 || target.is_empty() {
        return;
    }
    // The three sweeps' reads, the destination read and written, and the
    // pointwise inputs.
    let reads: usize = COMPONENT_AXES
        .iter()
        .map(|&axis| plane_reads(op, axis.index()))
        .sum();
    let element_bytes = (reads + 2 + M) * size_of::<T>();
    for_each_plane_mut_many_with(
        [target],
        plane,
        element_bytes,
        || vec![<T as NumericElement>::ZERO; 3 * nz],
        |divergences, x, [out]| {
            for (y, out) in out.chunks_exact_mut(nz).enumerate() {
                for ((axis, source), row) in COMPONENT_AXES
                    .into_iter()
                    .zip(sources)
                    .zip(divergences.chunks_exact_mut(nz))
                {
                    divergence_row(op, axis, source, row, [x, y], shape);
                }
                let (dx, rest) = divergences.split_at(nz);
                let (dy, dz) = rest.split_at(nz);
                let start = (x * ny + y) * nz;
                let values: [&[T]; M] = core::array::from_fn(|k| &values[k][start..start + nz]);
                for (i, held) in out.iter_mut().enumerate() {
                    *held = combine(
                        [dx[i], dy[i], dz[i]],
                        core::array::from_fn(|k| values[k][i]),
                        *held,
                    );
                }
            }
        },
    );
}

/// [`StaggeredLeapfrog3D::map_divergence_into`] over any layout: each
/// divergence is swept whole into a temporary, then combined in logical
/// order.
fn map_divergence_logical<T, const M: usize, F>(
    op: &StaggeredLeapfrog3D<T>,
    fields: [ArrayView3<'_, T>; 3],
    pointwise: [ArrayView3<'_, T>; M],
    dst: &mut ArrayViewMut3<'_, T>,
    shape: [usize; 3],
    combine: &F,
) where
    T: RealField + FloatElement + Copy,
    F: Fn([T; 3], [T; M], T) -> T + Send + Sync,
{
    let swept = COMPONENT_AXES.map(|axis| {
        let mut swept = Array3::from_elem(shape, <T as NumericElement>::ZERO);
        divergence(op, axis, fields[axis.index()], &mut swept.view_mut(), shape);
        swept
    });
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let at = [i, j, k];
                dst[at] = combine(
                    swept.each_ref().map(|swept| swept[at]),
                    pointwise.each_ref().map(|values| values[at]),
                    dst[at],
                );
            }
        }
    }
}

/// [`StaggeredLeapfrog3D::map_gradient_into`] over C-contiguous storage: a
/// plane task sweeps each row's gradient into the row buffer it keeps, then
/// combines it into the destination row.
fn map_dense<T, const M: usize, F>(
    op: &StaggeredLeapfrog3D<T>,
    axis: Axis,
    source: &[T],
    values: [&[T]; M],
    target: &mut [T],
    shape: [usize; 3],
    combine: &F,
) where
    T: RealField + FloatElement + Copy,
    F: Fn(T, [T; M], T) -> T + Send + Sync,
{
    let [_, ny, nz] = shape;
    let plane = ny * nz;
    if plane == 0 || source.is_empty() {
        return;
    }
    // The sweep's reads, the destination read and written, and the
    // pointwise inputs.
    let element_bytes = (plane_reads(op, axis.index()) + 2 + M) * size_of::<T>();
    for_each_plane_mut_many_with(
        [target],
        plane,
        element_bytes,
        || vec![<T as NumericElement>::ZERO; nz],
        |gradient, x, [out]| {
            for (y, out) in out.chunks_exact_mut(nz).enumerate() {
                gradient_row(op, axis, source, gradient, [x, y], shape);
                let start = (x * ny + y) * nz;
                let values: [&[T]; M] = core::array::from_fn(|k| &values[k][start..start + nz]);
                for (i, (held, &g)) in out.iter_mut().zip(gradient.iter()).enumerate() {
                    *held = combine(g, core::array::from_fn(|k| values[k][i]), *held);
                }
            }
        },
    );
}

/// [`StaggeredLeapfrog3D::map_gradient_into`] over any layout: the gradient
/// is swept whole into a temporary, then combined in logical order.
fn map_logical<T, const M: usize, F>(
    op: &StaggeredLeapfrog3D<T>,
    axis: Axis,
    field: ArrayView3<'_, T>,
    pointwise: [ArrayView3<'_, T>; M],
    dst: &mut ArrayViewMut3<'_, T>,
    shape: [usize; 3],
    combine: &F,
) where
    T: RealField + FloatElement + Copy,
    F: Fn(T, [T; M], T) -> T + Send + Sync,
{
    let mut swept = Array3::from_elem(shape, <T as NumericElement>::ZERO);
    gradient(op, axis, field, &mut swept.view_mut(), shape);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let at = [i, j, k];
                dst[at] = combine(
                    swept[at],
                    pointwise.each_ref().map(|values| values[at]),
                    dst[at],
                );
            }
        }
    }
}

#[cfg(test)]
mod tests;
