//! Volume ray line integrals: [`ray_line_integrals_into`] and [`ray_line_integrals`].
//!
//! CPU counterpart of `hephaestus_core::RayIntegralOps` (1:1 parity): each ray
//! is clipped to the node-centre bounding box by the slab method, the chord
//! is split into `max(ceil(len / step), 1)` equal segments, and each
//! segment's midpoint sample of the trilinearly interpolated field is
//! weighted by the segment length. Interpolation reuses this crate's
//! [`trilinear_index_space`](crate::application::interpolation::trilinear_index_space)
//! directly; a midpoint outside the node range contributes zero, as in the
//! kernels.
//!
//! The scalar type is fixed at `f32`, exactly as the seam fixes it: every
//! backend kernel marches in `f32`, and a generic scalar dimension enters
//! here when the kernels ship it, not before. All arithmetic below keeps the
//! host reference's operation order so CPU and device agree lane for lane.
//! Dimensions ride on the field view's own shape, so the seam's `u32` / 2^24
//! parameter-ABI limits have no CPU analog and are not checked: on the host
//! there are no `f32` parameter lanes to keep exact.
//! Zero-copy: integrals accumulate in scalar registers and land directly in
//! the caller-owned output view.

use crate::application::index::validate_mutable_output;
use crate::application::interpolation::trilinear_index_space;
use leto::{Array, ArrayView, ArrayViewMut, Layout, LetoError, Result, VecStorage};

/// Number of `f32` lanes per packed ray (`origin.xyz`, then `direction.xyz`).
pub const RAY_STRIDE: usize = 6;

/// Parameter `t` along a ray where it enters and leaves the node-centre box,
/// or `None` when it misses or grazes it.
///
/// Verbatim port of the host reference's slab clip, including the infinities
/// an axis-parallel ray divides into: they fall out of the min/max exactly
/// as in the kernels.
fn chord(
    origin: [f32; 3],
    direction: [f32; 3],
    dims: [usize; 3],
    box_origin: [f32; 3],
    spacing: [f32; 3],
) -> Option<(f32, f32)> {
    let mut enter = f32::NEG_INFINITY;
    let mut exit = f32::INFINITY;
    for axis in 0..3 {
        let low = box_origin[axis];
        let high = low + (dims[axis] - 1) as f32 * spacing[axis];
        // An axis-parallel ray divides by zero; the resulting infinities fall
        // out of the min/max below exactly as in the kernels.
        let inverse = 1.0 / direction[axis];
        let near = (low - origin[axis]) * inverse;
        let far = (high - origin[axis]) * inverse;
        enter = enter.max(near.min(far));
        exit = exit.min(near.max(far));
    }
    (exit - enter > 0.0).then_some((enter, exit))
}

/// Number of march segments for a chord, as the kernels count them.
fn segment_count(length: f32, step: f32) -> Result<u32> {
    let segments = (length / step).ceil().max(1.0);
    // `u32::MAX as f32` rounds up to 2^32, so the bound is strict: every
    // whole `f32` below it converts exactly.
    if segments < u32::MAX as f32 {
        Ok(segments as u32)
    } else {
        Err(LetoError::InvalidInput(format!(
            "a ray chord of length {length} at step {step} needs more than u32::MAX segments"
        )))
    }
}

/// Integrate `field` along each ray of `rays` into `out` (one value per ray).
///
/// `field` is the `[nx, ny, nz]` node field with node `(ix, iy, iz)` at
/// `origin + index * spacing` per axis; `rays` is `[n, 6]` packed
/// `[origin xyz, direction xyz]` records; `out` must be `[n]` exactly.
/// Strided and offset views are served on `rays` and `out`; the output must
/// be injective.
///
/// # Errors
///
/// Returns [`LetoError::ShapeMismatch`] when `rays` is not `[n, 6]` or `out`
/// is not `[n]`, [`LetoError::InvalidInput`] for a zero field axis, a
/// non-finite or non-positive `step`, non-finite origin/spacing entries,
/// non-positive spacing, or a chord needing more than `u32::MAX` segments,
/// and the layout errors for invalid storage lengths or an aliased output.
pub fn ray_line_integrals_into(
    field: &ArrayView<'_, f32, 3>,
    origin: [f32; 3],
    spacing: [f32; 3],
    rays: &ArrayView<'_, f32, 2>,
    step: f32,
    out: &mut ArrayViewMut<'_, f32, 1>,
) -> Result<()> {
    let dims = field.shape();
    if dims.contains(&0) {
        return Err(LetoError::InvalidInput(format!(
            "ray field dimensions must be positive, got {dims:?}"
        )));
    }
    let [n_rays, stride] = rays.shape();
    if stride != RAY_STRIDE {
        return Err(LetoError::ShapeMismatch {
            lhs: [n_rays, RAY_STRIDE].to_vec(),
            rhs: rays.shape().to_vec(),
        });
    }
    if out.shape() != [n_rays] {
        return Err(LetoError::ShapeMismatch {
            lhs: [n_rays].to_vec(),
            rhs: out.shape().to_vec(),
        });
    }
    if !(step.is_finite() && step > 0.0) {
        return Err(LetoError::InvalidInput(format!(
            "ray-march step must be finite and positive, got {step}"
        )));
    }
    if origin
        .into_iter()
        .chain(spacing)
        .any(|value| !value.is_finite())
    {
        return Err(LetoError::InvalidInput(
            "ray field origin and spacing must be finite".to_string(),
        ));
    }
    if spacing.iter().any(|&value| value <= 0.0) {
        return Err(LetoError::InvalidInput(format!(
            "ray field spacing must be positive, got {spacing:?}"
        )));
    }
    field.layout().validate_storage_len(field.data().len())?;
    rays.layout().validate_storage_len(rays.data().len())?;
    validate_mutable_output(out, "ray integral output")?;

    let last_node = dims.map(|extent| (extent - 1) as f32);
    let rays_layout = rays.layout();
    let out_layout = out.layout();
    let rays_data = rays.data();
    let out_data = out.data_mut();
    let rays_base = rays_layout.offset() as isize;
    let out_base = out_layout.offset() as isize;
    let (ray_row, ray_col) = (rays_layout.strides()[0], rays_layout.strides()[1]);
    let out_stride = out_layout.strides()[0];
    for ray in 0..n_rays {
        let record = |lane: usize| {
            rays_data[(rays_base + ray as isize * ray_row + lane as isize * ray_col) as usize]
        };
        let ray_origin = [record(0), record(1), record(2)];
        let direction = [record(3), record(4), record(5)];
        let integral = match chord(ray_origin, direction, dims, origin, spacing) {
            None => 0.0,
            Some((enter, exit)) => {
                let length = exit - enter;
                let segments = segment_count(length, step)?;
                let segment = length / segments as f32;
                let mut sum = 0.0f32;
                for index in 0..segments {
                    let t = enter + (index as f32 + 0.5) * segment;
                    let node = [0, 1, 2].map(|axis| {
                        (ray_origin[axis] + direction[axis] * t - origin[axis]) / spacing[axis]
                    });
                    let inside = node
                        .iter()
                        .zip(last_node)
                        .all(|(&coordinate, last)| (0.0..=last).contains(&coordinate));
                    if inside {
                        sum += trilinear_index_space(*field, node[0], node[1], node[2]);
                    }
                }
                sum * segment
            }
        };
        out_data[(out_base + ray as isize * out_stride) as usize] = integral;
    }
    Ok(())
}

/// Integrate `field` along each ray of `rays` into a newly allocated
/// C-contiguous `[n]` output.
///
/// See [`ray_line_integrals_into`] for the contract and errors.
pub fn ray_line_integrals(
    field: &ArrayView<'_, f32, 3>,
    origin: [f32; 3],
    spacing: [f32; 3],
    rays: &ArrayView<'_, f32, 2>,
    step: f32,
) -> Result<Array<f32, VecStorage<f32>, 1>> {
    let [n_rays, _] = rays.shape();
    let layout = Layout::c_contiguous([n_rays])?;
    let size = layout.checked_size()?;
    let mut out = Array::new(layout, VecStorage::uninit(size))?;
    ray_line_integrals_into(field, origin, spacing, rays, step, &mut out.view_mut())?;
    Ok(out)
}
