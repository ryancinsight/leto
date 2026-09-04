//! Yee staggered stencil kernels (forward and backward) behind
//! [`super::FiniteDifference3D`].
use eunomia::{FloatElement, NumericElement, RealField};
use leto::{ArrayView3, ArrayViewMut3, LetoError, Result};

use crate::application::zip::zip_mut_with;

// ── Staggered forward (Yee face forward) ─────────────────────────────────────

pub(super) fn staggered_forward_x_into<T>(
    field: ArrayView3<T>,
    dst: &mut ArrayViewMut3<'_, T>,
    nx: usize,
    ny: usize,
    nz: usize,
    hx: T,
) -> Result<()>
where
    T: RealField + FloatElement + Copy,
{
    if nx < 2 {
        return Err(LetoError::InvalidInput(
            "StaggeredForward: need at least 2 points on the differentiated axis".into(),
        ));
    }
    let inv_h = <T as NumericElement>::ONE / hx;
    let mut dst_slice = dst
        .reborrow()
        .slice_mut(&[(0, nx - 1, 1), (0, ny, 1), (0, nz, 1)])
        .unwrap();
    let field_hi = field.slice(&[(1, nx, 1), (0, ny, 1), (0, nz, 1)]).unwrap();
    let field_lo = field
        .slice(&[(0, nx - 1, 1), (0, ny, 1), (0, nz, 1)])
        .unwrap();
    zip_mut_with(&mut dst_slice, (&field_hi, &field_lo), |r, (&hi, &lo)| {
        *r = (hi - lo) * inv_h
    })
    .unwrap();
    Ok(())
}

pub(super) fn staggered_forward_y_into<T>(
    field: ArrayView3<T>,
    dst: &mut ArrayViewMut3<'_, T>,
    nx: usize,
    ny: usize,
    nz: usize,
    hy: T,
) -> Result<()>
where
    T: RealField + FloatElement + Copy,
{
    if ny < 2 {
        return Err(LetoError::InvalidInput(
            "StaggeredForward: need at least 2 points on the differentiated axis".into(),
        ));
    }
    let inv_h = <T as NumericElement>::ONE / hy;
    let mut dst_slice = dst
        .reborrow()
        .slice_mut(&[(0, nx, 1), (0, ny - 1, 1), (0, nz, 1)])
        .unwrap();
    let field_hi = field.slice(&[(0, nx, 1), (1, ny, 1), (0, nz, 1)]).unwrap();
    let field_lo = field
        .slice(&[(0, nx, 1), (0, ny - 1, 1), (0, nz, 1)])
        .unwrap();
    zip_mut_with(&mut dst_slice, (&field_hi, &field_lo), |r, (&hi, &lo)| {
        *r = (hi - lo) * inv_h
    })
    .unwrap();
    Ok(())
}

pub(super) fn staggered_forward_z_into<T>(
    field: ArrayView3<T>,
    dst: &mut ArrayViewMut3<'_, T>,
    nx: usize,
    ny: usize,
    nz: usize,
    hz: T,
) -> Result<()>
where
    T: RealField + FloatElement + Copy,
{
    if nz < 2 {
        return Err(LetoError::InvalidInput(
            "StaggeredForward: need at least 2 points on the differentiated axis".into(),
        ));
    }
    let inv_h = <T as NumericElement>::ONE / hz;
    let mut dst_slice = dst
        .reborrow()
        .slice_mut(&[(0, nx, 1), (0, ny, 1), (0, nz - 1, 1)])
        .unwrap();
    let field_hi = field.slice(&[(0, nx, 1), (0, ny, 1), (1, nz, 1)]).unwrap();
    let field_lo = field
        .slice(&[(0, nx, 1), (0, ny, 1), (0, nz - 1, 1)])
        .unwrap();
    zip_mut_with(&mut dst_slice, (&field_hi, &field_lo), |r, (&hi, &lo)| {
        *r = (hi - lo) * inv_h
    })
    .unwrap();
    Ok(())
}

// ── Staggered backward (kwavers-side convention: dst matches field shape) ────

pub(super) fn staggered_backward_x_into<T>(
    field: ArrayView3<T>,
    dst: &mut ArrayViewMut3<'_, T>,
    nx: usize,
    ny: usize,
    nz: usize,
    hx: T,
) -> Result<()>
where
    T: RealField + FloatElement + Copy,
{
    if nx < 2 {
        return Err(LetoError::InvalidInput(
            "StaggeredBackward: need at least 2 points on the differentiated axis".into(),
        ));
    }
    let inv_h = <T as NumericElement>::ONE / hx;

    // Backward on i ∈ [1, nx-1]
    let mut dst_int = dst
        .reborrow()
        .slice_mut(&[(1, nx, 1), (0, ny, 1), (0, nz, 1)])
        .unwrap();
    let field_hi = field.slice(&[(1, nx, 1), (0, ny, 1), (0, nz, 1)]).unwrap();
    let field_lo = field
        .slice(&[(0, nx - 1, 1), (0, ny, 1), (0, nz, 1)])
        .unwrap();
    zip_mut_with(&mut dst_int, (&field_hi, &field_lo), |r, (&hi, &lo)| {
        *r = (hi - lo) * inv_h
    })
    .unwrap();
    // Forward fall-back at i=0
    let mut dst_left = dst
        .reborrow()
        .slice_mut(&[(0, 1, 1), (0, ny, 1), (0, nz, 1)])
        .unwrap();
    let field_hi_left = field.slice(&[(1, 2, 1), (0, ny, 1), (0, nz, 1)]).unwrap();
    let field_lo_left = field.slice(&[(0, 1, 1), (0, ny, 1), (0, nz, 1)]).unwrap();
    zip_mut_with(
        &mut dst_left,
        (&field_hi_left, &field_lo_left),
        |r, (&hi, &lo)| *r = (hi - lo) * inv_h,
    )
    .unwrap();
    Ok(())
}

pub(super) fn staggered_backward_y_into<T>(
    field: ArrayView3<T>,
    dst: &mut ArrayViewMut3<'_, T>,
    nx: usize,
    ny: usize,
    nz: usize,
    hy: T,
) -> Result<()>
where
    T: RealField + FloatElement + Copy,
{
    if ny < 2 {
        return Err(LetoError::InvalidInput(
            "StaggeredBackward: need at least 2 points on the differentiated axis".into(),
        ));
    }
    let inv_h = <T as NumericElement>::ONE / hy;

    let mut dst_int = dst
        .reborrow()
        .slice_mut(&[(0, nx, 1), (1, ny, 1), (0, nz, 1)])
        .unwrap();
    let field_hi = field.slice(&[(0, nx, 1), (1, ny, 1), (0, nz, 1)]).unwrap();
    let field_lo = field
        .slice(&[(0, nx, 1), (0, ny - 1, 1), (0, nz, 1)])
        .unwrap();
    zip_mut_with(&mut dst_int, (&field_hi, &field_lo), |r, (&hi, &lo)| {
        *r = (hi - lo) * inv_h
    })
    .unwrap();
    let mut dst_bot = dst
        .reborrow()
        .slice_mut(&[(0, nx, 1), (0, 1, 1), (0, nz, 1)])
        .unwrap();
    let field_hi_bot = field.slice(&[(0, nx, 1), (1, 2, 1), (0, nz, 1)]).unwrap();
    let field_lo_bot = field.slice(&[(0, nx, 1), (0, 1, 1), (0, nz, 1)]).unwrap();
    zip_mut_with(
        &mut dst_bot,
        (&field_hi_bot, &field_lo_bot),
        |r, (&hi, &lo)| *r = (hi - lo) * inv_h,
    )
    .unwrap();
    Ok(())
}

pub(super) fn staggered_backward_z_into<T>(
    field: ArrayView3<T>,
    dst: &mut ArrayViewMut3<'_, T>,
    nx: usize,
    ny: usize,
    nz: usize,
    hz: T,
) -> Result<()>
where
    T: RealField + FloatElement + Copy,
{
    if nz < 2 {
        return Err(LetoError::InvalidInput(
            "StaggeredBackward: need at least 2 points on the differentiated axis".into(),
        ));
    }
    let inv_h = <T as NumericElement>::ONE / hz;

    let mut dst_int = dst
        .reborrow()
        .slice_mut(&[(0, nx, 1), (0, ny, 1), (1, nz, 1)])
        .unwrap();
    let field_hi = field.slice(&[(0, nx, 1), (0, ny, 1), (1, nz, 1)]).unwrap();
    let field_lo = field
        .slice(&[(0, nx, 1), (0, ny, 1), (0, nz - 1, 1)])
        .unwrap();
    zip_mut_with(&mut dst_int, (&field_hi, &field_lo), |r, (&hi, &lo)| {
        *r = (hi - lo) * inv_h
    })
    .unwrap();
    let mut dst_near = dst
        .reborrow()
        .slice_mut(&[(0, nx, 1), (0, ny, 1), (0, 1, 1)])
        .unwrap();
    let field_hi_near = field.slice(&[(0, nx, 1), (0, ny, 1), (1, 2, 1)]).unwrap();
    let field_lo_near = field.slice(&[(0, nx, 1), (0, ny, 1), (0, 1, 1)]).unwrap();
    zip_mut_with(
        &mut dst_near,
        (&field_hi_near, &field_lo_near),
        |r, (&hi, &lo)| *r = (hi - lo) * inv_h,
    )
    .unwrap();
    Ok(())
}
