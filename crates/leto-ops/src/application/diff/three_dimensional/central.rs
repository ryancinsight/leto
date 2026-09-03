//! Central-difference stencil kernels (2nd, 4th, and 6th order with
//! boundary fall-back) behind [`super::FiniteDifference3D`].
use eunomia::{FloatElement, NumericElement, RealField};
use leto::{Array3, ArrayView3, LetoError, Result};

use crate::application::zip::zip_mut_with;

use super::f;

// ── Central 2nd-order ─────────────────────────────────────────────────────────

pub(super) fn central2_x_into<T>(
    field: ArrayView3<T>,
    dst: &mut Array3<T>,
    nx: usize,
    ny: usize,
    nz: usize,
    hx: T,
) -> Result<()>
where
    T: RealField + FloatElement + Copy,
{
    if nx < 3 {
        return Err(LetoError::InvalidInput(
            "CentralSecondOrder: need at least 3 points on the differentiated axis".into(),
        ));
    }
    let two = f::<T>(2.0);
    let inv_2h = <T as NumericElement>::ONE / (two * hx);
    let inv_h = <T as NumericElement>::ONE / hx;

    // Interior: central difference via leto slice pair (contiguous on innermost Z).
    let mut dst_int = dst
        .slice_mut(&[(1, nx - 1, 1), (0, ny, 1), (0, nz, 1)])
        .unwrap();
    let field_hi = field.slice(&[(2, nx, 1), (0, ny, 1), (0, nz, 1)]).unwrap();
    let field_lo = field
        .slice(&[(0, nx - 2, 1), (0, ny, 1), (0, nz, 1)])
        .unwrap();
    zip_mut_with(&mut dst_int, (&field_hi, &field_lo), |r, (&hi, &lo)| {
        *r = (hi - lo) * inv_2h
    })
    .unwrap();

    // Boundaries: forward / backward one-sided.
    let mut dst_left = dst.slice_mut(&[(0, 1, 1), (0, ny, 1), (0, nz, 1)]).unwrap();
    let field_hi_left = field.slice(&[(1, 2, 1), (0, ny, 1), (0, nz, 1)]).unwrap();
    let field_lo_left = field.slice(&[(0, 1, 1), (0, ny, 1), (0, nz, 1)]).unwrap();
    zip_mut_with(
        &mut dst_left,
        (&field_hi_left, &field_lo_left),
        |r, (&hi, &lo)| *r = (hi - lo) * inv_h,
    )
    .unwrap();
    let mut dst_right = dst
        .slice_mut(&[(nx - 1, nx, 1), (0, ny, 1), (0, nz, 1)])
        .unwrap();
    let field_hi_right = field
        .slice(&[(nx - 1, nx, 1), (0, ny, 1), (0, nz, 1)])
        .unwrap();
    let field_lo_right = field
        .slice(&[(nx - 2, nx - 1, 1), (0, ny, 1), (0, nz, 1)])
        .unwrap();
    zip_mut_with(
        &mut dst_right,
        (&field_hi_right, &field_lo_right),
        |r, (&hi, &lo)| *r = (hi - lo) * inv_h,
    )
    .unwrap();
    Ok(())
}

pub(super) fn central2_y_into<T>(
    field: ArrayView3<T>,
    dst: &mut Array3<T>,
    nx: usize,
    ny: usize,
    nz: usize,
    hy: T,
) -> Result<()>
where
    T: RealField + FloatElement + Copy,
{
    if ny < 3 {
        return Err(LetoError::InvalidInput(
            "CentralSecondOrder: need at least 3 points on the differentiated axis".into(),
        ));
    }
    let two = f::<T>(2.0);
    let inv_2h = <T as NumericElement>::ONE / (two * hy);
    let inv_h = <T as NumericElement>::ONE / hy;

    let mut dst_int = dst
        .slice_mut(&[(0, nx, 1), (1, ny - 1, 1), (0, nz, 1)])
        .unwrap();
    let field_hi = field.slice(&[(0, nx, 1), (2, ny, 1), (0, nz, 1)]).unwrap();
    let field_lo = field
        .slice(&[(0, nx, 1), (0, ny - 2, 1), (0, nz, 1)])
        .unwrap();
    zip_mut_with(&mut dst_int, (&field_hi, &field_lo), |r, (&hi, &lo)| {
        *r = (hi - lo) * inv_2h
    })
    .unwrap();
    let mut dst_bot = dst.slice_mut(&[(0, nx, 1), (0, 1, 1), (0, nz, 1)]).unwrap();
    let field_hi_bot = field.slice(&[(0, nx, 1), (1, 2, 1), (0, nz, 1)]).unwrap();
    let field_lo_bot = field.slice(&[(0, nx, 1), (0, 1, 1), (0, nz, 1)]).unwrap();
    zip_mut_with(
        &mut dst_bot,
        (&field_hi_bot, &field_lo_bot),
        |r, (&hi, &lo)| *r = (hi - lo) * inv_h,
    )
    .unwrap();
    let mut dst_top = dst
        .slice_mut(&[(0, nx, 1), (ny - 1, ny, 1), (0, nz, 1)])
        .unwrap();
    let field_hi_top = field
        .slice(&[(0, nx, 1), (ny - 1, ny, 1), (0, nz, 1)])
        .unwrap();
    let field_lo_top = field
        .slice(&[(0, nx, 1), (ny - 2, ny - 1, 1), (0, nz, 1)])
        .unwrap();
    zip_mut_with(
        &mut dst_top,
        (&field_hi_top, &field_lo_top),
        |r, (&hi, &lo)| *r = (hi - lo) * inv_h,
    )
    .unwrap();
    Ok(())
}

pub(super) fn central2_z_into<T>(
    field: ArrayView3<T>,
    dst: &mut Array3<T>,
    nx: usize,
    ny: usize,
    nz: usize,
    hz: T,
) -> Result<()>
where
    T: RealField + FloatElement + Copy,
{
    if nz < 3 {
        return Err(LetoError::InvalidInput(
            "CentralSecondOrder: need at least 3 points on the differentiated axis".into(),
        ));
    }
    let two = f::<T>(2.0);
    let inv_2h = <T as NumericElement>::ONE / (two * hz);
    let inv_h = <T as NumericElement>::ONE / hz;

    let mut dst_int = dst
        .slice_mut(&[(0, nx, 1), (0, ny, 1), (1, nz - 1, 1)])
        .unwrap();
    let field_hi = field.slice(&[(0, nx, 1), (0, ny, 1), (2, nz, 1)]).unwrap();
    let field_lo = field
        .slice(&[(0, nx, 1), (0, ny, 1), (0, nz - 2, 1)])
        .unwrap();
    zip_mut_with(&mut dst_int, (&field_hi, &field_lo), |r, (&hi, &lo)| {
        *r = (hi - lo) * inv_2h
    })
    .unwrap();
    let mut dst_near = dst.slice_mut(&[(0, nx, 1), (0, ny, 1), (0, 1, 1)]).unwrap();
    let field_hi_near = field.slice(&[(0, nx, 1), (0, ny, 1), (1, 2, 1)]).unwrap();
    let field_lo_near = field.slice(&[(0, nx, 1), (0, ny, 1), (0, 1, 1)]).unwrap();
    zip_mut_with(
        &mut dst_near,
        (&field_hi_near, &field_lo_near),
        |r, (&hi, &lo)| *r = (hi - lo) * inv_h,
    )
    .unwrap();
    let mut dst_far = dst
        .slice_mut(&[(0, nx, 1), (0, ny, 1), (nz - 1, nz, 1)])
        .unwrap();
    let field_hi_far = field
        .slice(&[(0, nx, 1), (0, ny, 1), (nz - 1, nz, 1)])
        .unwrap();
    let field_lo_far = field
        .slice(&[(0, nx, 1), (0, ny, 1), (nz - 2, nz - 1, 1)])
        .unwrap();
    zip_mut_with(
        &mut dst_far,
        (&field_hi_far, &field_lo_far),
        |r, (&hi, &lo)| *r = (hi - lo) * inv_h,
    )
    .unwrap();
    Ok(())
}

// ── Central 4th-order ─────────────────────────────────────────────────────────
//
// Arithmetic uses plain `(+)(-)(*)` chains rather than `mul_add` so the
// implementation compiles against the leaner `eunomia::FloatElement` trait
// surface, which does not expose `mul_add` as a method on `T`. The
// `mul_add(a, b) = a*b+c` chain expands to `(a * b) + c`, so the expansion
// is exact and the compiler still folds the multiply-add pattern at -Copt.

pub(super) fn central4_x_into<T>(
    field: ArrayView3<T>,
    dst: &mut Array3<T>,
    nx: usize,
    ny: usize,
    nz: usize,
    hx: T,
) -> Result<()>
where
    T: RealField + FloatElement + Copy,
{
    if nx < 5 {
        return Err(LetoError::InvalidInput(
            "CentralFourthOrder: need at least 5 points on the differentiated axis".into(),
        ));
    }
    let twelve = f::<T>(12.0);
    let two = f::<T>(2.0);
    let inv_12h = <T as NumericElement>::ONE / (twelve * hx);
    let inv_2h = <T as NumericElement>::ONE / (two * hx);
    let inv_h = <T as NumericElement>::ONE / hx;

    // Interior 5-point stencil — expand to plain arithmetic chain.
    //
    // The kwavers-side formula is `(-f[i+2] + 8·f[i+1] - 8·f[i-1] + f[i-2]) / 12h`,
    // which the kwavers code wrote as `8.0.mul_add(-f[i-1], 8.0.mul_add(f[i+1], -f[i+2])) + f[i-2]`.
    // We expand that fused-multiply-add chain into plain arith.
    let eight = f::<T>(8.0);
    for i in 2..nx - 2 {
        for j in 0..ny {
            for k in 0..nz {
                dst[[i, j, k]] = ((-eight * field[[i - 1, j, k]])
                    + (eight * field[[i + 1, j, k]])
                    + (-field[[i + 2, j, k]])
                    + field[[i - 2, j, k]])
                    * inv_12h;
            }
        }
    }
    // Near-boundary: 2nd-order central; boundaries: 1st-order one-sided.
    for j in 0..ny {
        for k in 0..nz {
            dst[[1, j, k]] = (field[[2, j, k]] - field[[0, j, k]]) * inv_2h;
            dst[[nx - 2, j, k]] = (field[[nx - 1, j, k]] - field[[nx - 3, j, k]]) * inv_2h;
            dst[[0, j, k]] = (field[[1, j, k]] - field[[0, j, k]]) * inv_h;
            dst[[nx - 1, j, k]] = (field[[nx - 1, j, k]] - field[[nx - 2, j, k]]) * inv_h;
        }
    }
    Ok(())
}

pub(super) fn central4_y_into<T>(
    field: ArrayView3<T>,
    dst: &mut Array3<T>,
    nx: usize,
    ny: usize,
    nz: usize,
    hy: T,
) -> Result<()>
where
    T: RealField + FloatElement + Copy,
{
    if ny < 5 {
        return Err(LetoError::InvalidInput(
            "CentralFourthOrder: need at least 5 points on the differentiated axis".into(),
        ));
    }
    let twelve = f::<T>(12.0);
    let two = f::<T>(2.0);
    let inv_12h = <T as NumericElement>::ONE / (twelve * hy);
    let inv_2h = <T as NumericElement>::ONE / (two * hy);
    let inv_h = <T as NumericElement>::ONE / hy;

    let eight = f::<T>(8.0);
    for i in 0..nx {
        for j in 2..ny - 2 {
            for k in 0..nz {
                dst[[i, j, k]] = ((-eight * field[[i, j - 1, k]])
                    + (eight * field[[i, j + 1, k]])
                    + (-field[[i, j + 2, k]])
                    + field[[i, j - 2, k]])
                    * inv_12h;
            }
        }
    }
    for i in 0..nx {
        for k in 0..nz {
            dst[[i, 1, k]] = (field[[i, 2, k]] - field[[i, 0, k]]) * inv_2h;
            dst[[i, ny - 2, k]] = (field[[i, ny - 1, k]] - field[[i, ny - 3, k]]) * inv_2h;
            dst[[i, 0, k]] = (field[[i, 1, k]] - field[[i, 0, k]]) * inv_h;
            dst[[i, ny - 1, k]] = (field[[i, ny - 1, k]] - field[[i, ny - 2, k]]) * inv_h;
        }
    }
    Ok(())
}

pub(super) fn central4_z_into<T>(
    field: ArrayView3<T>,
    dst: &mut Array3<T>,
    nx: usize,
    ny: usize,
    nz: usize,
    hz: T,
) -> Result<()>
where
    T: RealField + FloatElement + Copy,
{
    if nz < 5 {
        return Err(LetoError::InvalidInput(
            "CentralFourthOrder: need at least 5 points on the differentiated axis".into(),
        ));
    }
    let twelve = f::<T>(12.0);
    let two = f::<T>(2.0);
    let inv_12h = <T as NumericElement>::ONE / (twelve * hz);
    let inv_2h = <T as NumericElement>::ONE / (two * hz);
    let inv_h = <T as NumericElement>::ONE / hz;

    let eight = f::<T>(8.0);
    for i in 0..nx {
        for j in 0..ny {
            for k in 2..nz - 2 {
                dst[[i, j, k]] = ((-eight * field[[i, j, k - 1]])
                    + (eight * field[[i, j, k + 1]])
                    + (-field[[i, j, k + 2]])
                    + field[[i, j, k - 2]])
                    * inv_12h;
            }
        }
    }
    for i in 0..nx {
        for j in 0..ny {
            dst[[i, j, 1]] = (field[[i, j, 2]] - field[[i, j, 0]]) * inv_2h;
            dst[[i, j, nz - 2]] = (field[[i, j, nz - 1]] - field[[i, j, nz - 3]]) * inv_2h;
            dst[[i, j, 0]] = (field[[i, j, 1]] - field[[i, j, 0]]) * inv_h;
            dst[[i, j, nz - 1]] = (field[[i, j, nz - 1]] - field[[i, j, nz - 2]]) * inv_h;
        }
    }
    Ok(())
}

// ── Central 6th-order ─────────────────────────────────────────────────────────

pub(super) fn central6_x_into<T>(
    field: ArrayView3<T>,
    dst: &mut Array3<T>,
    nx: usize,
    ny: usize,
    nz: usize,
    hx: T,
) -> Result<()>
where
    T: RealField + FloatElement + Copy,
{
    if nx < 7 {
        return Err(LetoError::InvalidInput(
            "CentralSixthOrder: need at least 7 points on the differentiated axis".into(),
        ));
    }
    let twelve = f::<T>(12.0);
    let two = f::<T>(2.0);
    let sixty = f::<T>(60.0);
    let inv_60h = <T as NumericElement>::ONE / (sixty * hx);
    let inv_12h = <T as NumericElement>::ONE / (twelve * hx);
    let inv_2h = <T as NumericElement>::ONE / (two * hx);
    let inv_h = <T as NumericElement>::ONE / hx;
    let eight = f::<T>(8.0);
    let nine = f::<T>(9.0);
    let forty_five = f::<T>(45.0);

    // Interior 7-point stencil — expand the kwavers-side fused-multiply-add
    // chain into plain arith: `9·(-f[i+2]) + 45·f[i+1] + 45·(-f[i-1])
    //  + 9·(-f[i-2]) + (-f[i-3]) + f[i+3]`.
    for i in 3..nx - 3 {
        for j in 0..ny {
            for k in 0..nz {
                dst[[i, j, k]] = ((-nine * field[[i + 2, j, k]])
                    + (forty_five * field[[i + 1, j, k]])
                    + (-forty_five * field[[i - 1, j, k]])
                    + (nine * field[[i - 2, j, k]])
                    + (-field[[i - 3, j, k]])
                    + field[[i + 3, j, k]])
                    * inv_60h;
            }
        }
    }
    // Near-boundary fall-back: 4th-order at i=2/nx-3, 2nd-order at i=1/nx-2,
    // 1st-order at i=0/nx-1.
    for j in 0..ny {
        for k in 0..nz {
            // 4th-order at i=2 — same expansion as central4_x_into.
            dst[[2, j, k]] = ((-eight * field[[1, j, k]])
                + (eight * field[[3, j, k]])
                + (-field[[4, j, k]])
                + field[[0, j, k]])
                * inv_12h;
            // 4th-order at i=nx-3.
            dst[[nx - 3, j, k]] = ((-eight * field[[nx - 4, j, k]])
                + (eight * field[[nx - 2, j, k]])
                + (-field[[nx - 1, j, k]])
                + field[[nx - 5, j, k]])
                * inv_12h;
            dst[[1, j, k]] = (field[[2, j, k]] - field[[0, j, k]]) * inv_2h;
            dst[[nx - 2, j, k]] = (field[[nx - 1, j, k]] - field[[nx - 3, j, k]]) * inv_2h;
            dst[[0, j, k]] = (field[[1, j, k]] - field[[0, j, k]]) * inv_h;
            dst[[nx - 1, j, k]] = (field[[nx - 1, j, k]] - field[[nx - 2, j, k]]) * inv_h;
        }
    }
    Ok(())
}

pub(super) fn central6_y_into<T>(
    field: ArrayView3<T>,
    dst: &mut Array3<T>,
    nx: usize,
    ny: usize,
    nz: usize,
    hy: T,
) -> Result<()>
where
    T: RealField + FloatElement + Copy,
{
    if ny < 7 {
        return Err(LetoError::InvalidInput(
            "CentralSixthOrder: need at least 7 points on the differentiated axis".into(),
        ));
    }
    let twelve = f::<T>(12.0);
    let two = f::<T>(2.0);
    let sixty = f::<T>(60.0);
    let inv_60h = <T as NumericElement>::ONE / (sixty * hy);
    let inv_12h = <T as NumericElement>::ONE / (twelve * hy);
    let inv_2h = <T as NumericElement>::ONE / (two * hy);
    let inv_h = <T as NumericElement>::ONE / hy;
    let eight = f::<T>(8.0);
    let nine = f::<T>(9.0);
    let forty_five = f::<T>(45.0);

    for i in 0..nx {
        for j in 3..ny - 3 {
            for k in 0..nz {
                dst[[i, j, k]] = ((-nine * field[[i, j + 2, k]])
                    + (forty_five * field[[i, j + 1, k]])
                    + (-forty_five * field[[i, j - 1, k]])
                    + (nine * field[[i, j - 2, k]])
                    + (-field[[i, j - 3, k]])
                    + field[[i, j + 3, k]])
                    * inv_60h;
            }
        }
    }
    for i in 0..nx {
        for k in 0..nz {
            dst[[i, 2, k]] = ((-eight * field[[i, 1, k]])
                + (eight * field[[i, 3, k]])
                + (-field[[i, 4, k]])
                + field[[i, 0, k]])
                * inv_12h;
            dst[[i, ny - 3, k]] = ((-eight * field[[i, ny - 4, k]])
                + (eight * field[[i, ny - 2, k]])
                + (-field[[i, ny - 1, k]])
                + field[[i, ny - 5, k]])
                * inv_12h;
            dst[[i, 1, k]] = (field[[i, 2, k]] - field[[i, 0, k]]) * inv_2h;
            dst[[i, ny - 2, k]] = (field[[i, ny - 1, k]] - field[[i, ny - 3, k]]) * inv_2h;
            dst[[i, 0, k]] = (field[[i, 1, k]] - field[[i, 0, k]]) * inv_h;
            dst[[i, ny - 1, k]] = (field[[i, ny - 1, k]] - field[[i, ny - 2, k]]) * inv_h;
        }
    }
    Ok(())
}

pub(super) fn central6_z_into<T>(
    field: ArrayView3<T>,
    dst: &mut Array3<T>,
    nx: usize,
    ny: usize,
    nz: usize,
    hz: T,
) -> Result<()>
where
    T: RealField + FloatElement + Copy,
{
    if nz < 7 {
        return Err(LetoError::InvalidInput(
            "CentralSixthOrder: need at least 7 points on the differentiated axis".into(),
        ));
    }
    let twelve = f::<T>(12.0);
    let two = f::<T>(2.0);
    let sixty = f::<T>(60.0);
    let inv_60h = <T as NumericElement>::ONE / (sixty * hz);
    let inv_12h = <T as NumericElement>::ONE / (twelve * hz);
    let inv_2h = <T as NumericElement>::ONE / (two * hz);
    let inv_h = <T as NumericElement>::ONE / hz;
    let eight = f::<T>(8.0);
    let nine = f::<T>(9.0);
    let forty_five = f::<T>(45.0);

    for i in 0..nx {
        for j in 0..ny {
            for k in 3..nz - 3 {
                dst[[i, j, k]] = ((-nine * field[[i, j, k + 2]])
                    + (forty_five * field[[i, j, k + 1]])
                    + (-forty_five * field[[i, j, k - 1]])
                    + (nine * field[[i, j, k - 2]])
                    + (-field[[i, j, k - 3]])
                    + field[[i, j, k + 3]])
                    * inv_60h;
            }
        }
    }
    for i in 0..nx {
        for j in 0..ny {
            dst[[i, j, 2]] = ((-eight * field[[i, j, 1]])
                + (eight * field[[i, j, 3]])
                + (-field[[i, j, 4]])
                + field[[i, j, 0]])
                * inv_12h;
            dst[[i, j, nz - 3]] = ((-eight * field[[i, j, nz - 4]])
                + (eight * field[[i, j, nz - 2]])
                + (-field[[i, j, nz - 1]])
                + field[[i, j, nz - 5]])
                * inv_12h;
            dst[[i, j, 1]] = (field[[i, j, 2]] - field[[i, j, 0]]) * inv_2h;
            dst[[i, j, nz - 2]] = (field[[i, j, nz - 1]] - field[[i, j, nz - 3]]) * inv_2h;
            dst[[i, j, 0]] = (field[[i, j, 1]] - field[[i, j, 0]]) * inv_h;
            dst[[i, j, nz - 1]] = (field[[i, j, nz - 1]] - field[[i, j, nz - 2]]) * inv_h;
        }
    }
    Ok(())
}
