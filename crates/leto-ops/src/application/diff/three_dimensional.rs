//! Generic 3-D finite-difference operators.
//!
//! SSOT extension of the 1-D [`FiniteDifference`](super::FiniteDifference) and
//! the 2-D [`crate::laplacian_2d_into`] operators to three spatial dimensions.
//! Replaces the per-consumer FD kernels previously duplicated in
//! `kwavers-math`, `cfd-math`, and `helios-imaging`.
//!
//! The provider covers the families the FDTD / acoustic / CFD / RT kernels
//! actually call:
//!
//! | Scheme | Order | Stencil | dst shape on diff axis |
//! |--------|-------|---------|------------------------|
//! | [`FiniteDifference3DScheme::CentralSecondOrder`] | O(Δx²) 3-point | symmetric interior | matches `field` |
//! | [`FiniteDifference3DScheme::CentralFourthOrder`] | O(Δx⁴) 5-point + 2nd/1st fall-back | matches `field` |
//! | [`FiniteDifference3DScheme::CentralSixthOrder`] | O(Δx⁶) 7-point + 4th/2nd/1st fall-back | matches `field` |
//! | [`FiniteDifference3DScheme::StaggeredForward`] | O(Δx) Yee face | one cell smaller |
//! | [`FiniteDifference3DScheme::StaggeredBackward`] | O(Δx) cell-on-integer-grid | matches `field` |
//!
//! All stencils are explicit, allocation-free, and operate on caller-supplied
//! `ArrayView3<T>` slices writing into pre-allocated `&mut Array3<T>` buffers.
//!
//! ```rust,ignore
//! use leto_ops::{FiniteDifference3D, FiniteDifference3DScheme};
//!
//! let op = FiniteDifference3D::central_fourth_order(0.001, 0.001, 0.001)?;
//! let grad_x = op.apply_x(field.view());
//! ```

use eunomia::{FloatElement, NumericElement, RealField};
use leto::{Array3, ArrayView3, LetoError, Result};

use crate::application::zip::zip_mut_with;

#[inline]
fn f<T: FloatElement>(v: f64) -> T {
    T::from_f64(v)
}

/// Stencil family + kernel ordering for [`FiniteDifference3D`].
///
/// Variant naming follows the FDTD / CFD cell-sweep vocabulary. Note that
/// [`Self::StaggeredBackward`] is the kwavers-side convention: dst shape
/// matches `field.shape` (the integer-cell arrangement rather than the
/// half-cell staggered arrangement), and `i=0` falls back to a forward
/// difference. This preserves the kwavers-side Yee-coupling solver contract
/// bit-equivalent to the previous `StaggeredGridOperator::apply_backward_*_into`
/// kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiniteDifference3DScheme {
    /// Second-order central difference `dst = (f[i+1] − f[i−1]) / (2Δ)`.
    CentralSecondOrder,
    /// Fourth-order central `dst = (−f[i+2] + 8f[i+1] − 8f[i−1] + f[i−2]) / (12Δ)`.
    CentralFourthOrder,
    /// Sixth-order central:
    /// `dst = (−f[i+3] + 9f[i+2] − 45f[i+1] + 45f[i−1] − 9f[i−2] + f[i−3]) / (60Δ)`.
    CentralSixthOrder,
    /// Yee staggered forward face derivative:
    /// `dst[i,j,k] = (f[i+1,j,k] − f[i,j,k]) / Δ`. `dst` has one fewer cell on
    /// the differentiated axis.
    StaggeredForward,
    /// Yee coupling-field backward sweep (kwavers-side convention):
    /// `dst[0,j,k] = (f[1,j,k] − f[0,j,k]) / Δ` (forward fall-back at `i=0`),
    /// `dst[i>0,j,k] = (f[i,j,k] − f[i−1,j,k]) / Δ`. `dst` shape matches `field`.
    StaggeredBackward,
}

/// Generic 3-D finite-difference operator acting on `u(x, y, z)`.
#[derive(Debug, Clone, Copy)]
pub struct FiniteDifference3D<T> {
    scheme: FiniteDifference3DScheme,
    dx: T,
    dy: T,
    dz: T,
}

impl<T: RealField + FloatElement + Copy> FiniteDifference3D<T> {
    /// Construct an operator from a stencil scheme and per-axis spacings.
    ///
    /// # Errors
    /// - [`LetoError::InvalidInput`] when any of `dx`, `dy`, `dz` is not
    ///   strictly positive.
    pub fn new(scheme: FiniteDifference3DScheme, dx: T, dy: T, dz: T) -> Result<Self> {
        let zero = <T as NumericElement>::ZERO;
        if dx <= zero || dy <= zero || dz <= zero {
            return Err(LetoError::InvalidInput(
                "FiniteDifference3D: dx, dy, dz must all be strictly positive".into(),
            ));
        }
        Ok(Self { scheme, dx, dy, dz })
    }

    /// 2nd-order central-difference scheme.
    pub fn central_second_order(dx: T, dy: T, dz: T) -> Result<Self> {
        Self::new(FiniteDifference3DScheme::CentralSecondOrder, dx, dy, dz)
    }

    /// 4th-order central-difference scheme.
    pub fn central_fourth_order(dx: T, dy: T, dz: T) -> Result<Self> {
        Self::new(FiniteDifference3DScheme::CentralFourthOrder, dx, dy, dz)
    }

    /// 6th-order central-difference scheme.
    pub fn central_sixth_order(dx: T, dy: T, dz: T) -> Result<Self> {
        Self::new(FiniteDifference3DScheme::CentralSixthOrder, dx, dy, dz)
    }

    /// Yee staggered forward scheme.
    pub fn staggered_forward(dx: T, dy: T, dz: T) -> Result<Self> {
        Self::new(FiniteDifference3DScheme::StaggeredForward, dx, dy, dz)
    }

    /// Yee staggered backward scheme (kwavers-side convention).
    pub fn staggered_backward(dx: T, dy: T, dz: T) -> Result<Self> {
        Self::new(FiniteDifference3DScheme::StaggeredBackward, dx, dy, dz)
    }

    /// Returns the configured stencil scheme.
    #[must_use]
    pub fn scheme(&self) -> FiniteDifference3DScheme {
        self.scheme
    }

    /// Returns the per-axis grid spacing `(dx, dy, dz)`.
    #[must_use]
    pub fn spacing(&self) -> (T, T, T) {
        (self.dx, self.dy, self.dz)
    }

    /// Stencil width = number of grid points used by the interior kernel.
    #[must_use]
    pub fn stencil_width(&self) -> usize {
        match self.scheme {
            FiniteDifference3DScheme::CentralSecondOrder => 3,
            FiniteDifference3DScheme::CentralFourthOrder => 5,
            FiniteDifference3DScheme::CentralSixthOrder => 7,
            FiniteDifference3DScheme::StaggeredForward
            | FiniteDifference3DScheme::StaggeredBackward => 2,
        }
    }

    /// Apply ∂/∂x into a pre-allocated destination.
    ///
    /// # Errors
    /// - [`LetoError::InvalidInput`] when the diff axis has fewer than the
    ///   minimum required points for the chosen scheme, or when the dst shape
    ///   does not match the scheme's documented contract.
    pub fn apply_x_into(&self, field: ArrayView3<T>, dst: &mut Array3<T>) -> Result<()> {
        let [nx, ny, nz] = field.shape();
        match self.scheme {
            FiniteDifference3DScheme::CentralSecondOrder => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                central2_x_into(field, dst, nx, ny, nz, self.dx)
            }
            FiniteDifference3DScheme::CentralFourthOrder => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                central4_x_into(field, dst, nx, ny, nz, self.dx)
            }
            FiniteDifference3DScheme::CentralSixthOrder => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                central6_x_into(field, dst, nx, ny, nz, self.dx)
            }
            FiniteDifference3DScheme::StaggeredForward => {
                assert_dst_shape(&dst.shape(), &[nx - 1, ny, nz])?;
                staggered_forward_x_into(field, dst, nx, ny, nz, self.dx)
            }
            FiniteDifference3DScheme::StaggeredBackward => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                staggered_backward_x_into(field, dst, nx, ny, nz, self.dx)
            }
        }
    }

    /// Apply ∂/∂y into a pre-allocated destination.
    /// # Errors
    /// See [`Self::apply_x_into`].
    pub fn apply_y_into(&self, field: ArrayView3<T>, dst: &mut Array3<T>) -> Result<()> {
        let [nx, ny, nz] = field.shape();
        match self.scheme {
            FiniteDifference3DScheme::CentralSecondOrder => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                central2_y_into(field, dst, nx, ny, nz, self.dy)
            }
            FiniteDifference3DScheme::CentralFourthOrder => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                central4_y_into(field, dst, nx, ny, nz, self.dy)
            }
            FiniteDifference3DScheme::CentralSixthOrder => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                central6_y_into(field, dst, nx, ny, nz, self.dy)
            }
            FiniteDifference3DScheme::StaggeredForward => {
                assert_dst_shape(&dst.shape(), &[nx, ny - 1, nz])?;
                staggered_forward_y_into(field, dst, nx, ny, nz, self.dy)
            }
            FiniteDifference3DScheme::StaggeredBackward => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                staggered_backward_y_into(field, dst, nx, ny, nz, self.dy)
            }
        }
    }

    /// Apply ∂/∂z into a pre-allocated destination.
    /// # Errors
    /// See [`Self::apply_x_into`].
    pub fn apply_z_into(&self, field: ArrayView3<T>, dst: &mut Array3<T>) -> Result<()> {
        let [nx, ny, nz] = field.shape();
        match self.scheme {
            FiniteDifference3DScheme::CentralSecondOrder => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                central2_z_into(field, dst, nx, ny, nz, self.dz)
            }
            FiniteDifference3DScheme::CentralFourthOrder => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                central4_z_into(field, dst, nx, ny, nz, self.dz)
            }
            FiniteDifference3DScheme::CentralSixthOrder => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                central6_z_into(field, dst, nx, ny, nz, self.dz)
            }
            FiniteDifference3DScheme::StaggeredForward => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz - 1])?;
                staggered_forward_z_into(field, dst, nx, ny, nz, self.dz)
            }
            FiniteDifference3DScheme::StaggeredBackward => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                staggered_backward_z_into(field, dst, nx, ny, nz, self.dz)
            }
        }
    }
}

/// Runtime dst-shape check (errors on mismatch, not debug-only).
#[inline]
fn assert_dst_shape(actual: &[usize], expected: &[usize]) -> Result<()> {
    if actual != expected {
        return Err(LetoError::InvalidInput(format!(
            "FiniteDifference3D: dst shape {actual:?} does not match expected {expected:?}"
        )));
    }
    Ok(())
}

// ── Central 2nd-order ─────────────────────────────────────────────────────────

fn central2_x_into<T>(
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

fn central2_y_into<T>(
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

fn central2_z_into<T>(
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

fn central4_x_into<T>(
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

fn central4_y_into<T>(
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

fn central4_z_into<T>(
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

fn central6_x_into<T>(
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

fn central6_y_into<T>(
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

fn central6_z_into<T>(
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

// ── Staggered forward (Yee face forward) ─────────────────────────────────────

fn staggered_forward_x_into<T>(
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
    if nx < 2 {
        return Err(LetoError::InvalidInput(
            "StaggeredForward: need at least 2 points on the differentiated axis".into(),
        ));
    }
    let inv_h = <T as NumericElement>::ONE / hx;
    let mut dst_slice = dst
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

fn staggered_forward_y_into<T>(
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
    if ny < 2 {
        return Err(LetoError::InvalidInput(
            "StaggeredForward: need at least 2 points on the differentiated axis".into(),
        ));
    }
    let inv_h = <T as NumericElement>::ONE / hy;
    let mut dst_slice = dst
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

fn staggered_forward_z_into<T>(
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
    if nz < 2 {
        return Err(LetoError::InvalidInput(
            "StaggeredForward: need at least 2 points on the differentiated axis".into(),
        ));
    }
    let inv_h = <T as NumericElement>::ONE / hz;
    let mut dst_slice = dst
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

fn staggered_backward_x_into<T>(
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
    if nx < 2 {
        return Err(LetoError::InvalidInput(
            "StaggeredBackward: need at least 2 points on the differentiated axis".into(),
        ));
    }
    let inv_h = <T as NumericElement>::ONE / hx;

    // Backward on i ∈ [1, nx-1]
    let mut dst_int = dst
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
    let mut dst_left = dst.slice_mut(&[(0, 1, 1), (0, ny, 1), (0, nz, 1)]).unwrap();
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

fn staggered_backward_y_into<T>(
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
    if ny < 2 {
        return Err(LetoError::InvalidInput(
            "StaggeredBackward: need at least 2 points on the differentiated axis".into(),
        ));
    }
    let inv_h = <T as NumericElement>::ONE / hy;

    let mut dst_int = dst
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
    let mut dst_bot = dst.slice_mut(&[(0, nx, 1), (0, 1, 1), (0, nz, 1)]).unwrap();
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

fn staggered_backward_z_into<T>(
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
    if nz < 2 {
        return Err(LetoError::InvalidInput(
            "StaggeredBackward: need at least 2 points on the differentiated axis".into(),
        ));
    }
    let inv_h = <T as NumericElement>::ONE / hz;

    let mut dst_int = dst
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
    let mut dst_near = dst.slice_mut(&[(0, nx, 1), (0, ny, 1), (0, 1, 1)]).unwrap();
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use leto::Array3;

    // ── X-axis parity (central 2nd/4th/6th are exact for linear/quadratic/quartic) ──

    #[test]
    fn central2_x_of_linear_function_is_exact() {
        let dx = 0.1;
        let op = FiniteDifference3D::<f64>::central_second_order(dx, dx, dx).unwrap();
        let mut field = Array3::zeros([20, 5, 5]);
        for i in 0..20 {
            for j in 0..5 {
                for k in 0..5 {
                    field[[i, j, k]] = 3.7 * (i as f64) * dx + 0.2 * (j as f64) * dx;
                }
            }
        }
        let mut g = Array3::zeros([20, 5, 5]);
        op.apply_x_into(field.view(), &mut g).unwrap();
        for i in 0..20 {
            assert!((g[[i, 2, 2]] - 3.7).abs() < 1e-12, "i={i}");
        }
    }

    #[test]
    fn central4_x_of_quadratic_is_exact() {
        let dx = 0.1;
        let op = FiniteDifference3D::<f64>::central_fourth_order(dx, dx, dx).unwrap();
        let mut field = Array3::zeros([20, 5, 5]);
        for i in 0..20 {
            let x = (i as f64) * dx;
            for j in 0..5 {
                for k in 0..5 {
                    field[[i, j, k]] = x * x;
                }
            }
        }
        let mut g = Array3::zeros([20, 5, 5]);
        op.apply_x_into(field.view(), &mut g).unwrap();
        for i in 2..18 {
            let x = (i as f64) * dx;
            assert!((g[[i, 2, 2]] - 2.0 * x).abs() < 1e-12, "i={i}");
        }
    }

    #[test]
    fn central6_x_of_quartic_polynomial_is_exact() {
        // 6th-order central is exact for polynomials up to degree 7. Quartic
        // gives ample interior accuracy to floating-point precision (error
        // ~1e-15). Cubic is also exact for 6th-order; we keep quartic here
        // for forward symmetry with central4_x_of_quadratic.
        let dx = 0.1;
        let op = FiniteDifference3D::<f64>::central_sixth_order(dx, dx, dx).unwrap();
        let mut field = Array3::zeros([20, 5, 5]);
        for i in 0..20 {
            let x = (i as f64) * dx;
            for j in 0..5 {
                for k in 0..5 {
                    field[[i, j, k]] = x.powi(4);
                }
            }
        }
        let mut g = Array3::zeros([20, 5, 5]);
        op.apply_x_into(field.view(), &mut g).unwrap();
        for i in 3..17 {
            let x = (i as f64) * dx;
            let expected = 4.0 * x.powi(3);
            assert!((g[[i, 2, 2]] - expected).abs() < 1e-10, "i={i}");
        }
    }

    // ── Y-axis parity ──────────────────────────────────────────────────────────

    #[test]
    fn central2_y_of_linear_function_is_exact() {
        let dx = 0.1;
        let op = FiniteDifference3D::<f64>::central_second_order(dx, dx, dx).unwrap();
        let mut field = Array3::zeros([5, 20, 5]);
        for i in 0..5 {
            for j in 0..20 {
                for k in 0..5 {
                    field[[i, j, k]] = 2.5 * (j as f64) * dx + 0.7 * (i as f64) * dx;
                }
            }
        }
        let mut g = Array3::zeros([5, 20, 5]);
        op.apply_y_into(field.view(), &mut g).unwrap();
        for j in 0..20 {
            assert!((g[[2, j, 2]] - 2.5).abs() < 1e-12, "j={j}");
        }
    }

    // ── Z-axis parity ──────────────────────────────────────────────────────────

    #[test]
    fn central4_z_of_quadratic_is_exact() {
        let dx = 0.1;
        let op = FiniteDifference3D::<f64>::central_fourth_order(dx, dx, dx).unwrap();
        let mut field = Array3::zeros([5, 5, 20]);
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..20 {
                    let z = (k as f64) * dx;
                    field[[i, j, k]] = z * z;
                }
            }
        }
        let mut g = Array3::zeros([5, 5, 20]);
        op.apply_z_into(field.view(), &mut g).unwrap();
        for k in 2..18 {
            let z = (k as f64) * dx;
            assert!((g[[2, 2, k]] - 2.0 * z).abs() < 1e-12, "k={k}");
        }
    }

    #[test]
    fn central6_z_linear_function_is_exact() {
        let dx = 0.1;
        let op = FiniteDifference3D::<f64>::central_sixth_order(dx, dx, dx).unwrap();
        let mut field = Array3::zeros([5, 5, 20]);
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..20 {
                    let z = (k as f64) * dx;
                    field[[i, j, k]] = 4.0 * z;
                }
            }
        }
        let mut g = Array3::zeros([5, 5, 20]);
        op.apply_z_into(field.view(), &mut g).unwrap();
        for k in 0..20 {
            assert!((g[[2, 2, k]] - 4.0).abs() < 1e-12, "k={k}");
        }
    }

    // ── Boundary fall-back parity (central_6 multi-order) ─────────────────────
    //
    // We use a quintic `u = x⁵` field to make the fall-back error visible.
    // - 4th-order central stencil applied to x⁵ introduces O(h⁴) error from
    //   the f^{(5)} = 120 term (error coefficient 4·h⁴ ≈ 4·1e-4 = 4e-4).
    // - 2nd-order stencil applied to x⁵ has O(h²) error (much larger).
    // - 1st-order one-sided at the boundary has O(h) error (largest).
    //
    // The test asserts the 4th-order fall-back at i=2 lands in the 4·h⁴ ≈ 4e-4
    // window, distinct from exact and from the O(h²) baseline.

    #[test]
    fn central6_x_boundary_fall_back_orders() {
        let dx = 0.1;
        let op = FiniteDifference3D::<f64>::central_sixth_order(dx, dx, dx).unwrap();
        let nx = 12_usize;
        let mut field = Array3::zeros([nx, 5, 5]);
        for i in 0..nx {
            let x = (i as f64) * dx;
            for j in 0..5 {
                for k in 0..5 {
                    field[[i, j, k]] = x.powi(5);
                }
            }
        }
        let mut g = Array3::zeros([nx, 5, 5]);
        op.apply_x_into(field.view(), &mut g).unwrap();

        // Interior cell i=6 — 6th-order central, exact for quintic (degree ≤ 7).
        let x_interior = 6.0 * dx;
        let exact_interior: f64 = 5.0 * x_interior.powi(4);
        assert!(
            (g[[interior_i(6), 2, 2]] - exact_interior).abs() < 1e-10,
            "interior mismatch"
        );

        // Near-boundary i=2 — 4th-order fall-back. O(h⁴) error ≈ 4·h⁴ = 4e-4.
        let x_near = 2.0 * dx;
        let exact_near: f64 = 5.0 * x_near.powi(4);
        let err_near = (g[[2, 2, 2]] - exact_near).abs();
        let err_coefficient = err_near / dx.powi(4);
        assert!(
            err_coefficient > 1.0 && err_coefficient < 10.0,
            "4th-order fall-back O(h⁴) error coefficient out of expected band: {err_coefficient}"
        );

        // Near-boundary i=1 — 2nd-order fall-back. O(h²) error ≈ C·h².
        let x_nb1 = 1.0 * dx;
        let exact_nb1: f64 = 5.0 * x_nb1.powi(4);
        let err_nb1 = (g[[1, 2, 2]] - exact_nb1).abs();
        let err_coefficient_2nd = err_nb1 / dx.powi(2);
        assert!(
            err_coefficient_2nd > 0.1,
            "2nd-order fall-back O(h²) error coefficient too small: {err_coefficient_2nd}"
        );

        // Boundary i=0 — 1st-order one-sided forward. For u = x⁵, returns
        // `(f(x+h) - f(x)) / h` at the origin: `(dx⁵ − 0) / dx = dx⁴ = 1e-4`.
        let computed_boundary = (field[[1, 2, 2]] - field[[0, 2, 2]]) / dx;
        assert!(
            (g[[0, 2, 2]] - computed_boundary).abs() < 1e-12,
            "1st-order forward fall-back must return (f[1]-f[0])/dx at i=0"
        );
        // Exact derivative at the origin is 0; the 1st-order value is 1e-4.
        assert!(
            (g[[0, 2, 2]] - 1e-4).abs() < 1e-12,
            "quintic 1st-order @ origin must equal dx⁴ = 1e-4"
        );
    }

    fn interior_i(_idx: usize) -> usize {
        6
    }

    // ── Staggered schemes (X + Y + Z shapes) ───────────────────────────────────

    #[test]
    fn staggered_forward_x_face_centered() {
        let dx = 0.1;
        let op = FiniteDifference3D::<f64>::staggered_forward(dx, dx, dx).unwrap();
        let field = Array3::from_elem((10, 5, 5), 0.0_f64);
        let mut g = Array3::zeros([9, 5, 5]);
        op.apply_x_into(field.view(), &mut g).unwrap();
        for v in g.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn staggered_backward_x_zero_field() {
        let dx = 0.1;
        let op = FiniteDifference3D::<f64>::staggered_backward(dx, dx, dx).unwrap();
        let field = Array3::from_elem((10, 5, 5), 0.0_f64);
        let mut g = Array3::zeros([10, 5, 5]);
        op.apply_x_into(field.view(), &mut g).unwrap();
        for v in g.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn staggered_backward_z_mixed_dst_shape() {
        let dx = 0.1;
        let op = FiniteDifference3D::<f64>::staggered_backward(dx, dx, dx).unwrap();
        let field = Array3::from_elem((5, 5, 10), 0.0_f64);
        let mut g = Array3::zeros([5, 5, 10]);
        op.apply_z_into(field.view(), &mut g).unwrap();
        assert_eq!(g.shape(), [5, 5, 10]);
        for v in g.iter() {
            assert_eq!(*v, 0.0);
        }
        // Linear field along z: dst at i=0 forward, interior backward.
        let mut field = Array3::from_elem((5, 5, 10), 0.0_f64);
        for k in 0..10 {
            for i in 0..5 {
                for j in 0..5 {
                    field[[i, j, k]] = (k as f64) * dx;
                }
            }
        }
        let mut g = Array3::zeros([5, 5, 10]);
        op.apply_z_into(field.view(), &mut g).unwrap();
        assert!((g[[2, 2, 0]] - 1.0).abs() < 1e-12); // forward fall-back
        for k in 1..10 {
            assert!((g[[2, 2, k]] - 1.0).abs() < 1e-12, "k={k}");
        }
    }

    // ── dst shape contract (release-mode safety) ──────────────────────────────

    #[test]
    fn staggered_forward_rejects_dst_shape_mismatch() {
        let dx = 0.1;
        let op = FiniteDifference3D::<f64>::staggered_forward(dx, dx, dx).unwrap();
        let field = Array3::from_elem((10, 5, 5), 0.0_f64);
        let mut g = Array3::zeros([10, 5, 5]); // wrong: should be [9, 5, 5]
        assert!(op.apply_x_into(field.view(), &mut g).is_err());
    }

    #[test]
    fn central4_rejects_dst_shape_mismatch() {
        let dx = 0.1;
        let op = FiniteDifference3D::<f64>::central_fourth_order(dx, dx, dx).unwrap();
        let field = Array3::from_elem((10, 5, 5), 0.0_f64);
        let mut g = Array3::zeros([9, 5, 5]); // wrong: should be [10, 5, 5]
        assert!(op.apply_x_into(field.view(), &mut g).is_err());
    }

    // ── Dispersion ordering & spacing rejection (regression suite) ────────────

    #[test]
    fn dispersion_ordering_central_2_4_6() {
        let dx = 0.1;
        let lambda = 2.0;
        let k = 2.0 * std::f64::consts::PI / lambda;
        let n = 80_usize;
        let mut field = Array3::zeros([n, 5, 5]);
        for i in 0..n {
            let x = (i as f64) * dx;
            for j in 0..5 {
                for kk in 0..5 {
                    field[[i, j, kk]] = (k * x).sin();
                }
            }
        }
        let op2 = FiniteDifference3D::<f64>::central_second_order(dx, dx, dx).unwrap();
        let op4 = FiniteDifference3D::<f64>::central_fourth_order(dx, dx, dx).unwrap();
        let op6 = FiniteDifference3D::<f64>::central_sixth_order(dx, dx, dx).unwrap();

        let mut g2 = Array3::zeros([n, 5, 5]);
        let mut g4 = Array3::zeros([n, 5, 5]);
        let mut g6 = Array3::zeros([n, 5, 5]);
        op2.apply_x_into(field.view(), &mut g2).unwrap();
        op4.apply_x_into(field.view(), &mut g4).unwrap();
        op6.apply_x_into(field.view(), &mut g6).unwrap();

        let mut err2 = 0.0;
        let mut err4 = 0.0;
        let mut err6 = 0.0;
        let mut count = 0_usize;
        for i in 10..(n - 10) {
            let x = (i as f64) * dx;
            let exact = k * (k * x).cos();
            err2 += (g2[[i, 2, 2]] - exact).abs();
            err4 += (g4[[i, 2, 2]] - exact).abs();
            err6 += (g6[[i, 2, 2]] - exact).abs();
            count += 1;
        }
        let count = count as f64;
        err2 /= count;
        err4 /= count;
        err6 /= count;
        assert!(
            err4 < err2,
            "4th-order should be more accurate than 2nd-order: err4={err4}, err2={err2}"
        );
        assert!(
            err6 < err4,
            "6th-order should be more accurate than 4th-order: err6={err6}, err4={err4}"
        );
    }

    #[test]
    fn rejects_non_positive_spacing() {
        assert!(FiniteDifference3D::<f64>::central_second_order(0.0, 0.1, 0.1).is_err());
        assert!(FiniteDifference3D::<f64>::central_fourth_order(0.1, -0.1, 0.1).is_err());
        assert!(FiniteDifference3D::<f64>::central_sixth_order(0.1, 0.1, 0.0).is_err());
        assert!(FiniteDifference3D::<f64>::staggered_forward(0.0, 0.1, 0.1).is_err());
        assert!(FiniteDifference3D::<f64>::staggered_backward(0.1, 0.1, 0.0).is_err());
    }

    #[test]
    fn rejects_too_few_points() {
        let dx = 0.1;
        let op = FiniteDifference3D::<f64>::central_sixth_order(dx, dx, dx).unwrap();
        let small = Array3::zeros([6, 10, 10]);
        let mut g = Array3::zeros([6, 10, 10]);
        assert!(op.apply_x_into(small.view(), &mut g).is_err());
    }

    #[test]
    fn stencil_width_matches_scheme() {
        let dx = 0.1;
        assert_eq!(
            FiniteDifference3D::<f64>::central_second_order(dx, dx, dx)
                .unwrap()
                .stencil_width(),
            3
        );
        assert_eq!(
            FiniteDifference3D::<f64>::central_fourth_order(dx, dx, dx)
                .unwrap()
                .stencil_width(),
            5
        );
        assert_eq!(
            FiniteDifference3D::<f64>::central_sixth_order(dx, dx, dx)
                .unwrap()
                .stencil_width(),
            7
        );
        assert_eq!(
            FiniteDifference3D::<f64>::staggered_forward(dx, dx, dx)
                .unwrap()
                .stencil_width(),
            2
        );
        assert_eq!(
            FiniteDifference3D::<f64>::staggered_backward(dx, dx, dx)
                .unwrap()
                .stencil_width(),
            2
        );
    }
}
