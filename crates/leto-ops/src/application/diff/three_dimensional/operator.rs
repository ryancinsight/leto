//! The 3-D finite-difference operator: its spacings, and the dispatch from a
//! stencil scheme to the stencil that implements it.
//!
//! The stencils live in [`super::central`] and [`super::staggered`]; this
//! drives them and checks the destination shape.

use eunomia::{FloatElement, NumericElement, RealField};
use leto::{ArrayView3, ArrayViewMut3, LetoError, Result};

use super::central::{
    central2_x_into, central2_y_into, central2_z_into, central4_x_into, central4_y_into,
    central4_z_into, central6_x_into, central6_y_into, central6_z_into,
};
use super::staggered::{
    staggered_backward_x_into, staggered_backward_y_into, staggered_backward_z_into,
    staggered_forward_x_into, staggered_forward_y_into, staggered_forward_z_into,
};
use super::FiniteDifference3DScheme;

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
    pub fn apply_x_into(&self, field: ArrayView3<T>, dst: &mut ArrayViewMut3<'_, T>) -> Result<()> {
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
    pub fn apply_y_into(&self, field: ArrayView3<T>, dst: &mut ArrayViewMut3<'_, T>) -> Result<()> {
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
    pub fn apply_z_into(&self, field: ArrayView3<T>, dst: &mut ArrayViewMut3<'_, T>) -> Result<()> {
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
