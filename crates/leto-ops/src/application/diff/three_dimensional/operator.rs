//! The 3-D finite-difference operator: its spacings, and the dispatch from a
//! stencil scheme to the stencil that implements it.
//!
//! The stencils live in [`super::central`], [`super::fourth_order`] and
//! [`super::staggered`]; this
//! drives them and checks the destination shape.

use eunomia::{FloatElement, NumericElement, RealField};
use leto::{ArrayView3, ArrayViewMut3, LetoError, Result};

use super::central::{
    central2_x_into, central2_y_into, central2_z_into, central6_x_into, central6_y_into,
    central6_z_into,
};
use super::fourth_order::{
    central4_divergence_into, central4_into, central4_map_into, central4_map_triple_into,
};
use super::leapfrog::Axis;
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
                central4_into(field, dst, Axis::X, self.dx)
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

    /// Sum the three axis derivatives of three fields into one destination:
    /// `dst = ∂fields[0]/∂x + ∂fields[1]/∂y + ∂fields[2]/∂z`.
    ///
    /// The fused form reads each field once per output lane instead of
    /// writing three per-axis buffers a caller then adds, which is where a
    /// divergence spends its time once the stencils are lane-swept: at 64
    /// cubed it moves 8 MB against the composed form's 20 MB. Values are
    /// bit-identical to applying each axis separately and summing in x, y, z
    /// order.
    ///
    /// # Errors
    /// - [`LetoError::InvalidInput`] when the fields and `dst` do not share
    ///   one shape, or when the scheme is not `CentralFourthOrder`. The other
    ///   schemes have no fused kernel yet; compose [`Self::apply_x_into`] and
    ///   its siblings and sum, or extend `fourth_order` the same way.
    pub fn divergence_into(
        &self,
        fields: [ArrayView3<'_, T>; 3],
        dst: &mut ArrayViewMut3<'_, T>,
    ) -> Result<()> {
        let shape = fields[0].shape();
        for field in &fields[1..] {
            assert_dst_shape(&field.shape(), &shape)?;
        }
        assert_dst_shape(&dst.shape(), &shape)?;
        match self.scheme {
            FiniteDifference3DScheme::CentralFourthOrder => {
                central4_divergence_into(fields, dst, [self.dx, self.dy, self.dz])
            }
            other => Err(LetoError::InvalidInput(format!(
                "divergence_into has no fused kernel for {other:?}; apply each \
                 axis separately and sum"
            ))),
        }
    }

    /// Combine several axis derivatives and pointwise fields into one
    /// destination in a single pass: `dst[i] = combine(derivatives, values)`,
    /// where `derivatives[j]` is `∂terms[j].1/∂terms[j].0` at that lane and
    /// `values[k]` is `pointwise[k]` there.
    ///
    /// [`Self::divergence_into`] is the sum of three axis derivatives; this
    /// serves the combinations that are not a sum. An elastic shear stress
    /// `μ (∂u/∂b + ∂v/∂a)` is `N = 2, M = 1`: it reads two fields and the
    /// scale once per lane, where sweeping each axis into a buffer and
    /// scaling the sum moves twice the traffic. Each derivative is the value
    /// the corresponding [`Self::apply_x_into`] sweep would have written.
    ///
    /// Terms name distinct axes in the cases this serves; a repeated axis is
    /// read twice and both values are handed to `combine`.
    ///
    /// # Errors
    /// - [`LetoError::InvalidInput`] when the fields, the pointwise inputs
    ///   and `dst` do not share one shape, or when the scheme is not
    ///   `CentralFourthOrder`. The other schemes have no fused kernel yet;
    ///   sweep each axis separately and combine, or extend `fourth_order` the
    ///   same way.
    pub fn map_axis_derivatives<const N: usize, const M: usize, F>(
        &self,
        terms: [(Axis, ArrayView3<'_, T>); N],
        pointwise: [ArrayView3<'_, T>; M],
        dst: &mut ArrayViewMut3<'_, T>,
        combine: F,
    ) -> Result<()>
    where
        F: Fn([T; N], [T; M]) -> T + Send + Sync,
    {
        let shape = dst.shape();
        for (_, field) in &terms {
            assert_dst_shape(&field.shape(), &shape)?;
        }
        for field in &pointwise {
            assert_dst_shape(&field.shape(), &shape)?;
        }
        match self.scheme {
            FiniteDifference3DScheme::CentralFourthOrder => central4_map_into(
                terms,
                pointwise,
                dst,
                [self.dx, self.dy, self.dz],
                combine,
            ),
            other => Err(LetoError::InvalidInput(format!(
                "map_axis_derivatives has no fused kernel for {other:?}; sweep                  each axis separately and combine"
            ))),
        }
    }

    /// [`Self::map_axis_derivatives`] writing three destinations from one
    /// derivative pass.
    ///
    /// Where three results read the same axis derivatives -- the diagonal of
    /// an elastic stress tensor reads the same three normal strains -- this
    /// sweeps the stencils once and writes all three, instead of one call per
    /// destination sweeping them again. At 64 cubed that is 16 MB of traffic
    /// against 28 MB for sweeping the derivatives into buffers and combining
    /// them afterwards, and against 36 MB for three separate fused calls.
    ///
    /// Three is the runtime's lockstep split, not a domain limit.
    ///
    /// # Errors
    /// - [`LetoError::InvalidInput`] when the fields, the pointwise inputs
    ///   and the destinations do not share one shape, or when the scheme is
    ///   not `CentralFourthOrder`.
    pub fn map_axis_derivatives_triple<const N: usize, const M: usize, F>(
        &self,
        terms: [(Axis, ArrayView3<'_, T>); N],
        pointwise: [ArrayView3<'_, T>; M],
        dst: [&mut ArrayViewMut3<'_, T>; 3],
        combine: F,
    ) -> Result<()>
    where
        F: Fn([T; N], [T; M]) -> [T; 3] + Send + Sync,
    {
        let shape = dst[0].shape();
        for destination in &dst[1..] {
            assert_dst_shape(&destination.shape(), &shape)?;
        }
        for (_, field) in &terms {
            assert_dst_shape(&field.shape(), &shape)?;
        }
        for field in &pointwise {
            assert_dst_shape(&field.shape(), &shape)?;
        }
        match self.scheme {
            FiniteDifference3DScheme::CentralFourthOrder => central4_map_triple_into(
                terms,
                pointwise,
                dst,
                [self.dx, self.dy, self.dz],
                combine,
            ),
            other => Err(LetoError::InvalidInput(format!(
                "map_axis_derivatives_triple has no fused kernel for {other:?};                  sweep each axis separately and combine"
            ))),
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
                central4_into(field, dst, Axis::Y, self.dy)
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
                central4_into(field, dst, Axis::Z, self.dz)
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
